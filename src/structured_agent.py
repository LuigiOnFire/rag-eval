"""
Structured RAG Agent with Decompose-Retrieve-Reflect-Generate architecture.

Key insight: The agent must CHECK if retrieved context actually answers
the sub-question before moving on. One refinement retry per sub-query.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SubQuery:
    """A decomposed sub-question."""
    question: str
    depends_on: Optional[int] = None  # Index of sub-query this depends on
    answer: Optional[str] = None
    context: List[Dict] = field(default_factory=list)


@dataclass 
class Step:
    """A single step in the trajectory."""
    action: str
    input: str
    output: str
    

@dataclass
class TrajectoryResult:
    """Result of running the agent."""
    question: str
    sub_queries: List[SubQuery]
    steps: List[Step]
    final_answer: str
    success: bool


class StructuredAgent:
    """
    Structured agent with Context-Aware Oracle loop:
    1. DECOMPOSE: Break question into sub-queries
    2. For each sub-query:
       a. REWRITE_WITH_CONTEXT - Replace pronouns with actual entities
       b. IS_RETRIEVAL_NECESSARY - Skip retrieval for synthesis steps
       c. RETRIEVE + IS_RELEVANT filter
       d. REFINE if no relevant docs
       e. EXTRACT partial answer
    3. GENERATE: Combine partial answers into final answer
    """
    
    def __init__(self, llm, retriever, max_docs_per_query: int = 5):
        """
        Args:
            llm: Language model with .generate(prompt) -> str
            retriever: Retriever with .search(query, top_k) -> List[Dict]
            max_docs_per_query: Max documents to retrieve per query
        """
        self.llm = llm
        self.retriever = retriever
        self.max_docs = max_docs_per_query
        self.steps: List[Step] = []
        
    def run(self, question: str) -> TrajectoryResult:
        """Run the full pipeline on a question."""
        self.steps = []
        history = {}  # {sub_query: answer}
        accumulated_context = []
        
        # Step 1: Decompose
        sub_queries = self._decompose(question)
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
        
        # Step 2: Process each sub-query
        for i, sq in enumerate(sub_queries):
            # FIX 1: Rewrite with context (Shirley Temple fix)
            executable_query = self._rewrite_with_context(sq.question, history)
            logger.info(f"Sub-query {i+1}: {sq.question} → {executable_query}")
            
            # FIX 2: Check if retrieval is necessary (Ed Wood fix)
            if self._is_retrieval_necessary(executable_query):
                # Retrieve
                docs = self._retrieve(executable_query)
                
                # FIX 3: Filter for relevance (CRAG-style)
                valid_docs = self._filter_relevant(docs, executable_query)
                
                if not valid_docs:
                    # Refine and retry
                    refined_query = self._refine_query_simple(executable_query)
                    new_docs = self._retrieve(refined_query)
                    valid_docs = self._filter_relevant(new_docs, executable_query)
                
                # Add to context (deduplicate by title)
                seen_titles = {d.get('title') for d in accumulated_context}
                for d in valid_docs:
                    if d.get('title') not in seen_titles:
                        accumulated_context.append(d)
                        seen_titles.add(d.get('title'))
                
                sq.context = valid_docs
            else:
                self.steps.append(Step("SKIP_RETRIEVAL", executable_query, "Synthesis step - no retrieval needed"))
            
            # Extract answer using accumulated context
            sq.answer = self._extract_answer(executable_query, accumulated_context, history)
            history[sq.question] = sq.answer
            logger.info(f"Answer {i+1}: {sq.answer[:100]}...")
        
        # Step 3: Generate final answer (no retrieval)
        final_answer = self._generate_final(question, history, accumulated_context)
        
        return TrajectoryResult(
            question=question,
            sub_queries=sub_queries,
            steps=self.steps,
            final_answer=final_answer,
            success=True
        )
    
    def _decompose(self, question: str) -> List[SubQuery]:
        """Decompose question into sub-queries. Limit to 2-3 max."""
        prompt = f"""Break down this question into exactly 2 simple factual sub-questions.
Only include questions that require looking up specific facts.
Do NOT include synthesis/reasoning questions.

Question: {question}

Output exactly 2 sub-questions, numbered:
1. [first factual sub-question]
2. [second factual sub-question]

Sub-questions:"""

        response = self.llm.generate(prompt)
        self.steps.append(Step("DECOMPOSE", question, response))
        
        # Parse sub-questions
        sub_queries = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            match = re.match(r'^[\d]+[.)]\s*(.+)', line)
            if match:
                sq_text = match.group(1).strip()
                # Skip synthesis questions
                skip_phrases = ['based on', 'can we determine', 'summarize', 'conclude', 'compare the']
                if any(phrase in sq_text.lower() for phrase in skip_phrases):
                    continue
                sub_queries.append(SubQuery(question=sq_text))
                if len(sub_queries) >= 3:  # Max 3
                    break
        
        # Fallback: if parsing failed, use the whole question
        if not sub_queries:
            sub_queries = [SubQuery(question=question)]
            
        return sub_queries
    
    def _rewrite_with_context(self, query: str, history: Dict[str, str]) -> str:
        """FIX 1: Replace pronouns and references with actual entities from history."""
        if not history:
            return query
            
        # Build history string
        history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in history.items()])
        
        prompt = f"""Previous questions and answers:
{history_str}

Current question: {query}

Rewrite the current question to be fully self-contained. Replace pronouns like 'he', 'she', 'they', 'the person', 'the identified person', 'that person' with their actual names from the answers above.

Output ONLY the rewritten question, nothing else:"""

        response = self.llm.generate(prompt)
        rewritten = response.strip().strip('"').strip("'")
        
        # If rewrite failed or is empty, return original
        if not rewritten or len(rewritten) < 5:
            return query
            
        self.steps.append(Step("REWRITE", f"{query} | History: {list(history.keys())}", rewritten))
        return rewritten
    
    def _is_retrieval_necessary(self, query: str) -> bool:
        """FIX 2: Check if this query needs retrieval or is just synthesis."""
        # Quick rule-based check first
        synthesis_phrases = [
            'based on', 'according to the above', 'from the information',
            'can we determine', 'can we conclude', 'summarize', 
            'compare the answers', 'putting it together'
        ]
        if any(phrase in query.lower() for phrase in synthesis_phrases):
            return False
        
        # For ambiguous cases, ask LLM
        prompt = f"""Does this question require searching for external facts, or is it asking to reason/summarize from already known information?

Question: {query}

Answer with ONE word only: SEARCH or REASON"""

        response = self.llm.generate(prompt)
        needs_retrieval = 'search' in response.lower()
        
        self.steps.append(Step("CHECK_RETRIEVAL_NEEDED", query, f"{'SEARCH' if needs_retrieval else 'REASON'}"))
        return needs_retrieval
    
    def _filter_relevant(self, docs: List[Dict], query: str) -> List[Dict]:
        """FIX 3: CRAG-style relevance filter. Only keep docs that actually answer the query."""
        if not docs:
            return []
        
        relevant = []
        for doc in docs:
            title = doc.get('title', 'Unknown')
            text = doc.get('text', '')[:500]
            
            prompt = f"""Does this document contain information that helps answer the question?

Question: {query}
Document [{title}]: {text}

Answer YES or NO only:"""

            response = self.llm.generate(prompt)
            is_relevant = 'yes' in response.lower()
            
            if is_relevant:
                relevant.append(doc)
        
        self.steps.append(Step("FILTER_RELEVANT", query, f"Kept {len(relevant)}/{len(docs)} docs"))
        return relevant
    
    def _refine_query_simple(self, query: str) -> str:
        """Generate a refined query when initial retrieval fails. Focus on adding synonyms/related terms."""
        # Extract the key entity from the query
        prompt = f"""The search for "{query}" didn't find relevant results.

Generate a SHORT search query (3-6 words max) that might work better.
Focus on the main entity and try synonyms for key terms:
- "government position" → try "ambassador", "diplomat", "political career"
- "nationality" → try "born in", "American", "British"

Better query (short, 3-6 words):"""

        response = self.llm.generate(prompt)
        refined = response.strip().strip('"').strip("'").split('\n')[0]  # Take first line only
        
        self.steps.append(Step("REFINE_QUERY", query, refined))
        return refined
    
    def _retrieve(self, query: str) -> List[Dict]:
        """Retrieve documents for a query."""
        docs = self.retriever.search(query, top_k=self.max_docs)
        
        # Format for logging
        titles = [d.get('title', 'Unknown')[:30] for d in docs]
        self.steps.append(Step("RETRIEVE", query, f"Retrieved: {titles}"))
        
        return docs
    
    def _extract_answer(self, question: str, context: List[Dict], history: Dict[str, str]) -> str:
        """Extract the answer to a sub-question from context and history."""
        if not context and not history:
            return "Unknown"
        
        # Build context string
        context_str = ""
        if context:
            context_str = "\n".join([f"[{d.get('title', 'Unknown')}]: {d.get('text', '')}" for d in context])
        
        # Build history string
        history_str = ""
        if history:
            history_str = "Previous findings:\n" + "\n".join([f"- {q}: {a}" for q, a in history.items()])
        
        prompt = f"""Based on the context and previous findings, answer this question thoroughly.

{history_str}

Context:
{context_str}

Question: {question}

INSTRUCTIONS:
- Extract ALL relevant information that answers the question
- Include specific titles, positions, roles, or facts mentioned
- Don't just give one example - list all that apply
- Be complete but concise

Complete answer:"""

        response = self.llm.generate(prompt)
        answer = response.strip()
        
        self.steps.append(Step("EXTRACT", question, answer))
        return answer
    
    def _generate_final(self, question: str, history: Dict[str, str], context: List[Dict]) -> str:
        """Generate the final answer from history and context."""
        # Build summary of findings
        findings = "\n".join([f"- {q}: {a}" for q, a in history.items()])
        
        prompt = f"""Based on these findings, answer the main question.

Findings:
{findings}

Main question: {question}

Final answer (be direct and concise):"""

        response = self.llm.generate(prompt)
        final = response.strip()
        
        self.steps.append(Step("GENERATE_FINAL", question, final))
        return final
