"""
Structured RAG Agent with Decompose-Retrieve-Reflect-Generate architecture.

Key insight: The agent must CHECK if retrieved context actually answers
the sub-question before moving on. One refinement retry per sub-query.
Now with Tiered Retrieval: FAST → SMART → DEEP escalation."""
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
    
    def __init__(self, llm, retriever, max_docs_per_query: int = 5, use_tiered_retrieval: bool = True):
        """
        Args:
            llm: Language model with .generate(prompt) -> str
            retriever: Retriever with .search(query, top_k) -> List[Dict] or tiered search
            max_docs_per_query: Max documents to retrieve per query
            use_tiered_retrieval: Whether to use FAST → SMART → DEEP escalation
        """
        self.llm = llm
        self.retriever = retriever
        self.max_docs = max_docs_per_query
        self.use_tiered_retrieval = use_tiered_retrieval
        self.steps: List[Step] = []
        self.total_energy_cost = 0.0
        
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
                # Tiered retrieval: FAST → SMART → DEEP escalation
                valid_docs = self._retrieve_with_escalation(executable_query)
                
                if not valid_docs:
                    # Last resort: refine query and try DEEP tier
                    refined_query = self._refine_query_simple(executable_query)
                    valid_docs = self._retrieve_tier(refined_query, "DEEP")
                
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
        """FIX 3: CRAG-style relevance filter. Score docs by relevance and keep the best ones."""
        if not docs:
            return []
        
        # For small doc sets (<=3), keep all - filtering too aggressive
        if len(docs) <= 3:
            self.steps.append(Step("FILTER_RELEVANT", query, f"Kept all {len(docs)} docs (small set)"))
            return docs
        
        # Score each document for relevance
        scored_docs = []
        for doc in docs:
            title = doc.get('title', 'Unknown')
            text = doc.get('text', '')[:500]
            
            # Use confidence scoring instead of YES/NO
            prompt = f"""Rate how relevant this document is to answering the question.

Question: {query}
Document [{title}]: {text}

Answer with ONE word only: HIGH, MEDIUM, or LOW"""

            response = self.llm.generate(prompt).lower()
            
            # Score: HIGH=3, MEDIUM=2, LOW=1
            if 'high' in response:
                score = 3
            elif 'medium' in response:
                score = 2
            else:
                score = 1
            
            scored_docs.append((score, doc))
        
        # Sort by score and keep top documents (at least 1, up to 3)
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Keep HIGH-scored docs, or top 2 if none are HIGH
        relevant = [doc for score, doc in scored_docs if score >= 3]  # HIGH
        if not relevant:  # No HIGH scores, keep top 2
            relevant = [doc for _, doc in scored_docs[:2]]
        
        self.steps.append(Step("FILTER_RELEVANT", query, f"Kept {len(relevant)}/{len(docs)} docs (scores: {[s for s,_ in scored_docs[:3]]})"))
        return relevant
    
    def _refine_query_simple(self, query: str) -> str:
        """Generate a refined query when initial retrieval fails."""
        prompt = f"""The search query didn't find good results. Reformulate it to find the right Wikipedia article.

Original: {query}

Strategy:
1. Extract the main entity name (person, place, thing)
2. Add identifying context: occupation, type, category
3. Remove question words (what, who, when, where)

Examples:
- "What is the nationality of Scott Derrickson?" → "Scott Derrickson director"
- "When was Ed Wood born?" → "Ed Wood filmmaker"
- "What science fantasy series has companion books?" → "science fantasy young adult series companion books"
- "What government position did Shirley Temple hold?" → "Shirley Temple ambassador diplomat"

Better query (2-5 keywords, no questions):"""

        response = self.llm.generate(prompt)
        refined = response.strip().strip('"').strip("'").split('\n')[0]  # Take first line only
        
        self.steps.append(Step("REFINE_QUERY", query, refined))
        return refined
    
    def _retrieve_with_escalation(self, query: str) -> List[Dict]:
        """Tiered retrieval: FAST → SMART → DEEP escalation."""
        if not self.use_tiered_retrieval:
            return self._retrieve_tier(query, "FAST")
        
        # TIER 1: Try FAST (BM25 only)
        docs = self._retrieve_tier(query, "FAST")
        valid_docs = self._filter_relevant(docs, query)
        
        if valid_docs:
            return valid_docs
        
        # TIER 2: Escalate to SMART (Dense)
        self.steps.append(Step("ESCALATE_TO_SMART", query, "No relevant docs with BM25, trying dense retrieval"))
        docs = self._retrieve_tier(query, "SMART")
        valid_docs = self._filter_relevant(docs, query)
        
        if valid_docs:
            return valid_docs
        
        # TIER 3: Nuclear option - DEEP (Hybrid)
        self.steps.append(Step("ESCALATE_TO_DEEP", query, "No relevant docs with dense, trying hybrid retrieval"))
        docs = self._retrieve_tier(query, "DEEP")
        return self._filter_relevant(docs, query)
    
    def _retrieve_tier(self, query: str, tier: str) -> List[Dict]:
        """Retrieve documents using specific tier."""
        # Check if retriever supports tiers
        if hasattr(self.retriever, 'search') and 'tier' in self.retriever.search.__code__.co_varnames:
            docs = self.retriever.search(query, top_k=self.max_docs, tier=tier)
            
            # Track energy cost if available
            if hasattr(self.retriever, 'get_last_cost'):
                cost = self.retriever.get_last_cost()
                self.total_energy_cost += cost
        else:
            # Fallback for non-tiered retrievers
            docs = self.retriever.search(query, top_k=self.max_docs)
        
        # Format for logging
        titles = [d.get('title', 'Unknown')[:30] for d in docs]
        self.steps.append(Step(f"RETRIEVE_{tier}", query, f"Retrieved: {titles}"))
        
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
