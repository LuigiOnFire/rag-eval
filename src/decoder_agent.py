"""
Decoder-Based RAG Agent

Instead of an encoder classifier, we use an LLM that generates structured actions.
The LLM sees full context and emits action tokens with parameters.

Key differences from encoder approach:
1. Actions include parameters (e.g., RETRIEVE["specific query"])
2. Larger context window (no 512 token limit)
3. Natural reasoning traces
4. Can be trained via SFT on successful trajectories

Action Format:
    Action: ACTION_TYPE["parameter"]
    
    Examples:
    - Action: RETRIEVE_KEYWORD["Shirley Temple government position"]
    - Action: RETRIEVE_DENSE["actresses who held political office"]
    - Action: DECOMPOSE  (LLM then generates sub-questions)
    - Action: REASON  (LLM synthesizes from context)
    - Action: GENERATE_ANSWER  (terminal - produce final answer)
    - Action: DELEGATE_SLM  (hand off to smaller model for efficiency)

Energy Tracking:
    - LLM actions: measured per call
    - SLM actions: cheaper, used when confident
    - Retrieval: cheap baseline cost
"""

import re
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Available actions for the decoder agent (10 actions with SLM/LLM variants)."""
    # Retrieval (parameterized)
    RETRIEVE_KEYWORD = "RETRIEVE_KEYWORD"
    RETRIEVE_DENSE = "RETRIEVE_DENSE"
    # Decomposition
    DECOMPOSE_SLM = "DECOMPOSE_SLM"
    DECOMPOSE_LLM = "DECOMPOSE_LLM"
    # Reasoning
    REASON_SLM = "REASON_SLM"
    REASON_LLM = "REASON_LLM"
    # Verification (new)
    VERIFY_COVERAGE = "VERIFY_COVERAGE"  # Check if all entities/subqs have info
    VERIFY_RELEVANCE = "VERIFY_RELEVANCE"  # Check if retrieved docs are useful
    # Generation (terminal)
    GENERATE_SLM = "GENERATE_SLM"
    GENERATE_LLM = "GENERATE_LLM"


@dataclass
class Action:
    """Parsed action from LLM output."""
    action_type: ActionType
    parameters: Optional[str] = None  # Query string for retrieval, etc.
    sub_questions: List[str] = field(default_factory=list)  # For DECOMPOSE
    reasoning: Optional[str] = None  # For REASON
    answer: Optional[str] = None  # For GENERATE_ANSWER


@dataclass  
class SubQuestion:
    """Tracks a sub-question and its resolution status."""
    question: str
    status: str = "pending"  # pending, searching, answered, failed
    answer: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    attempts: int = 0


@dataclass
class AgentState:
    """Full state of the agent, passed to LLM at each step."""
    original_query: str
    sub_questions: List[SubQuestion] = field(default_factory=list)
    retrieved_docs: List[Dict[str, str]] = field(default_factory=list)  # [{title, content, source}]
    reasoning_trace: List[str] = field(default_factory=list)
    current_answer: Optional[str] = None
    step_count: int = 0
    total_energy_wh: float = 0.0
    
    def to_prompt(self, max_doc_tokens: int = 500) -> str:
        """Format state as a prompt for the LLM."""
        sections = []
        
        # Original query
        sections.append(f"=== QUERY ===\n{self.original_query}")
        
        # Sub-questions if any
        if self.sub_questions:
            sq_lines = ["=== SUB-QUESTIONS ==="]
            for i, sq in enumerate(self.sub_questions, 1):
                sq_lines.append(f"Q{i}: {sq.question}")
                sq_lines.append(f"  Status: {sq.status.upper()}")
                if sq.answer:
                    sq_lines.append(f"  Answer: {sq.answer}")
                if sq.evidence:
                    sq_lines.append(f"  Evidence: {sq.evidence[0][:100]}...")
                if sq.attempts > 0:
                    sq_lines.append(f"  Attempts: {sq.attempts}")
            sections.append("\n".join(sq_lines))
        
        # Retrieved documents
        if self.retrieved_docs:
            doc_lines = ["=== RETRIEVED CONTEXT ==="]
            for i, doc in enumerate(self.retrieved_docs[-5:], 1):  # Last 5 docs
                title = doc.get('title', f'Doc {i}')
                content = doc.get('content', '')[:max_doc_tokens]
                doc_lines.append(f"[{title}]: {content}")
            sections.append("\n".join(doc_lines))
        
        # Reasoning trace
        if self.reasoning_trace:
            trace_lines = ["=== REASONING TRACE ==="]
            for step in self.reasoning_trace[-5:]:  # Last 5 reasoning steps
                trace_lines.append(f"- {step}")
            sections.append("\n".join(trace_lines))
        
        # Current answer attempt
        if self.current_answer:
            sections.append(f"=== CURRENT ANSWER ATTEMPT ===\n{self.current_answer}")
        
        return "\n\n".join(sections)


class ActionParser:
    """Parse LLM output into structured actions."""
    
    # Pattern: Action: ACTION_TYPE["parameter"] or Action: ACTION_TYPE
    ACTION_PATTERN = re.compile(
        r'Action:\s*(RETRIEVE_KEYWORD|RETRIEVE_DENSE|DECOMPOSE_SLM|DECOMPOSE_LLM|REASON_SLM|REASON_LLM|VERIFY_COVERAGE|VERIFY_RELEVANCE|GENERATE_SLM|GENERATE_LLM)'
        r'(?:\s*\[\s*"([^"]+)"\s*\])?',
        re.IGNORECASE
    )
    
    # Pattern for sub-questions after DECOMPOSE
    SUBQ_PATTERN = re.compile(r'(?:^|\n)\s*(?:\d+[\.\)]\s*|[-•]\s*)(.+?)(?=\n|$)')
    
    # Pattern for answer after GENERATE_ANSWER
    ANSWER_PATTERN = re.compile(r'Answer:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.DOTALL)
    
    @classmethod
    def parse(cls, llm_output: str) -> Optional[Action]:
        """Parse LLM output into an Action object."""
        # Find the action declaration
        match = cls.ACTION_PATTERN.search(llm_output)
        if not match:
            logger.warning(f"Could not parse action from: {llm_output[:200]}")
            return None
        
        action_type_str = match.group(1).upper()
        parameter = match.group(2)
        
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            logger.warning(f"Unknown action type: {action_type_str}")
            return None
        
        action = Action(action_type=action_type, parameters=parameter)
        
        # Extract additional info based on action type
        # (Sub-questions and answers are now generated by workers, not parsed from controller output)
        
        return action


class DecoderAgent:
    """
    LLM-based agent that generates actions with parameters.
    
    Unlike the encoder approach, this agent:
    1. Sees full context (not compressed to 50 tokens)
    2. Generates retrieval queries dynamically
    3. Routes to SLM or LLM based on confidence/complexity
    """
    
    SYSTEM_PROMPT = """You are a question-answering agent. Answer questions using retrieved documents.

ACTIONS (pick ONE per turn):
1. RETRIEVE_KEYWORD["query"] - Search with keywords
2. RETRIEVE_DENSE["query"] - Semantic search
3. DECOMPOSE_SLM - Break into sub-questions
4. REASON_SLM - Analyze context
5. GENERATE_SLM - Output final answer
6. GENERATE_LLM - Output final answer (complex)

RULES:
1. If you see "=== RETRIEVED CONTEXT ===" with relevant info → GENERATE_SLM or GENERATE_LLM
2. For comparison questions ("Are X and Y both Z?") → retrieve about X, retrieve about Y, then GENERATE
3. For bridge questions ("What is Z of Y that did X?") → retrieve to find Y, retrieve about Y's Z, then GENERATE
4. ALWAYS include query in quotes: RETRIEVE_KEYWORD["your query here"]
5. After 2-3 retrievals, you MUST generate an answer
6. Do NOT repeat the same retrieval query

OUTPUT FORMAT:
Thought: [brief reasoning]
Action: ACTION_NAME["query"] or Action: ACTION_NAME"""

    def __init__(
        self,
        llm,  # Main LLM for decisions
        slm,  # Smaller model for delegation
        retriever,  # BM25 or dense retriever
        max_steps: int = 10,
        cost_table: Optional[Dict[str, float]] = None,
    ):
        self.llm = llm
        self.slm = slm
        self.retriever = retriever
        self.max_steps = max_steps
        self.cost_table = cost_table or {
            # Retrieval (cheap)
            'RETRIEVE_KEYWORD': 0.009,
            'RETRIEVE_DENSE': 0.009,
            # Decomposition
            'DECOMPOSE_SLM': 0.024,
            'DECOMPOSE_LLM': 0.028,
            # Reasoning
            'REASON_SLM': 0.065,
            'REASON_LLM': 0.027,
            # Generation
            'GENERATE_SLM': 0.020,
            'GENERATE_LLM': 0.015,
            # Controller overhead
            'LLM_CALL': 0.015,
        }
    
    def run(self, query: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the agent on a query.
        
        Returns trajectory with all steps, actions, and final answer.
        """
        state = AgentState(original_query=query)
        trajectory = {
            'query': query,
            'ground_truth': ground_truth,
            'steps': [],
            'answer': None,
            'is_correct': False,
            'total_energy_wh': 0.0,
        }
        
        while state.step_count < self.max_steps:
            state.step_count += 1
            
            # Force GENERATE if we have context and enough steps
            force_generate = (
                state.step_count >= 4 and 
                len(state.retrieved_docs) >= 2
            )
            
            # Get LLM decision
            prompt = self._build_prompt(state)
            if force_generate:
                prompt += "\n\nIMPORTANT: You have retrieved enough context. You MUST now use GENERATE_SLM to answer the question based on the retrieved documents."
            
            llm_output = self.llm.generate(prompt)
            state.total_energy_wh += self.cost_table['LLM_CALL']
            
            # Parse action
            action = ActionParser.parse(llm_output)
            if action is None:
                logger.warning(f"Failed to parse action at step {state.step_count}")
                state.reasoning_trace.append("Failed to parse action, retrying...")
                continue
            
            # Force GENERATE if LLM still won't generate
            if force_generate and action.action_type not in (ActionType.GENERATE_SLM, ActionType.GENERATE_LLM):
                logger.info(f"Forcing GENERATE at step {state.step_count}")
                action = Action(action_type=ActionType.GENERATE_SLM, parameters=None)
            
            # Record step
            step = {
                'step': state.step_count,
                'state_summary': state.to_prompt()[:500],
                'llm_output': llm_output,
                'action': action.action_type.value,
                'parameters': action.parameters,
            }
            
            # Execute action
            result = self._execute_action(action, state)
            step['result'] = result
            trajectory['steps'].append(step)
            
            # Update energy
            state.total_energy_wh += self.cost_table.get(action.action_type.value, 0.01)
            
            # Check for terminal action (GENERATE_SLM or GENERATE_LLM)
            if action.action_type in (ActionType.GENERATE_SLM, ActionType.GENERATE_LLM):
                trajectory['answer'] = state.current_answer or result
                trajectory['total_energy_wh'] = state.total_energy_wh
                
                # Check correctness if ground truth provided
                if ground_truth:
                    trajectory['is_correct'] = self._check_answer(
                        trajectory['answer'], ground_truth
                    )
                break
        
        return trajectory
    
    def _build_prompt(self, state: AgentState) -> str:
        """Build the full prompt for the LLM."""
        return f"{self.SYSTEM_PROMPT}\n\n{state.to_prompt()}\n\n=== NEXT ACTION ==="
    
    def _execute_action(self, action: Action, state: AgentState) -> str:
        """Execute an action and update state. Dispatches to appropriate worker."""
        
        # === RETRIEVAL ===
        if action.action_type == ActionType.RETRIEVE_KEYWORD:
            query = action.parameters or state.original_query
            docs = self.retriever.search(query, top_k=5)
            for doc in docs:
                state.retrieved_docs.append({
                    'title': doc.get('title', 'Unknown'),
                    'content': doc.get('text', doc.get('content', '')),
                    'source': 'keyword_search',
                    'query': query,
                })
            return f"Retrieved {len(docs)} documents for '{query}'"
        
        elif action.action_type == ActionType.RETRIEVE_DENSE:
            query = action.parameters or state.original_query
            docs = self.retriever.search(query, top_k=5)
            for doc in docs:
                state.retrieved_docs.append({
                    'title': doc.get('title', 'Unknown'),
                    'content': doc.get('text', doc.get('content', '')),
                    'source': 'dense_search',
                    'query': query,
                })
            return f"Retrieved {len(docs)} documents for '{query}'"
        
        # === DECOMPOSITION ===
        elif action.action_type in (ActionType.DECOMPOSE_SLM, ActionType.DECOMPOSE_LLM):
            worker = self.slm if action.action_type == ActionType.DECOMPOSE_SLM else self.llm
            
            # Build decomposition prompt
            decompose_prompt = f"""Break this question into simpler sub-questions that can be answered independently.

Question: {state.original_query}

Output each sub-question on a new line, numbered:
1. [first sub-question]
2. [second sub-question]
..."""
            
            result = worker.generate(decompose_prompt)
            
            # Parse sub-questions from result
            import re
            sub_qs = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?=\n|$)', result)
            for sq in sub_qs:
                state.sub_questions.append(SubQuestion(question=sq.strip()))
            
            state.reasoning_trace.append(f"Decomposed into {len(sub_qs)} sub-questions")
            return f"Sub-questions: {sub_qs}"
        
        # === REASONING ===
        elif action.action_type in (ActionType.REASON_SLM, ActionType.REASON_LLM):
            worker = self.slm if action.action_type == ActionType.REASON_SLM else self.llm
            
            # Build reasoning prompt with context
            context = "\n\n".join(
                f"[{doc['title']}]: {doc['content'][:400]}" 
                for doc in state.retrieved_docs[-5:]
            )
            
            reason_prompt = f"""Based on the following context, reason about the question.

Context:
{context}

Question: {state.original_query}

What can we conclude? What information is still missing?"""
            
            result = worker.generate(reason_prompt)
            state.reasoning_trace.append(result[:500])
            return f"Reasoning: {result[:200]}..."
        
        # === GENERATION (terminal) ===
        elif action.action_type in (ActionType.GENERATE_SLM, ActionType.GENERATE_LLM):
            worker = self.slm if action.action_type == ActionType.GENERATE_SLM else self.llm
            
            # Build generation prompt with all context
            context = "\n\n".join(
                f"[{doc['title']}]: {doc['content'][:400]}" 
                for doc in state.retrieved_docs[-5:]
            )
            
            reasoning = "\n".join(state.reasoning_trace[-3:]) if state.reasoning_trace else ""
            
            gen_prompt = f"""Answer the question based on the context provided. Be concise and factual.

Context:
{context}

{f"Reasoning so far: {reasoning}" if reasoning else ""}

Question: {state.original_query}

Answer:"""
            
            answer = worker.generate(gen_prompt)
            # Clean up answer - take first sentence/line
            answer = answer.strip().split('\n')[0].strip()
            state.current_answer = answer
            return answer
        
        return "Unknown action"
    
    def _check_answer(self, answer: str, ground_truth: str) -> bool:
        """Check if answer matches ground truth."""
        if not answer:
            return False
        answer_lower = answer.lower().strip()
        truth_lower = ground_truth.lower().strip()
        # Lenient matching - ground truth appears in answer
        return truth_lower in answer_lower or answer_lower in truth_lower


def create_training_data_from_trajectory(trajectory: Dict) -> List[Dict]:
    """
    Convert a successful trajectory into training examples for SFT.
    
    Each example is (state, expected_output) where expected_output
    is the Thought + Action from that step.
    """
    if not trajectory.get('is_correct', False):
        return []
    
    examples = []
    for step in trajectory['steps']:
        examples.append({
            'input': step['state_summary'],
            'output': step['llm_output'],
            'action': step['action'],
        })
    
    return examples
