"""
GreenTreeSearch: Cost-ordered trajectory generation for Green-DeepRAG.

This module implements Phase 1 of the training pipeline:
- Simulates the iterative Manager-Worker agent
- Tries cheaper actions first (cost-ordered search)
- Records the cheapest trajectory that yields a correct answer
- Workers compress observations to <50 tokens for the controller

Output: Training data of (state, action, observation) trajectories for behavior cloning.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import IntEnum

logger = logging.getLogger(__name__)


# =============================================================================
# Action Types and Parameterized Actions
# =============================================================================

class ActionType(IntEnum):
    """Base action types - these map to neural network output classes."""
    GENERATE = 0
    DECOMPOSE = 1
    RETRIEVE = 2
    REASON = 3


class Model(IntEnum):
    """Model choices for LLM-based actions."""
    SLM = 0
    LLM = 1


class RetrievalMethod(IntEnum):
    """Retrieval method choices."""
    KEYWORD = 0  # BM25
    DENSE = 1    # Vector/embedding-based


@dataclass(frozen=True)
class ActionCall:
    """
    A parameterized action call.
    
    Examples:
        ActionCall(ActionType.GENERATE, model=Model.SLM)
        ActionCall(ActionType.RETRIEVE, method=RetrievalMethod.DENSE, query="sub_question_1", top_k=5)
        ActionCall(ActionType.DECOMPOSE, model=Model.LLM)
    """
    action_type: ActionType
    model: Optional[Model] = None
    method: Optional[RetrievalMethod] = None
    query: Optional[str] = None  # For RETRIEVE - can be "original" or a sub-question
    top_k: int = 3  # For RETRIEVE - how many passages to retrieve
    
    @property
    def is_terminal(self) -> bool:
        """Terminal actions produce a final answer."""
        return self.action_type == ActionType.GENERATE
    
    @property
    def legacy_action_id(self) -> int:
        """
        Map to legacy 8-class action space for backward compatibility.
        
        Legacy mapping:
            0: GENERATE_END_SLM, 1: GENERATE_END_LLM
            2: DECOMPOSE_SLM, 3: DECOMPOSE_LLM
            4: RETRIEVE_KEYWORD, 5: RETRIEVE_DENSE
            6: REASON_SLM, 7: REASON_LLM
        """
        if self.action_type == ActionType.GENERATE:
            return 0 if self.model == Model.SLM else 1
        elif self.action_type == ActionType.DECOMPOSE:
            return 2 if self.model == Model.SLM else 3
        elif self.action_type == ActionType.RETRIEVE:
            return 4 if self.method == RetrievalMethod.KEYWORD else 5
        elif self.action_type == ActionType.REASON:
            return 6 if self.model == Model.SLM else 7
        return -1
    
    @classmethod
    def from_legacy_id(cls, action_id: int, query: Optional[str] = None) -> 'ActionCall':
        """Create ActionCall from legacy 8-class action ID."""
        mapping = {
            0: cls(ActionType.GENERATE, model=Model.SLM),
            1: cls(ActionType.GENERATE, model=Model.LLM),
            2: cls(ActionType.DECOMPOSE, model=Model.SLM),
            3: cls(ActionType.DECOMPOSE, model=Model.LLM),
            4: cls(ActionType.RETRIEVE, method=RetrievalMethod.KEYWORD, query=query),
            5: cls(ActionType.RETRIEVE, method=RetrievalMethod.DENSE, query=query),
            6: cls(ActionType.REASON, model=Model.SLM),
            7: cls(ActionType.REASON, model=Model.LLM),
        }
        return mapping.get(action_id, cls(ActionType.GENERATE, model=Model.SLM))
    
    def __str__(self) -> str:
        """Human-readable representation."""
        if self.action_type == ActionType.GENERATE:
            model_name = self.model.name if self.model is not None else "UNKNOWN"
            return f"GENERATE(model={model_name})"
        elif self.action_type == ActionType.DECOMPOSE:
            model_name = self.model.name if self.model is not None else "UNKNOWN"
            return f"DECOMPOSE(model={model_name})"
        elif self.action_type == ActionType.RETRIEVE:
            q_str = f", query='{self.query[:30]}...'" if self.query and len(self.query) > 30 else (f", query='{self.query}'" if self.query else "")
            method_name = self.method.name if self.method is not None else "UNKNOWN"
            k_str = f", k={self.top_k}" if self.top_k != 3 else ""  # Only show if not default
            return f"RETRIEVE(method={method_name}{q_str}{k_str})"
        elif self.action_type == ActionType.REASON:
            model_name = self.model.name if self.model is not None else "UNKNOWN"
            return f"REASON(model={model_name})"
        return f"ActionCall({self.action_type})"
    
    @property
    def name(self) -> str:
        """Legacy-compatible name property."""
        return str(self)


# Convenience constructors
def GENERATE(model: Union[Model, str] = Model.SLM) -> ActionCall:
    """Create a GENERATE action."""
    if isinstance(model, str):
        model = Model.SLM if model.lower() == 'slm' else Model.LLM
    return ActionCall(ActionType.GENERATE, model=model)

def DECOMPOSE(model: Union[Model, str] = Model.SLM) -> ActionCall:
    """Create a DECOMPOSE action."""
    if isinstance(model, str):
        model = Model.SLM if model.lower() == 'slm' else Model.LLM
    return ActionCall(ActionType.DECOMPOSE, model=model)

def RETRIEVE(method: Union[RetrievalMethod, str] = RetrievalMethod.KEYWORD, query: Optional[str] = None, top_k: int = 3) -> ActionCall:
    """Create a RETRIEVE action."""
    if isinstance(method, str):
        method = RetrievalMethod.KEYWORD if method.lower() in ('keyword', 'bm25') else RetrievalMethod.DENSE
    return ActionCall(ActionType.RETRIEVE, method=method, query=query, top_k=top_k)

def REASON(model: Union[Model, str] = Model.SLM) -> ActionCall:
    """Create a REASON action."""
    if isinstance(model, str):
        model = Model.SLM if model.lower() == 'slm' else Model.LLM
    return ActionCall(ActionType.REASON, model=model)


# =============================================================================
# Legacy Action Enum (for backward compatibility with behavior cloning)
# =============================================================================

class Action(IntEnum):
    """Legacy 8-class action space - kept for neural network compatibility."""
    GENERATE_END_SLM = 0
    GENERATE_END_LLM = 1
    DECOMPOSE_SLM = 2
    DECOMPOSE_LLM = 3
    RETRIEVE_KEYWORD = 4
    RETRIEVE_DENSE = 5
    REASON_SLM = 6
    REASON_LLM = 7
    
    def to_action_call(self, query: Optional[str] = None) -> ActionCall:
        """Convert legacy Action to ActionCall."""
        return ActionCall.from_legacy_id(int(self), query)


# Terminal actions that produce a final answer
TERMINAL_ACTIONS = {Action.GENERATE_END_SLM, Action.GENERATE_END_LLM}
TERMINAL_ACTION_TYPES = {ActionType.GENERATE}

# Default prompts for worker compression
COMPRESSION_PROMPT = """Summarize the following in under 50 tokens, focusing on what was found and what's missing:

{content}

Summary:"""

DECOMPOSE_PROMPT = """Break down this question into 2-3 simpler sub-questions that would help answer it:

Question: {question}

Sub-questions:"""

REASON_PROMPT = """Analyze the context below to extract facts relevant to the question.
Only cite facts that are EXPLICITLY stated in the context. Do NOT add information from your own knowledge.
Identify what key facts have been found and what (if anything) is still missing.

Context:
{context}

Question: {question}

Analysis (citing only facts from context above):"""

GENERATE_PROMPT = """Answer the question using ONLY the information provided in the context below. 
Be concise and factual. If the context provides the answer, use it directly.
Do NOT use your own knowledge - only use what is explicitly stated in the context.
If the context does not contain enough information to answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer (based only on the context above):"""


@dataclass
class Step:
    """A single step in a trajectory."""
    state: str  # The compressed state string fed to controller
    action: ActionCall  # The parameterized action taken
    observation: str  # The compressed observation from worker (<50 tokens)
    result: Optional[str] = None  # Raw result (e.g., full answer for terminal actions)
    energy_wh: float = 0.0  # Energy consumed by this action
    
    @property
    def action_id(self) -> int:
        """Get legacy action ID for neural network."""
        return self.action.legacy_action_id


@dataclass
class Trajectory:
    """A complete trajectory from query to answer."""
    query: str
    ground_truth: str
    steps: List[Step] = field(default_factory=list)
    correct: bool = False
    total_energy_wh: float = 0.0
    
    def to_training_pairs(self) -> List[Dict[str, Any]]:
        """
        Convert trajectory to (state, action) training pairs for behavior cloning.
        
        Returns:
            List of dicts with 'text' (state string) and 'label' (action ID)
        """
        pairs = []
        for step in self.steps:
            pairs.append({
                "text": step.state,
                "label": step.action_id,  # Use legacy ID for neural network
                "action_str": str(step.action),  # Full parameterized action for debugging
                "observation": step.observation,
                "energy_wh": step.energy_wh
            })
        return pairs
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize trajectory to dictionary."""
        return {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "steps": [
                {
                    "state": s.state,
                    "action_id": s.action_id,
                    "action": str(s.action),  # Full parameterized action
                    "action_type": s.action.action_type.name,
                    "action_params": {
                        "model": s.action.model.name if s.action.model else None,
                        "method": s.action.method.name if s.action.method else None,
                        "query": s.action.query,
                    },
                    "observation": s.observation,
                    "result": s.result,
                    "energy_wh": s.energy_wh
                }
                for s in self.steps
            ],
            "correct": self.correct,
            "total_energy_wh": self.total_energy_wh
        }


class GreenTreeSearch:
    """
    Cost-ordered search to generate training trajectories.
    
    Simulates the Manager-Worker iterative agent, trying cheaper actions first
    and recording the cheapest trajectory that yields a correct answer.
    
    Key design principles:
    1. Workers execute actions and compress results to <50 token observations
    2. Controller only sees compressed state: [CLS] Query [SEP] Obs1 [SEP] Obs2 [SEP] ...
    3. Actions are tried in cost order (from measured cost_table.json)
    4. Search terminates when a terminal action produces a correct answer
    """
    
    def __init__(
        self,
        slm,  # OllamaGenerator for SLM
        llm,  # OllamaGenerator for LLM
        retriever,  # BM25Retriever or DenseRetriever
        judge,  # Callable[[str, str], bool] - checks if answer is correct
        cost_table: Dict[int, float],  # Action ID -> energy cost in Wh
        max_steps: int = 5,
        max_obs_tokens: int = 50,
        dense_retriever = None  # Optional separate dense retriever
    ):
        """
        Initialize GreenTreeSearch.
        
        Args:
            slm: Small language model (OllamaGenerator)
            llm: Large language model (OllamaGenerator)
            retriever: Primary retriever (typically BM25)
            judge: Function(answer, ground_truth) -> bool
            cost_table: Dict mapping action ID to energy cost in Wh
            max_steps: Maximum steps per trajectory
            max_obs_tokens: Maximum tokens in compressed observations
            dense_retriever: Optional dense retriever for RETRIEVE_DENSE action
        """
        self.slm = slm
        self.llm = llm
        self.retriever = retriever
        self.dense_retriever = dense_retriever
        self.judge = judge
        self.cost_table = cost_table
        self.max_steps = max_steps
        self.max_obs_tokens = max_obs_tokens
        
        # Sort actions by cost (cheapest first)
        self._cost_ordered_actions = sorted(
            Action,
            key=lambda a: cost_table.get(int(a), float('inf'))
        )
        
        # Context accumulated during search
        self._current_context: List[Dict] = []
        self._sub_questions: List[str] = []
        self._reasoning_notes: str = ""
        
        logger.info(f"GreenTreeSearch initialized with cost order: "
                   f"{[a.name for a in self._cost_ordered_actions]}")
    
    def get_cost_ordered_actions(self) -> List[Action]:
        """Return actions sorted by cost (cheapest first)."""
        return self._cost_ordered_actions.copy()
    
    def search(self, query: str, ground_truth: str) -> Trajectory:
        """
        Search for the cheapest correct trajectory by trying strategies in order.
        
        Strategies (simplest to most complex):
        1. GENERATE immediately (no retrieval, no decomposition)
        2. RETRIEVE → GENERATE
        3. DECOMPOSE → RETRIEVE(sub-Qs) → GENERATE  
        4. RETRIEVE → DECOMPOSE → RETRIEVE(sub-Qs) → GENERATE
        5. DECOMPOSE → RETRIEVE(sub-Qs) → REASON → GENERATE
        6. Full pipeline with multiple retrieval/reasoning rounds
        
        Returns the trajectory from the FIRST strategy that produces a correct answer.
        This teaches the model when simpler approaches suffice vs when complexity is needed.
        
        Args:
            query: The question to answer
            ground_truth: The expected answer
            
        Returns:
            Trajectory with steps, correctness flag, and total energy
        """
        # Define strategies as sequences of actions
        # Each strategy is a list of ActionCalls to try in order
        strategies = self._get_strategies(query)
        
        for strategy_idx, strategy in enumerate(strategies):
            trajectory = self._try_strategy(query, ground_truth, strategy, strategy_idx)
            
            if trajectory.correct:
                logger.info(f"Strategy {strategy_idx + 1} succeeded for: {query[:50]}...")
                return trajectory
            else:
                logger.debug(f"Strategy {strategy_idx + 1} failed, trying next...")
        
        # All strategies failed - return the last trajectory (most complete attempt)
        logger.warning(f"All strategies failed for: {query[:50]}...")
        return trajectory
    
    def _get_strategies(self, query: str) -> List[List[ActionCall]]:
        """
        Define strategies from simplest to most complex.
        
        Each strategy is a sequence of actions to execute.
        We include many levels of granularity to learn when each is needed.
        """
        strategies = []
        
        # =================================================================
        # TIER 1: No retrieval (test if model already knows)
        # =================================================================
        
        # Strategy 1: Just generate with SLM (cheapest)
        strategies.append([
            GENERATE(Model.SLM),
        ])
        
        # Strategy 2: Generate with LLM (slightly more capable)
        strategies.append([
            GENERATE(Model.LLM),
        ])
        
        # =================================================================
        # TIER 2: Single retrieval strategies
        # =================================================================
        
        # Strategy 3: BM25 retrieval (k=3) → generate with SLM
        strategies.append([
            RETRIEVE(RetrievalMethod.KEYWORD, query=query, top_k=3),
            GENERATE(Model.SLM),
        ])
        
        # Strategy 4: BM25 retrieval (k=3) → generate with LLM
        strategies.append([
            RETRIEVE(RetrievalMethod.KEYWORD, query=query, top_k=3),
            GENERATE(Model.LLM),
        ])
        
        # Strategy 5: BM25 retrieval with MORE docs (k=5) → generate with LLM
        # Sometimes the right doc is just outside top-3
        strategies.append([
            RETRIEVE(RetrievalMethod.KEYWORD, query=query, top_k=5),
            GENERATE(Model.LLM),
        ])
        
        # Strategy 6: BM25 retrieval with even MORE docs (k=10) → generate with LLM
        strategies.append([
            RETRIEVE(RetrievalMethod.KEYWORD, query=query, top_k=10),
            GENERATE(Model.LLM),
        ])
        
        # Strategy 7: Dense retrieval → generate with LLM
        # (semantic search may find what BM25 misses)
        if self.dense_retriever:
            strategies.append([
                RETRIEVE(RetrievalMethod.DENSE, query=query, top_k=5),
                GENERATE(Model.LLM),
            ])
        
        # Strategy 8: Both retrieval methods → generate
        if self.dense_retriever:
            strategies.append([
                RETRIEVE(RetrievalMethod.KEYWORD, query=query, top_k=5),
                RETRIEVE(RetrievalMethod.DENSE, query=query, top_k=5),
                GENERATE(Model.LLM),
            ])
        
        # =================================================================
        # TIER 3: Decomposition strategies (multi-hop questions)
        # =================================================================
        
        # Strategy 9: Decompose → retrieve sub-questions (k=3) → generate
        strategies.append([
            DECOMPOSE(Model.SLM),
            RETRIEVE(RetrievalMethod.KEYWORD, query="__SUB_Q__", top_k=3),
            GENERATE(Model.LLM),
        ])
        
        # Strategy 10: LLM decompose → retrieve sub-questions (k=5) → generate
        strategies.append([
            DECOMPOSE(Model.LLM),
            RETRIEVE(RetrievalMethod.KEYWORD, query="__SUB_Q__", top_k=5),
            GENERATE(Model.LLM),
        ])
        
        # Strategy 11: LLM decompose → retrieve MORE (k=10) → generate
        strategies.append([
            DECOMPOSE(Model.LLM),
            RETRIEVE(RetrievalMethod.KEYWORD, query="__SUB_Q__", top_k=10),
            GENERATE(Model.LLM),
        ])
        
        # =================================================================
        # TIER 4: Retrieval + Decomposition combinations
        # =================================================================
        
        # Strategy 12: Retrieve first (for context) → decompose → retrieve subs → generate
        strategies.append([
            RETRIEVE(RetrievalMethod.KEYWORD, query=query, top_k=5),
            DECOMPOSE(Model.SLM),
            RETRIEVE(RetrievalMethod.KEYWORD, query="__SUB_Q__", top_k=5),
            GENERATE(Model.LLM),
        ])
        
        # =================================================================
        # TIER 5: With reasoning step
        # =================================================================
        
        # Strategy 13: Decompose → retrieve → reason → generate
        strategies.append([
            DECOMPOSE(Model.SLM),
            RETRIEVE(RetrievalMethod.KEYWORD, query="__SUB_Q__", top_k=5),
            REASON(Model.SLM),
            GENERATE(Model.LLM),
        ])
        
        # Strategy 14: Decompose → retrieve MORE (k=10) → reason with LLM → generate
        strategies.append([
            DECOMPOSE(Model.LLM),
            RETRIEVE(RetrievalMethod.KEYWORD, query="__SUB_Q__", top_k=10),
            REASON(Model.LLM),
            GENERATE(Model.LLM),
        ])
        
        # =================================================================
        # TIER 6: Follow-up retrieval (key new strategies!)
        # When first retrieval fails, try entity-specific retrieval
        # =================================================================
        
        # Strategy 15: Decompose → retrieve → reason → FOLLOW-UP retrieve → generate
        # The __FOLLOW_UP__ placeholder triggers retrieval based on what REASON found missing
        strategies.append([
            DECOMPOSE(Model.LLM),
            RETRIEVE(RetrievalMethod.KEYWORD, query="__SUB_Q__", top_k=5),
            REASON(Model.LLM),
            RETRIEVE(RetrievalMethod.KEYWORD, query="__FOLLOW_UP__", top_k=5),
            GENERATE(Model.LLM),
        ])
        
        # Strategy 16: Same but with MORE docs on follow-up (k=10)
        strategies.append([
            DECOMPOSE(Model.LLM),
            RETRIEVE(RetrievalMethod.KEYWORD, query="__SUB_Q__", top_k=5),
            REASON(Model.LLM),
            RETRIEVE(RetrievalMethod.KEYWORD, query="__FOLLOW_UP__", top_k=10),
            GENERATE(Model.LLM),
        ])
        
        # Strategy 17: Same but with dense retrieval for follow-up
        if self.dense_retriever:
            strategies.append([
                DECOMPOSE(Model.LLM),
                RETRIEVE(RetrievalMethod.KEYWORD, query="__SUB_Q__", top_k=5),
                REASON(Model.LLM),
                RETRIEVE(RetrievalMethod.DENSE, query="__FOLLOW_UP__", top_k=10),
                GENERATE(Model.LLM),
            ])
        
        # Strategy 18: Retrieve → reason → follow-up → generate (no decompose)
        strategies.append([
            RETRIEVE(RetrievalMethod.KEYWORD, query=query, top_k=10),
            REASON(Model.LLM),
            RETRIEVE(RetrievalMethod.KEYWORD, query="__FOLLOW_UP__", top_k=10),
            GENERATE(Model.LLM),
        ])
        
        # =================================================================
        # TIER 7: Full pipeline with multiple follow-ups
        # =================================================================
        
        # Strategy 19: Full pipeline with both retrieval methods and follow-up
        if self.dense_retriever:
            strategies.append([
                DECOMPOSE(Model.LLM),
                RETRIEVE(RetrievalMethod.KEYWORD, query="__SUB_Q__", top_k=10),
                RETRIEVE(RetrievalMethod.DENSE, query="__SUB_Q__", top_k=10),
                REASON(Model.LLM),
                RETRIEVE(RetrievalMethod.KEYWORD, query="__FOLLOW_UP__", top_k=10),
                GENERATE(Model.LLM),
            ])
        
        # Strategy 20: Maximum effort - decompose, multi-retrieve, reason, follow-up dense, generate
        if self.dense_retriever:
            strategies.append([
                DECOMPOSE(Model.LLM),
                RETRIEVE(RetrievalMethod.KEYWORD, query="__SUB_Q__", top_k=10),
                RETRIEVE(RetrievalMethod.DENSE, query="__SUB_Q__", top_k=10),
                REASON(Model.LLM),
                RETRIEVE(RetrievalMethod.DENSE, query="__FOLLOW_UP__", top_k=10),
                RETRIEVE(RetrievalMethod.KEYWORD, query="__FOLLOW_UP__", top_k=10),
                GENERATE(Model.LLM),
            ])
        
        # Filter out None entries
        strategies = [[a for a in s if a is not None] for s in strategies]
        
        return strategies
    
    def _try_strategy(
        self, 
        query: str, 
        ground_truth: str, 
        strategy: List[ActionCall],
        strategy_idx: int
    ) -> Trajectory:
        """
        Execute a single strategy and return the trajectory.
        
        Args:
            query: Original question
            ground_truth: Expected answer
            strategy: List of ActionCalls to execute
            strategy_idx: Index for logging
            
        Returns:
            Trajectory (may be correct or incorrect)
        """
        # Reset state for this strategy
        self._current_context = []
        self._sub_questions = []
        self._reasoning_notes = ""
        
        trajectory = Trajectory(query=query, ground_truth=ground_truth)
        state = f"[CLS] {query} [SEP]"
        
        for action in strategy:
            # Handle special "__SUB_Q__" placeholder - retrieve for each sub-question
            if action.action_type == ActionType.RETRIEVE and action.query == "__SUB_Q__":
                # Use the method from the action, defaulting to KEYWORD if None
                method = action.method if action.method is not None else RetrievalMethod.KEYWORD
                if self._sub_questions:
                    # Execute retrieval for each sub-question
                    for sub_q in self._sub_questions:
                        actual_action = RETRIEVE(method, query=sub_q)
                        result, observation, state, trajectory = self._execute_and_record(
                            actual_action, query, state, trajectory
                        )
                else:
                    # No sub-questions yet, retrieve with original query
                    actual_action = RETRIEVE(method, query=query)
                    result, observation, state, trajectory = self._execute_and_record(
                        actual_action, query, state, trajectory
                    )
            
            # Handle "__FOLLOW_UP__" placeholder - retrieve based on what REASON found missing
            elif action.action_type == ActionType.RETRIEVE and action.query == "__FOLLOW_UP__":
                method = action.method if action.method is not None else RetrievalMethod.KEYWORD
                follow_up_queries = self._extract_follow_up_queries(query)
                
                if follow_up_queries:
                    for fq in follow_up_queries:
                        actual_action = RETRIEVE(method, query=fq)
                        result, observation, state, trajectory = self._execute_and_record(
                            actual_action, query, state, trajectory
                        )
                else:
                    # Fall back to original query if no follow-ups extracted
                    actual_action = RETRIEVE(method, query=query)
                    result, observation, state, trajectory = self._execute_and_record(
                        actual_action, query, state, trajectory
                    )
            else:
                result, observation, state, trajectory = self._execute_and_record(
                    action, query, state, trajectory
                )
            
            # Check if terminal action succeeded
            if action.is_terminal:
                if self.judge(result, ground_truth):
                    trajectory.correct = True
                    return trajectory
        
        return trajectory
    
    def _execute_and_record(
        self,
        action: ActionCall,
        query: str,
        state: str,
        trajectory: Trajectory
    ) -> Tuple[str, str, str, Trajectory]:
        """Execute an action and record it in the trajectory."""
        result, observation = self._execute_action(action, query, state)
        
        # Record energy cost
        energy = self.cost_table.get(action.legacy_action_id, 0.0)
        
        # Create step
        step = Step(
            state=state,
            action=action,
            observation=observation,
            result=result,
            energy_wh=energy
        )
        trajectory.steps.append(step)
        trajectory.total_energy_wh += energy
        
        # Update state
        new_state = f"{state} {observation} [SEP]"
        
        return result, observation, new_state, trajectory
    
    def _execute_action(
        self, 
        action: ActionCall, 
        query: str, 
        state: str
    ) -> Tuple[str, str]:
        """
        Execute a parameterized action and return (result, compressed_observation).
        
        Args:
            action: ActionCall to execute (with parameters)
            query: Original query
            state: Current state (for context)
            
        Returns:
            Tuple of (raw_result, compressed_observation)
        """
        if action.action_type == ActionType.RETRIEVE:
            # Use the query from the action (sub-question) or fall back to original
            retrieve_query = action.query or query
            top_k = action.top_k  # Use the top_k from the action
            if action.method == RetrievalMethod.KEYWORD:
                return self._execute_retrieve_keyword(retrieve_query, top_k=top_k)
            else:
                return self._execute_retrieve_dense(retrieve_query, top_k=top_k)
        
        elif action.action_type == ActionType.GENERATE:
            model = self.slm if action.model == Model.SLM else self.llm
            model_name = "SLM" if action.model == Model.SLM else "LLM"
            return self._execute_generate(query, model, model_name)
        
        elif action.action_type == ActionType.DECOMPOSE:
            model = self.slm if action.model == Model.SLM else self.llm
            model_name = "SLM" if action.model == Model.SLM else "LLM"
            return self._execute_decompose(query, model, model_name)
        
        elif action.action_type == ActionType.REASON:
            model = self.slm if action.model == Model.SLM else self.llm
            model_name = "SLM" if action.model == Model.SLM else "LLM"
            return self._execute_reason(query, model, model_name)
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _execute_retrieve_keyword(self, query: str, top_k: int = 3) -> Tuple[str, str]:
        """Execute BM25 keyword retrieval for the given query."""
        try:
            passages, scores = self.retriever.retrieve(query, top_k=top_k)
            self._current_context.extend(passages)
            
            # Compress retrieval results
            if passages:
                titles = [p.get('title', 'Unknown')[:30] for p in passages[:top_k]]
                # Include the query in observation so controller knows what was searched
                query_preview = query[:40] + "..." if len(query) > 40 else query
                observation = f"BM25 '{query_preview}' (k={top_k}): {', '.join(titles)}."
                if scores and scores[0] < 5.0:
                    observation += " Low scores."
                observation = self._truncate_observation(observation)
            else:
                observation = f"BM25 '{query[:30]}': No docs found."
            
            return json.dumps([p.get('text', '')[:200] for p in passages[:top_k]]), observation
            
        except Exception as e:
            logger.error(f"Retrieve keyword failed: {e}")
            return "", f"Retrieval failed: {str(e)[:30]}"
    
    def _execute_retrieve_dense(self, query: str, top_k: int = 3) -> Tuple[str, str]:
        """Execute dense vector retrieval for the given query."""
        retriever = self.dense_retriever or self.retriever
        retrieve_fn = retriever.retrieve_dense if hasattr(retriever, 'retrieve_dense') else retriever.retrieve
        
        try:
            passages, scores = retrieve_fn(query, top_k=top_k)
            self._current_context.extend(passages)
            
            if passages:
                titles = [p.get('title', 'Unknown')[:30] for p in passages[:top_k]]
                query_preview = query[:40] + "..." if len(query) > 40 else query
                observation = f"Dense '{query_preview}' (k={top_k}): {', '.join(titles)}."
                observation = self._truncate_observation(observation)
            else:
                observation = f"Dense '{query[:30]}': No docs found."
            
            return json.dumps([p.get('text', '')[:200] for p in passages[:top_k]]), observation
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return "", f"Dense retrieval failed: {str(e)[:30]}"
    
    def _execute_generate(
        self, 
        query: str, 
        model, 
        model_name: str
    ) -> Tuple[str, str]:
        """Execute answer generation with SLM or LLM."""
        try:
            # Build context from accumulated passages
            context_passages = self._current_context[-5:] if self._current_context else []
            
            # Generate answer with explicit prompt to use context faithfully
            result = model.generate(query, context_passages, prompt_template=GENERATE_PROMPT)
            answer = result.get('answer', '') if isinstance(result, dict) else str(result)
            
            # Compress to observation
            observation = f"Generated ({model_name}): {answer[:80]}"
            observation = self._truncate_observation(observation)
            
            return answer, observation
            
        except Exception as e:
            logger.error(f"Generate ({model_name}) failed: {e}")
            return "", f"Generation failed ({model_name}): {str(e)[:20]}"
    
    def _execute_decompose(
        self, 
        query: str, 
        model, 
        model_name: str
    ) -> Tuple[str, str]:
        """Decompose query into sub-questions for better retrieval."""
        try:
            prompt = DECOMPOSE_PROMPT.format(question=query)
            result = model.generate(prompt, [])  # No context needed for decomposition
            
            sub_qs_raw = result.get('answer', '') if isinstance(result, dict) else str(result)
            
            # Parse sub-questions (split by newlines or numbered list)
            sub_qs_list = []
            for line in sub_qs_raw.split('\n'):
                line = line.strip()
                # Remove numbering like "1." or "- "
                if line and len(line) > 5:
                    cleaned = line.lstrip('0123456789.-) ').strip()
                    if cleaned and '?' in cleaned:
                        sub_qs_list.append(cleaned)
            
            # Fall back to full text if parsing fails
            if not sub_qs_list:
                sub_qs_list = [sub_qs_raw[:200]]
            
            # Store parsed sub-questions for retrieval
            self._sub_questions.extend(sub_qs_list[:3])
            
            # Compress observation - show the sub-questions clearly
            sub_qs_summary = ' | '.join([q[:40] for q in sub_qs_list[:2]])
            observation = f"Sub-Qs ({model_name}): {sub_qs_summary}"
            observation = self._truncate_observation(observation)
            
            return sub_qs_raw, observation
            
        except Exception as e:
            logger.error(f"Decompose ({model_name}) failed: {e}")
            return "", f"Decomposition failed: {str(e)[:20]}"
    
    def _execute_reason(self, query: str, model, model_name: str) -> Tuple[str, str]:
        """Execute intermediate reasoning with SLM or LLM."""
        try:
            # Build context string
            context = self._format_context_for_reasoning()
            prompt = REASON_PROMPT.format(context=context, question=query)
            
            result = model.generate(prompt, [])
            reasoning = result.get('answer', '') if isinstance(result, dict) else str(result)
            self._reasoning_notes += f" {reasoning}"
            
            # Compress
            observation = f"Reasoning ({model_name}): {reasoning[:70]}"
            observation = self._truncate_observation(observation)
            
            return reasoning, observation
            
        except Exception as e:
            logger.error(f"Reason failed: {e}")
            return "", f"Reasoning failed: {str(e)[:20]}"
    
    def _format_context_for_reasoning(self) -> str:
        """Format accumulated context for reasoning prompt."""
        if not self._current_context:
            return "No context available."
        
        parts = []
        for i, passage in enumerate(self._current_context[-3:], 1):
            title = passage.get('title', 'Unknown')
            text = passage.get('text', '')[:300]
            parts.append(f"[{i}] {title}: {text}")
        
        return "\n".join(parts)
    
    def _extract_follow_up_queries(self, original_query: str) -> List[str]:
        """
        Extract follow-up queries from reasoning notes and original query.
        
        This identifies what information is still missing and generates
        targeted queries to find it. For example, if reasoning says
        "Scott Derrickson's nationality is not mentioned", we extract
        "Scott Derrickson" as an entity to search for directly.
        """
        follow_ups = []
        
        # Strategy 1: Extract entity names from the original query
        # Look for capitalized words/phrases that might be entities
        import re
        
        # Find potential entity names (capitalized words, excluding common words)
        common_words = {'what', 'who', 'where', 'when', 'how', 'which', 'the', 'a', 'an', 
                       'is', 'are', 'was', 'were', 'and', 'or', 'of', 'in', 'to', 'for',
                       'from', 'by', 'same', 'both', 'did', 'does', 'has', 'have', 'had'}
        
        # Extract capitalized sequences (potential names)
        words = original_query.split()
        current_entity = []
        entities = []
        
        for word in words:
            # Clean punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and clean_word[0].isupper() and clean_word.lower() not in common_words:
                current_entity.append(clean_word)
            else:
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
        if current_entity:
            entities.append(' '.join(current_entity))
        
        # Generate direct entity searches
        for entity in entities:
            if len(entity) > 2:  # Skip very short matches
                follow_ups.append(f"{entity}")  # Direct entity name search
        
        # Strategy 2: Look for "missing" patterns in reasoning notes
        if self._reasoning_notes:
            # Look for patterns like "X's nationality is not mentioned" or "no information about X"
            missing_patterns = [
                r"no (?:information|mention|data) (?:about|of|on) ([A-Z][a-zA-Z\s]+)",
                r"([A-Z][a-zA-Z\s]+)'s (?:nationality|location|age|birth) (?:is not|isn't) (?:mentioned|stated|provided)",
                r"missing.*?([A-Z][a-zA-Z\s]+)",
            ]
            
            for pattern in missing_patterns:
                matches = re.findall(pattern, self._reasoning_notes, re.IGNORECASE)
                for match in matches:
                    if len(match) > 2 and match.lower() not in common_words:
                        follow_ups.append(match.strip())
        
        # Strategy 3: If we have sub-questions that weren't well answered,
        # try more direct versions
        for sub_q in self._sub_questions:
            # Extract the subject of the sub-question
            # e.g., "What is Ed Wood's nationality?" -> "Ed Wood"
            match = re.search(r"(?:What is|Who is|Where is) ([A-Z][^?]+?)(?:'s|\?)", sub_q)
            if match:
                entity = match.group(1).strip()
                if entity and entity not in follow_ups:
                    follow_ups.append(entity)
        
        # Deduplicate while preserving order
        seen = set()
        unique_follow_ups = []
        for fq in follow_ups:
            if fq.lower() not in seen:
                seen.add(fq.lower())
                unique_follow_ups.append(fq)
        
        logger.debug(f"Extracted follow-up queries: {unique_follow_ups}")
        return unique_follow_ups[:3]  # Limit to 3 follow-up queries
    
    def _truncate_observation(self, observation: str) -> str:
        """Truncate observation to max_obs_tokens (approximate by characters)."""
        # Rough approximation: ~4 chars per token
        max_chars = self.max_obs_tokens * 4
        if len(observation) > max_chars:
            return observation[:max_chars-3] + "..."
        return observation
    
    def batch_search(
        self, 
        queries: List[str], 
        ground_truths: List[str],
        save_path: Optional[str] = None
    ) -> List[Trajectory]:
        """
        Search for trajectories on a batch of queries.
        
        Args:
            queries: List of questions
            ground_truths: List of expected answers
            save_path: Optional path to save trajectories as JSON
            
        Returns:
            List of Trajectory objects
        """
        if len(queries) != len(ground_truths):
            raise ValueError("queries and ground_truths must have same length")
        
        trajectories = []
        correct_count = 0
        total_energy = 0.0
        
        for i, (query, gt) in enumerate(zip(queries, ground_truths)):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            
            trajectory = self.search(query, gt)
            trajectories.append(trajectory)
            
            if trajectory.correct:
                correct_count += 1
            total_energy += trajectory.total_energy_wh
        
        # Summary stats
        accuracy = correct_count / len(queries) if queries else 0
        avg_energy = total_energy / len(queries) if queries else 0
        logger.info(f"Batch complete: accuracy={accuracy:.2%}, "
                   f"avg_energy={avg_energy:.6f} Wh, "
                   f"total_energy={total_energy:.6f} Wh")
        
        # Save if requested
        if save_path:
            self._save_trajectories(trajectories, save_path)
        
        return trajectories
    
    def _save_trajectories(self, trajectories: List[Trajectory], path: str):
        """Save trajectories to JSON file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "trajectories": [t.to_dict() for t in trajectories],
            "summary": {
                "total": len(trajectories),
                "correct": sum(1 for t in trajectories if t.correct),
                "accuracy": sum(1 for t in trajectories if t.correct) / len(trajectories) if trajectories else 0,
                "total_energy_wh": sum(t.total_energy_wh for t in trajectories),
                "avg_energy_wh": sum(t.total_energy_wh for t in trajectories) / len(trajectories) if trajectories else 0
            }
        }
        
        with open(path_obj, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(trajectories)} trajectories to {path}")
    
    @staticmethod
    def load_trajectories(path: str) -> Tuple[List[Trajectory], Dict]:
        """
        Load trajectories from JSON file.
        
        Handles both legacy format (action as int) and new format (action as dict).
        
        Args:
            path: Path to JSON file
            
        Returns:
            Tuple of (list of Trajectory objects, summary dict)
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        trajectories = []
        for t_dict in data["trajectories"]:
            trajectory = Trajectory(
                query=t_dict["query"],
                ground_truth=t_dict["ground_truth"],
                correct=t_dict["correct"],
                total_energy_wh=t_dict["total_energy_wh"]
            )
            for s_dict in t_dict["steps"]:
                # Handle both old (action as int) and new (action_id, action_params) formats
                if "action_id" in s_dict:
                    # New format with parameterized actions
                    params = s_dict.get("action_params", {})
                    action = ActionCall.from_legacy_id(
                        s_dict["action_id"],
                        query=params.get("query")
                    )
                elif isinstance(s_dict.get("action"), int):
                    # Legacy format - action as int
                    action = ActionCall.from_legacy_id(s_dict["action"])
                else:
                    # Very old format - try to parse
                    action = ActionCall.from_legacy_id(0)
                
                step = Step(
                    state=s_dict["state"],
                    action=action,
                    observation=s_dict["observation"],
                    result=s_dict.get("result"),
                    energy_wh=s_dict.get("energy_wh", 0.0)
                )
                trajectory.steps.append(step)
            trajectories.append(trajectory)
        
        return trajectories, data.get("summary", {})
    
    def extract_training_data(
        self, 
        trajectories: List[Trajectory],
        only_correct: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract training pairs from trajectories for behavior cloning.
        
        Args:
            trajectories: List of Trajectory objects
            only_correct: If True, only use trajectories that found correct answers
            
        Returns:
            List of dicts with 'text' and 'label' for training
        """
        training_data = []
        
        for trajectory in trajectories:
            if only_correct and not trajectory.correct:
                continue
            training_data.extend(trajectory.to_training_pairs())
        
        logger.info(f"Extracted {len(training_data)} training pairs from "
                   f"{len(trajectories)} trajectories")
        return training_data


def create_simple_judge(exact_match: bool = True):
    """
    Create a simple judge function for answer correctness.
    
    Args:
        exact_match: If True, require exact string match (case-insensitive)
                    If False, check if ground truth is contained in answer
    
    Returns:
        Judge function(answer, ground_truth) -> bool
    """
    def judge(answer: str, ground_truth: str) -> bool:
        if not answer or not ground_truth:
            return False
        
        answer_clean = answer.lower().strip()
        gt_clean = ground_truth.lower().strip()
        
        if exact_match:
            return answer_clean == gt_clean
        else:
            # Check if ground truth appears in answer
            return gt_clean in answer_clean
    
    return judge


def load_cost_table(path: str) -> Dict[int, float]:
    """
    Load cost table from JSON file.
    
    Args:
        path: Path to cost_table.json
        
    Returns:
        Dict mapping action ID (int) to energy cost in Wh
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Convert string keys to int, extract avg_wh
    cost_table = {}
    for key, value in data.items():
        action_id = int(key)
        if isinstance(value, dict):
            cost_table[action_id] = value.get('avg_wh', value.get('avg_joules', 0) / 3600)
        else:
            cost_table[action_id] = float(value)
    
    return cost_table
