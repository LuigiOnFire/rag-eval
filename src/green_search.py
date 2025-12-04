"""
GreenSearch: Cost-Priority Search for Minimum-Energy RAG Trajectories

This module implements a priority queue search algorithm that finds the
cheapest correct path through the RAG action space. Unlike DeepRAG's binary
(retrieve/don't retrieve) search, GreenSearch explores a richer action space
while guaranteeing the first solution found is the minimum-cost solution.

Key Features:
- Cost-priority expansion (Uniform Cost Search / A*)
- Rich action space (8 actions with parameters)
- Sub-question tracking for decomposition
- History tracking to prevent infinite loops
- Compressed state representation for BERT controller

Usage:
    from src.green_search import GreenSearch
    
    searcher = GreenSearch(retriever, slm, llm, cost_table)
    trajectory = searcher.search(query, ground_truth)
"""

import heapq
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Protocol, runtime_checkable
from enum import IntEnum

logger = logging.getLogger(__name__)


# =============================================================================
# Type Protocols for Components
# =============================================================================

@runtime_checkable
class GeneratorProtocol(Protocol):
    """Protocol for LLM/SLM generators."""
    def generate(
        self, 
        question: str, 
        context_passages: List[Dict],
        prompt_template: Optional[str] = None
    ) -> Dict:
        """Generate answer given question and context."""
        ...


@runtime_checkable  
class RetrieverProtocol(Protocol):
    """Protocol for retrievers (BM25 or Dense)."""
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[List[Dict], List[float]]:
        """Retrieve passages for a query."""
        ...


# =============================================================================
# Action Types (reuse from green_tree_search for compatibility)
# =============================================================================

class ActionType(IntEnum):
    """High-level action categories."""
    GENERATE = 0
    DECOMPOSE = 1
    RETRIEVE = 2
    REASON = 3


class Model(IntEnum):
    """Model choices for generation/reasoning."""
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
    
    Immutable so it can be used in sets for deduplication.
    """
    action_type: ActionType
    model: Optional[Model] = None
    method: Optional[RetrievalMethod] = None
    query: Optional[str] = None
    top_k: int = 3
    
    @property
    def is_terminal(self) -> bool:
        """Terminal actions produce a final answer."""
        return self.action_type == ActionType.GENERATE
    
    @property
    def legacy_action_id(self) -> int:
        """Map to 8-class action space for behavior cloning."""
        if self.action_type == ActionType.GENERATE:
            return 0 if self.model == Model.SLM else 1
        elif self.action_type == ActionType.DECOMPOSE:
            return 2 if self.model == Model.SLM else 3
        elif self.action_type == ActionType.RETRIEVE:
            return 4 if self.method == RetrievalMethod.KEYWORD else 5
        elif self.action_type == ActionType.REASON:
            return 6 if self.model == Model.SLM else 7
        return -1
    
    def __str__(self) -> str:
        if self.action_type == ActionType.GENERATE:
            model_name = self.model.name if self.model is not None else "SLM"
            return f"GENERATE({model_name})"
        elif self.action_type == ActionType.DECOMPOSE:
            model_name = self.model.name if self.model is not None else "SLM"
            return f"DECOMPOSE({model_name})"
        elif self.action_type == ActionType.RETRIEVE:
            method_name = self.method.name if self.method is not None else "KEYWORD"
            q_short = f"'{self.query[:25]}...'" if self.query and len(self.query) > 25 else f"'{self.query}'"
            k_str = f", k={self.top_k}" if self.top_k != 3 else ""
            return f"RETRIEVE({method_name}, {q_short}{k_str})"
        elif self.action_type == ActionType.REASON:
            model_name = self.model.name if self.model is not None else "SLM"
            return f"REASON({model_name})"
        return f"Action({self.action_type})"


# Convenience constructors
def GENERATE(model: Model = Model.SLM) -> ActionCall:
    return ActionCall(ActionType.GENERATE, model=model)

def DECOMPOSE(model: Model = Model.SLM) -> ActionCall:
    return ActionCall(ActionType.DECOMPOSE, model=model)

def RETRIEVE(method: RetrievalMethod = RetrievalMethod.KEYWORD, 
             query: Optional[str] = None, top_k: int = 3) -> ActionCall:
    return ActionCall(ActionType.RETRIEVE, method=method, query=query, top_k=top_k)

def REASON(model: Model = Model.SLM) -> ActionCall:
    return ActionCall(ActionType.REASON, model=model)


# =============================================================================
# Search Node
# =============================================================================

@dataclass
class SearchNode:
    """
    Node in the cost-priority search tree.
    
    Each node represents a state in the RAG reasoning process.
    Nodes track their lineage for trajectory reconstruction.
    """
    
    # Identity
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Query information
    query: str = ""  # Current query (may be sub-question)
    original_query: str = ""  # Root query for the entire search
    
    # Lineage (for trajectory reconstruction)
    parent: Optional['SearchNode'] = field(default=None, repr=False)
    action_taken: Optional[ActionCall] = None  # Action that led to this node
    depth: int = 0
    
    # Accumulated state
    context: List[Dict] = field(default_factory=list)  # Retrieved passages
    sub_questions: List[str] = field(default_factory=list)  # Pending sub-questions
    sub_answers: Dict[str, str] = field(default_factory=dict)  # Completed sub-answers
    reasoning_notes: str = ""  # Accumulated reasoning
    
    # History for deduplication
    action_history: List[ActionCall] = field(default_factory=list)
    retrieved_queries: set = field(default_factory=set)  # Queries already retrieved
    decomposed_queries: set = field(default_factory=set)  # Queries already decomposed
    
    # Cost tracking (THE GREEN PART)
    accumulated_cost: float = 0.0
    accumulated_energy_wh: float = 0.0
    
    # Result (for terminal nodes)
    answer: Optional[str] = None
    observation: Optional[str] = None  # Compressed observation from action
    is_correct: Optional[bool] = None
    
    def __lt__(self, other: 'SearchNode') -> bool:
        """For priority queue ordering."""
        return self.accumulated_cost < other.accumulated_cost
    
    def get_compressed_state(self, max_observations: int = 5) -> str:
        """
        Get BERT-compatible state string.
        
        Uses sliding window to keep only recent observations.
        Roughly targets 512 tokens (~2000 chars).
        """
        parts = [f"[CLS] {self.original_query}"]
        
        # Add sub-question context if we're solving a sub-question
        if self.query != self.original_query:
            parts.append(f"Sub-Q: {self.query}")
        
        # Add completed sub-answers if any
        if self.sub_answers:
            answers_str = " | ".join([f"{q[:30]}: {a[:30]}" 
                                      for q, a in list(self.sub_answers.items())[:2]])
            parts.append(f"Answered: {answers_str}")
        
        # Add recent observations (sliding window)
        observations = self._get_recent_observations(max_observations)
        parts.extend(observations)
        
        state = " [SEP] ".join(parts) + " [SEP]"
        
        # Hard truncate if too long (512 tokens ≈ 2000 chars)
        max_chars = 2000
        if len(state) > max_chars:
            state = state[:max_chars-3] + "..."
        
        return state
    
    def _get_recent_observations(self, max_obs: int) -> List[str]:
        """Extract observations from recent action history."""
        observations = []
        
        # Walk back through parent chain to collect observations
        node = self
        while node and len(observations) < max_obs:
            if node.observation:
                observations.append(node.observation)
            node = node.parent
        
        # Reverse to get chronological order
        return list(reversed(observations))
    
    def can_take_action(self, action: ActionCall, max_depth: int = 10) -> bool:
        """
        Check if action is valid from this state.
        
        Prevents infinite loops and enforces depth limits.
        """
        # Depth limit: only allow GENERATE at max depth
        if self.depth >= max_depth:
            return action.action_type == ActionType.GENERATE
        
        # Don't retrieve the exact same query twice
        if action.action_type == ActionType.RETRIEVE:
            if action.query and action.query in self.retrieved_queries:
                return False
        
        # Don't decompose the same query twice
        if action.action_type == ActionType.DECOMPOSE:
            if self.query in self.decomposed_queries:
                return False
        
        # Don't reason twice in a row without new information
        if action.action_type == ActionType.REASON:
            if self.action_taken and self.action_taken.action_type == ActionType.REASON:
                return False
        
        return True
    
    def create_child(
        self,
        action: ActionCall,
        observation: str,
        cost: float,
        energy_wh: float,
        new_context: Optional[List[Dict]] = None,
        new_sub_questions: Optional[List[str]] = None,
        new_sub_answer: Optional[Tuple[str, str]] = None,
        new_reasoning: Optional[str] = None,
        answer: Optional[str] = None,
        new_query: Optional[str] = None,
        is_correct: Optional[bool] = None,
    ) -> 'SearchNode':
        """
        Create a child node from this node after taking an action.
        
        Properly inherits and updates state.
        """
        # Copy mutable state
        new_retrieved = self.retrieved_queries.copy()
        new_decomposed = self.decomposed_queries.copy()
        new_sub_answers = self.sub_answers.copy()
        
        # Update based on action
        if action.action_type == ActionType.RETRIEVE and action.query:
            new_retrieved.add(action.query)
        if action.action_type == ActionType.DECOMPOSE:
            new_decomposed.add(self.query)
        if new_sub_answer:
            new_sub_answers[new_sub_answer[0]] = new_sub_answer[1]
        
        child = SearchNode(
            # Query
            query=new_query or self.query,
            original_query=self.original_query,
            
            # Lineage
            parent=self,
            action_taken=action,
            depth=self.depth + 1,
            
            # State
            context=self.context + (new_context or []),
            sub_questions=new_sub_questions if new_sub_questions is not None else self.sub_questions.copy(),
            sub_answers=new_sub_answers,
            reasoning_notes=self.reasoning_notes + (f" {new_reasoning}" if new_reasoning else ""),
            
            # History
            action_history=self.action_history + [action],
            retrieved_queries=new_retrieved,
            decomposed_queries=new_decomposed,
            
            # Cost
            accumulated_cost=self.accumulated_cost + cost,
            accumulated_energy_wh=self.accumulated_energy_wh + energy_wh,
            
            # Result
            observation=observation,
            answer=answer,
            is_correct=is_correct,
        )
        
        return child
    
    def get_trajectory(self) -> List['SearchNode']:
        """Backtrack to root to extract the path."""
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))


# =============================================================================
# Trajectory Output
# =============================================================================

@dataclass
class TrajectoryStep:
    """A single step in a trajectory for training."""
    state: str  # Compressed state for controller
    action: ActionCall
    action_id: int  # Legacy 8-class ID
    observation: str
    cost: float
    cumulative_cost: float
    energy_wh: float
    cumulative_energy_wh: float


@dataclass 
class Trajectory:
    """Complete trajectory from query to answer."""
    query: str
    ground_truth: str
    steps: List[TrajectoryStep]
    answer: str
    is_correct: bool
    total_cost: float
    total_energy_wh: float
    nodes_explored: int
    search_depth: int
    
    def to_training_pairs(self) -> List[Dict[str, Any]]:
        """Convert to (state, action) pairs for behavior cloning."""
        return [
            {
                "text": step.state,
                "label": step.action_id,
                "action_str": str(step.action),
                "observation": step.observation,
                "cost": step.cost,
                "cumulative_cost": step.cumulative_cost,
            }
            for step in self.steps
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "answer": self.answer,
            "is_correct": self.is_correct,
            "total_cost": self.total_cost,
            "total_energy_wh": self.total_energy_wh,
            "nodes_explored": self.nodes_explored,
            "search_depth": self.search_depth,
            "steps": [
                {
                    "state": s.state,
                    "action": str(s.action),
                    "action_id": s.action_id,
                    "observation": s.observation,
                    "cost": s.cost,
                    "cumulative_cost": s.cumulative_cost,
                }
                for s in self.steps
            ]
        }


# =============================================================================
# Judge
# =============================================================================

class Judge:
    """
    Multi-level judge for answer correctness.
    
    Levels:
    1. Fast rejection (empty, "don't know")
    2. Substring match
    3. LLM-as-judge (optional, for semantic equivalence)
    """
    
    def __init__(
        self, 
        llm: Optional[GeneratorProtocol] = None, 
        use_llm_judge: bool = True
    ):
        """
        Initialize judge.
        
        Args:
            llm: LLM for semantic judgment (optional)
            use_llm_judge: Whether to use LLM for ambiguous cases
        """
        self.llm = llm
        self.use_llm_judge = use_llm_judge and llm is not None
    
    def __call__(self, generated: str, ground_truth: str) -> bool:
        """Check if generated answer is correct."""
        return self.judge(generated, ground_truth)
    
    def judge(self, generated: str, ground_truth: str) -> bool:
        """
        Multi-level correctness check.
        """
        if not generated:
            return False
        
        generated_lower = generated.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Level 1: Fast rejection - explicit "don't know" responses
        rejection_phrases = [
            "i don't have enough information",
            "i don't know",
            "cannot determine",
            "unable to answer",
            "not enough context",
            "no information",
        ]
        for phrase in rejection_phrases:
            if phrase in generated_lower:
                return False
        
        # Level 2: Substring match (ground truth in generated)
        if ground_truth_lower in generated_lower:
            return True
        
        # Level 2b: Generated contains key part of ground truth
        # Handle cases like ground_truth="yes" and generated="Yes, they are both American"
        gt_words = ground_truth_lower.split()
        if len(gt_words) <= 3:
            # Short ground truth - check if all words appear
            if all(word in generated_lower for word in gt_words):
                return True
        
        # Level 3: LLM-as-judge for semantic equivalence
        if self.use_llm_judge:
            return self._llm_judge(generated, ground_truth)
        
        return False
    
    def _llm_judge(self, generated: str, ground_truth: str) -> bool:
        """Use LLM to judge semantic equivalence."""
        prompt = f"""Determine if the generated answer correctly answers the question with the expected answer.

Generated Answer: {generated}
Expected Answer: {ground_truth}

Consider:
- Semantic equivalence (same meaning, different words)
- The generated answer may contain extra information
- Focus on whether the core answer is correct

Reply with ONLY "YES" or "NO"."""
        
        try:
            # Generator interface: generate(question, context_passages)
            result = self.llm.generate(prompt, [])  # type: ignore[union-attr]
            answer = result.get('answer', '') if isinstance(result, dict) else str(result)
            return "YES" in answer.upper()
        except Exception as e:
            logger.warning(f"LLM judge failed: {e}, falling back to False")
            return False


# =============================================================================
# Green Search Algorithm
# =============================================================================

# Fallback cost table (used if no cost file is found)
_FALLBACK_COST_TABLE = {
    0: 0.020,   # GENERATE_SLM
    1: 0.015,   # GENERATE_LLM
    2: 0.024,   # DECOMPOSE_SLM
    3: 0.028,   # DECOMPOSE_LLM
    4: 0.009,   # RETRIEVE_KEYWORD
    5: 0.009,   # RETRIEVE_DENSE
    6: 0.065,   # REASON_SLM
    7: 0.026,   # REASON_LLM
}


def load_cost_table(path: Optional[str] = None) -> Dict[int, float]:
    """
    Load cost table from JSON file generated by benchmark_costs.py.
    
    Args:
        path: Path to cost_table.json. If None, tries default locations.
        
    Returns:
        Dict mapping action_id -> avg_wh cost
    """
    # Default paths to search
    search_paths = [
        path,
        "results/cost_table.json",
        Path(__file__).parent.parent / "results" / "cost_table.json",
    ]
    
    for p in search_paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            try:
                with open(p) as f:
                    raw_data = json.load(f)
                
                # Extract avg_wh from each action's benchmark data
                cost_table = {}
                for action_id_str, stats in raw_data.items():
                    action_id = int(action_id_str)
                    # The benchmark outputs "avg_wh" for each action
                    cost_table[action_id] = stats.get("avg_wh", _FALLBACK_COST_TABLE.get(action_id, 0.01))
                
                logger.info(f"Loaded cost table from {p}")
                return cost_table
            except Exception as e:
                logger.warning(f"Failed to load cost table from {p}: {e}")
                continue
    
    logger.warning("No cost table found, using fallback values")
    return _FALLBACK_COST_TABLE.copy()


# Default cost table - loaded lazily
_loaded_cost_table: Optional[Dict[int, float]] = None


def get_default_cost_table() -> Dict[int, float]:
    """Get the default cost table, loading from file if available."""
    global _loaded_cost_table
    if _loaded_cost_table is None:
        _loaded_cost_table = load_cost_table()
    return _loaded_cost_table


class GreenSearch:
    """
    Cost-Priority Search for minimum-energy RAG trajectories.
    
    Uses Uniform Cost Search (priority queue ordered by accumulated cost)
    to find the cheapest correct path through the action space.
    
    The first correct solution found is guaranteed to be the minimum-cost
    solution (assuming non-negative action costs).
    """
    
    def __init__(
        self,
        retriever: RetrieverProtocol,
        slm: GeneratorProtocol,
        llm: GeneratorProtocol,
        cost_table: Optional[Dict[int, float]] = None,
        judge: Optional['Judge'] = None,
        dense_retriever: Optional[RetrieverProtocol] = None,
        max_depth: int = 10,
        max_nodes: int = 500,
        max_observations: int = 5,
    ):
        """
        Initialize GreenSearch.
        
        Args:
            retriever: BM25/keyword retriever (must have .retrieve(query, top_k) method)
            slm: Small language model (must have .generate(question, context_passages) method)
            llm: Large language model (must have .generate(question, context_passages) method)
            cost_table: Action ID -> cost mapping (loaded from results/cost_table.json by default)
            judge: Judge for answer correctness
            dense_retriever: Optional dense retriever
            max_depth: Maximum search depth
            max_nodes: Maximum nodes to explore
            max_observations: Max observations in state (for BERT window)
        """
        self.retriever = retriever
        self.slm = slm
        self.llm = llm
        self.cost_table = cost_table or get_default_cost_table()
        self.judge = judge or Judge(llm=llm)
        self.dense_retriever = dense_retriever
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_observations = max_observations
    
    def _get_state_key(self, node: SearchNode) -> str:
        """
        Get a hashable state key for deduplication.
        
        States are equivalent if they have the same:
        - Current query
        - Retrieved queries (what info we have)
        - Decomposed queries (what's been broken down)
        """
        # Frozen set of retrieved and decomposed queries
        retrieved = frozenset(node.retrieved_queries)
        decomposed = frozenset(node.decomposed_queries)
        # Include context length to differentiate states with different amounts of context
        context_len = len(node.context)
        
        return f"{node.query}|r={retrieved}|d={decomposed}|c={context_len}"
    
    def search(self, query: str, ground_truth: str) -> Optional[Trajectory]:
        """
        Find the cheapest correct path for a query.
        
        Args:
            query: The question to answer
            ground_truth: Expected answer for judging
            
        Returns:
            Trajectory if solution found, None otherwise
        """
        # Initialize
        root = SearchNode(
            query=query,
            original_query=query,
            accumulated_cost=0.0,
        )
        
        # Priority queue: (cost, counter, node)
        # counter for FIFO tie-breaking (avoid comparing nodes directly)
        counter = 0
        frontier = [(0.0, counter, root)]
        heapq.heapify(frontier)
        
        # CRITICAL: Track visited states to prevent infinite loops
        visited: set = set()
        visited.add(self._get_state_key(root))
        
        nodes_explored = 0
        
        logger.info(f"Starting GreenSearch for: {query[:50]}...")
        
        while frontier and nodes_explored < self.max_nodes:
            cost, _, node = heapq.heappop(frontier)
            nodes_explored += 1
            
            if nodes_explored % 10 == 0:
                logger.info(f"Progress: explored {nodes_explored} nodes, frontier size: {len(frontier)}, depth: {node.depth}")
            
            logger.debug(f"Exploring node {node.node_id} at depth {node.depth}, cost {cost:.4f}")
            
            # Expand valid actions (EAGER execution, but limited action set)
            children = self._expand_node(node, ground_truth)
            
            for child in children:
                # Check if this is a correct terminal node
                if child.answer is not None and child.is_correct:
                    logger.info(f"Found solution at depth {child.depth}, cost {child.accumulated_cost:.4f}, explored {nodes_explored} nodes")
                    return self._build_trajectory(child, nodes_explored)
                
                # Skip if we've already seen this state (at any cost)
                state_key = self._get_state_key(child)
                if state_key in visited:
                    continue
                visited.add(state_key)
                
                # Add to frontier for further exploration
                counter += 1
                heapq.heappush(frontier, (child.accumulated_cost, counter, child))
        
        logger.warning(f"No solution found after exploring {nodes_explored} nodes")
        return None
    
    def _expand_node(self, node: SearchNode, ground_truth: str) -> List[SearchNode]:
        """
        Generate child nodes for SELECTIVE actions based on state.
        
        Uses a SMART action selection strategy to limit branching:
        - Always try GENERATE_LLM (cheapest terminal action)
        - If no context: try RETRIEVE
        - If context but first time: try REASON_LLM
        - Decompose only at root or depth 0
        
        This limits to 2-3 actions per node instead of 7+.
        """
        children = []
        
        # Get SMART action set based on state
        actions = self._get_smart_actions(node)
        
        for action in actions:
            if not node.can_take_action(action, self.max_depth):
                continue
            
            try:
                child = self._execute_action(node, action, ground_truth)
                if child:
                    children.append(child)
            except Exception as e:
                logger.warning(f"Action {action} failed: {e}")
                continue
        
        return children
    
    def _get_smart_actions(self, node: SearchNode) -> List[ActionCall]:
        """
        Get a SMART set of actions based on current state.
        
        Supports multi-hop reasoning:
        - Can RETRIEVE multiple times on DIFFERENT queries (original, sub-questions)
        - DECOMPOSE creates sub-questions which become new queries to retrieve
        - Track retrieved_queries to avoid duplicate retrievals
        
        Strategy:
        1. Always try GENERATE_LLM (can terminate at any point)
        2. RETRIEVE current query if not yet retrieved
        3. RETRIEVE original query if different and not retrieved
        4. REASON if we have context (synthesize information)
        5. DECOMPOSE to create sub-questions (depth-limited)
        
        This allows multi-hop while limiting to 3-5 actions per node.
        """
        actions = []
        
        # 1. Always offer GENERATE_LLM (usually cheapest terminal action)
        actions.append(GENERATE(Model.LLM))
        
        # 2. RETRIEVE current query if not yet retrieved
        #    This handles sub-questions after DECOMPOSE
        if node.query not in node.retrieved_queries:
            actions.append(RETRIEVE(RetrievalMethod.KEYWORD, query=node.query, top_k=5))
        
        # 3. RETRIEVE original query if we're on a sub-question and haven't retrieved original
        #    This supports "retrieve both parts of the question" pattern
        if node.query != node.original_query and node.original_query not in node.retrieved_queries:
            actions.append(RETRIEVE(RetrievalMethod.KEYWORD, query=node.original_query, top_k=5))
        
        # 4. REASON if we have context and haven't just reasoned
        #    Helps synthesize info from multiple retrievals
        if node.context and (not node.action_taken or node.action_taken.action_type != ActionType.REASON):
            actions.append(REASON(Model.LLM))
        
        # 5. DECOMPOSE to create sub-questions
        #    Only at shallow depth, and only once per query
        if node.depth <= 2 and node.query not in node.decomposed_queries:
            actions.append(DECOMPOSE(Model.LLM))
        
        # 6. If we have pending sub-questions, we can "switch" to next one
        #    (This is handled implicitly via decompose creating new queries)
        
        return actions
    
    def _get_possible_actions(self, node: SearchNode) -> List[ActionCall]:
        """
        Get all possible actions from current state.
        
        Returns different actions based on state:
        - Always can try GENERATE
        - Can RETRIEVE with current query
        - Can DECOMPOSE if not already decomposed
        - Can REASON if have context
        
        NOTE: _get_smart_actions() is used instead for reduced branching.
        """
        actions = []
        
        # GENERATE actions (always available)
        actions.append(GENERATE(Model.SLM))
        actions.append(GENERATE(Model.LLM))
        
        # RETRIEVE actions - only one top_k to reduce branching factor
        current_query = node.query
        actions.append(RETRIEVE(RetrievalMethod.KEYWORD, query=current_query, top_k=5))
        
        # Dense retrieval if available
        if self.dense_retriever:
            actions.append(RETRIEVE(RetrievalMethod.DENSE, query=current_query, top_k=5))
        
        # DECOMPOSE actions
        actions.append(DECOMPOSE(Model.SLM))
        actions.append(DECOMPOSE(Model.LLM))
        
        # REASON actions (only if we have some context)
        if node.context:
            actions.append(REASON(Model.SLM))
            actions.append(REASON(Model.LLM))
        
        return actions
    
    def _execute_action(
        self, 
        node: SearchNode, 
        action: ActionCall,
        ground_truth: str
    ) -> Optional[SearchNode]:
        """
        Execute an action and create a child node.
        
        Returns None if action fails.
        """
        cost = self.cost_table.get(action.legacy_action_id, 0.01)
        energy_wh = cost  # Using cost as proxy for energy
        
        if action.action_type == ActionType.GENERATE:
            return self._execute_generate(node, action, cost, energy_wh, ground_truth)
        
        elif action.action_type == ActionType.RETRIEVE:
            return self._execute_retrieve(node, action, cost, energy_wh)
        
        elif action.action_type == ActionType.DECOMPOSE:
            return self._execute_decompose(node, action, cost, energy_wh)
        
        elif action.action_type == ActionType.REASON:
            return self._execute_reason(node, action, cost, energy_wh)
        
        return None
    
    def _execute_generate(
        self, 
        node: SearchNode, 
        action: ActionCall, 
        cost: float,
        energy_wh: float,
        ground_truth: str
    ) -> SearchNode:
        """
        Execute GENERATE action.
        
        Handles two cases:
        1. On original query: Check correctness against ground_truth (terminal)
        2. On sub-question: Store answer and switch to next sub-q (non-terminal)
        """
        model = self.slm if action.model == Model.SLM else self.llm
        model_name = "SLM" if action.model == Model.SLM else "LLM"
        
        # Build context from accumulated passages
        context_passages = node.context[-5:] if node.context else []
        
        # Are we on a sub-question or the original?
        is_sub_question = node.query != node.original_query
        
        # Build prompt
        if is_sub_question:
            # Answering a sub-question - just get the answer
            query = node.query
            if context_passages:
                prompt = f"""Answer the question based on the context below.
Be concise - just give the answer.

Context:
{self._format_context(context_passages)}

Question: {query}

Answer:"""
            else:
                prompt = f"""Answer this question concisely. Just give the answer.

Question: {query}

Answer:"""
        else:
            # Answering original question - use sub-answers if available
            query = node.query
            if node.sub_answers:
                sub_info = "\n".join([f"- {q}: {a}" for q, a in node.sub_answers.items()])
                prompt = f"""Answer the question based on the information below.
Be concise - just give the answer (yes/no if appropriate).

Relevant information:
{sub_info}

Question: {query}

Answer:"""
            elif context_passages:
                prompt = f"""Answer the question based on the context below.
Be concise - just give the answer, no explanation needed.

Context:
{self._format_context(context_passages)}

Question: {query}

Answer:"""
            else:
                prompt = f"""Answer this question concisely. Just give the answer, no explanation.

Question: {query}

Answer:"""
        
        result = model.generate(prompt, [])
        answer = result.get('answer', '') if isinstance(result, dict) else str(result)
        
        if is_sub_question:
            # Sub-question: store answer and potentially switch to next sub-question
            observation = f"SubQ Answer ({model_name}): {answer[:60]}"
            
            # Determine next query
            if node.sub_questions:
                # More sub-questions to answer
                next_query = node.sub_questions[0]
                remaining_sub_qs = node.sub_questions[1:]
            else:
                # All sub-questions answered, return to original
                next_query = node.original_query
                remaining_sub_qs = []
            
            return node.create_child(
                action=action,
                observation=observation,
                cost=cost,
                energy_wh=energy_wh,
                new_query=next_query,
                new_sub_questions=remaining_sub_qs,
                new_sub_answer=(node.query, answer),  # Store sub-question -> answer
                # NOT a terminal node - no answer/is_correct
            )
        else:
            # Original question: judge correctness
            is_correct = self.judge.judge(answer, ground_truth)
            
            # Debug logging
            logger.debug(f"GENERATE ({model_name}): '{answer[:100]}' vs ground_truth='{ground_truth}' -> {is_correct}")
            if not is_correct:
                logger.info(f"Incorrect answer: '{answer[:60]}...' (expected: '{ground_truth}')")
            
            observation = f"Generated ({model_name}): {answer[:80]}"
            
            return node.create_child(
                action=action,
                observation=observation,
                cost=cost,
                energy_wh=energy_wh,
                answer=answer,
                is_correct=is_correct,
            )
    
    def _execute_retrieve(
        self, 
        node: SearchNode, 
        action: ActionCall, 
        cost: float,
        energy_wh: float
    ) -> SearchNode:
        """Execute RETRIEVE action."""
        query = action.query or node.query
        top_k = action.top_k
        
        if action.method == RetrievalMethod.KEYWORD:
            passages, scores = self.retriever.retrieve(query, top_k=top_k)
            method_name = "BM25"
        else:
            retriever = self.dense_retriever or self.retriever
            retrieve_fn = getattr(retriever, 'retrieve_dense', retriever.retrieve)
            passages, scores = retrieve_fn(query, top_k=top_k)
            method_name = "Dense"
        
        # Create observation
        if passages:
            titles = [p.get('title', 'Unknown')[:25] for p in passages[:3]]
            observation = f"{method_name} '{query[:30]}' (k={top_k}): {', '.join(titles)}"
        else:
            observation = f"{method_name} '{query[:30]}': No results"
        
        # Update action with actual query (for deduplication tracking)
        method = action.method if action.method is not None else RetrievalMethod.KEYWORD
        actual_action = RETRIEVE(method, query=query, top_k=top_k)
        
        return node.create_child(
            action=actual_action,
            observation=observation,
            cost=cost,
            energy_wh=energy_wh,
            new_context=passages,
        )
    
    def _execute_decompose(
        self, 
        node: SearchNode, 
        action: ActionCall, 
        cost: float,
        energy_wh: float
    ) -> SearchNode:
        """Execute DECOMPOSE action."""
        model = self.slm if action.model == Model.SLM else self.llm
        model_name = "SLM" if action.model == Model.SLM else "LLM"
        
        prompt = f"""Break down this question into 2-3 simpler sub-questions that would help answer it:

Question: {node.query}

Sub-questions:"""
        
        result = model.generate(prompt, [])
        sub_qs_raw = result.get('answer', '') if isinstance(result, dict) else str(result)
        
        # Parse sub-questions
        sub_qs = []
        for line in sub_qs_raw.split('\n'):
            line = line.strip()
            cleaned = line.lstrip('0123456789.-) ').strip()
            if cleaned and '?' in cleaned and len(cleaned) > 10:
                sub_qs.append(cleaned)
        
        if not sub_qs:
            sub_qs = [node.query]  # Fallback
        
        sub_qs = sub_qs[:3]  # Limit to 3
        
        # Create observation
        sub_qs_preview = " | ".join([q[:35] for q in sub_qs[:2]])
        observation = f"Decomposed ({model_name}): {sub_qs_preview}"
        
        # First sub-question becomes the new query
        # Rest go into sub_questions list
        return node.create_child(
            action=action,
            observation=observation,
            cost=cost,
            energy_wh=energy_wh,
            new_query=sub_qs[0],
            new_sub_questions=sub_qs[1:] if len(sub_qs) > 1 else [],
        )
    
    def _execute_reason(
        self, 
        node: SearchNode, 
        action: ActionCall, 
        cost: float,
        energy_wh: float
    ) -> SearchNode:
        """Execute REASON action."""
        model = self.slm if action.model == Model.SLM else self.llm
        model_name = "SLM" if action.model == Model.SLM else "LLM"
        
        context = self._format_context(node.context[-3:])
        
        prompt = f"""Analyze the context below to extract facts relevant to the question.
Only cite facts that are EXPLICITLY stated in the context.

Context:
{context}

Question: {node.query}

Analysis:"""
        
        result = model.generate(prompt, [])
        reasoning = result.get('answer', '') if isinstance(result, dict) else str(result)
        
        observation = f"Reasoning ({model_name}): {reasoning[:70]}"
        
        return node.create_child(
            action=action,
            observation=observation,
            cost=cost,
            energy_wh=energy_wh,
            new_reasoning=reasoning,
        )
    
    def _format_context(self, passages: List[Dict]) -> str:
        """Format passages for prompts."""
        if not passages:
            return "No context available."
        
        parts = []
        for i, p in enumerate(passages, 1):
            title = p.get('title', 'Unknown')
            text = p.get('text', '')[:300]
            parts.append(f"[{i}] {title}: {text}")
        
        return "\n\n".join(parts)
    
    def _build_trajectory(self, final_node: SearchNode, nodes_explored: int) -> Trajectory:
        """Build Trajectory from final node by backtracking."""
        path = final_node.get_trajectory()
        
        steps = []
        for i, node in enumerate(path):
            if node.action_taken is None:
                continue  # Skip root node
            
            # Get state from parent (what the controller saw before taking action)
            parent = path[i-1] if i > 0 else None
            state = parent.get_compressed_state(self.max_observations) if parent else f"[CLS] {node.original_query} [SEP]"
            
            step = TrajectoryStep(
                state=state,
                action=node.action_taken,
                action_id=node.action_taken.legacy_action_id,
                observation=node.observation or "",
                cost=self.cost_table.get(node.action_taken.legacy_action_id, 0.01),
                cumulative_cost=node.accumulated_cost,
                energy_wh=node.accumulated_energy_wh - (parent.accumulated_energy_wh if parent else 0),
                cumulative_energy_wh=node.accumulated_energy_wh,
            )
            steps.append(step)
        
        return Trajectory(
            query=final_node.original_query,
            ground_truth="",  # Will be filled in by caller
            steps=steps,
            answer=final_node.answer or "",
            is_correct=final_node.is_correct or False,
            total_cost=final_node.accumulated_cost,
            total_energy_wh=final_node.accumulated_energy_wh,
            nodes_explored=nodes_explored,
            search_depth=final_node.depth,
        )
    
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
            save_path: Optional path to save results
            
        Returns:
            List of Trajectory objects (None for failed searches)
        """
        trajectories = []
        
        for i, (query, gt) in enumerate(zip(queries, ground_truths)):
            logger.info(f"Processing {i+1}/{len(queries)}: {query[:50]}...")
            
            trajectory = self.search(query, gt)
            
            if trajectory:
                trajectory.ground_truth = gt
                trajectories.append(trajectory)
                logger.info(f"  ✓ Found solution: cost={trajectory.total_cost:.4f}, depth={trajectory.search_depth}")
            else:
                logger.warning(f"  ✗ No solution found")
                trajectories.append(None)
        
        # Save if requested
        if save_path:
            self._save_trajectories(trajectories, save_path)
        
        # Summary
        successful = sum(1 for t in trajectories if t is not None)
        logger.info(f"Batch complete: {successful}/{len(queries)} successful")
        
        return trajectories
    
    def _save_trajectories(self, trajectories: List[Optional[Trajectory]], path: str):
        """Save trajectories to JSON file."""
        data = {
            "trajectories": [t.to_dict() if t else None for t in trajectories],
            "summary": {
                "total": len(trajectories),
                "successful": sum(1 for t in trajectories if t is not None),
                "avg_cost": sum(t.total_cost for t in trajectories if t) / max(1, sum(1 for t in trajectories if t)),
                "avg_depth": sum(t.search_depth for t in trajectories if t) / max(1, sum(1 for t in trajectories if t)),
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved trajectories to {path}")


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    # Test basic functionality
    print("Testing GreenSearch module...")
    
    # Test ActionCall
    a1 = GENERATE(Model.SLM)
    a2 = RETRIEVE(RetrievalMethod.KEYWORD, query="test", top_k=5)
    a3 = DECOMPOSE(Model.LLM)
    
    print(f"Action 1: {a1} -> legacy_id={a1.legacy_action_id}")
    print(f"Action 2: {a2} -> legacy_id={a2.legacy_action_id}")
    print(f"Action 3: {a3} -> legacy_id={a3.legacy_action_id}")
    
    # Test SearchNode
    node = SearchNode(query="Test query", original_query="Test query")
    print(f"\nNode: {node.node_id}, can_take_action(GENERATE)={node.can_take_action(GENERATE())}")
    
    # Test Judge
    judge = Judge(llm=None, use_llm_judge=False)
    print(f"\nJudge tests:")
    print(f"  'Yes they are' contains 'yes': {judge('Yes they are both American', 'yes')}")
    print(f"  'I dont know' rejected: {judge('I dont have enough information', 'yes')}")
    
    print("\nAll tests passed!")
