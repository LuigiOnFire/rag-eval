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
from typing import List, Dict, Tuple, Optional, Any
from enum import IntEnum

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """7-class action space for the Manager-Worker agent."""
    GENERATE_END_SLM = 0
    GENERATE_END_LLM = 1
    DECOMPOSE_SLM = 2
    DECOMPOSE_LLM = 3
    RETRIEVE_KEYWORD = 4
    RETRIEVE_DENSE = 5
    REASON_LLM = 6


# Terminal actions that produce a final answer
TERMINAL_ACTIONS = {Action.GENERATE_END_SLM, Action.GENERATE_END_LLM}

# Default prompts for worker compression
COMPRESSION_PROMPT = """Summarize the following in under 50 tokens, focusing on what was found and what's missing:

{content}

Summary:"""

DECOMPOSE_PROMPT = """Break down this question into 2-3 simpler sub-questions that would help answer it:

Question: {question}

Sub-questions:"""

REASON_PROMPT = """Given the following context and question, provide intermediate reasoning and identify what additional information is needed:

Context: {context}

Question: {question}

Reasoning:"""

GENERATE_PROMPT = """Answer the following question based on the context provided. Be concise and factual.

Context: {context}

Question: {question}

Answer:"""


@dataclass
class Step:
    """A single step in a trajectory."""
    state: str  # The compressed state string fed to controller
    action: Action  # The action taken
    observation: str  # The compressed observation from worker (<50 tokens)
    result: Optional[str] = None  # Raw result (e.g., full answer for terminal actions)
    energy_wh: float = 0.0  # Energy consumed by this action


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
                "label": int(step.action),
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
                    "action": int(s.action),
                    "action_name": Action(s.action).name,
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
        Search for the cheapest correct trajectory.
        
        Uses a greedy cost-ordered strategy:
        1. Start with retrieval (usually cheapest)
        2. Try generate actions in cost order
        3. If generation fails, try decomposition/reasoning
        4. Repeat until correct answer or max steps
        
        Args:
            query: The question to answer
            ground_truth: The expected answer
            
        Returns:
            Trajectory with steps, correctness flag, and total energy
        """
        # Reset search state
        self._current_context = []
        self._sub_questions = []
        self._reasoning_notes = ""
        
        trajectory = Trajectory(query=query, ground_truth=ground_truth)
        state = f"[CLS] {query} [SEP]"
        
        # Track which actions we've tried at each context level
        tried_actions = set()  # All actions tried
        tried_terminals_this_context = set()  # Terminal actions tried with current context
        
        for step_num in range(self.max_steps):
            # Select next action using cost-ordered strategy
            action = self._select_action(state, tried_actions, tried_terminals_this_context)
            
            if action is None:
                logger.warning(f"No more actions to try for query: {query[:50]}...")
                break
            
            # Execute action and get compressed observation
            result, observation = self._execute_action(action, query, state)
            
            # Record energy cost
            energy = self.cost_table.get(int(action), 0.0)
            
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
            
            # Update state with compressed observation
            state = f"{state} {observation} [SEP]"
            
            # Check if terminal action
            if action in TERMINAL_ACTIONS:
                tried_terminals_this_context.add(action)
                if self.judge(result, ground_truth):
                    trajectory.correct = True
                    logger.info(f"Found correct answer with {action.name} "
                               f"(energy: {trajectory.total_energy_wh:.6f} Wh)")
                    return trajectory
                else:
                    # Terminal action failed, will try more expensive terminal or non-terminal
                    logger.debug(f"{action.name} produced incorrect answer, continuing search")
            else:
                # Non-terminal action completed - reset terminal tracking since context changed
                tried_actions.add(action)
                tried_terminals_this_context.clear()  # Can retry terminals with new context
        
        # Max steps reached or no correct answer found
        logger.info(f"Search exhausted for query: {query[:50]}... "
                   f"(correct={trajectory.correct}, energy={trajectory.total_energy_wh:.6f} Wh)")
        return trajectory
    
    def _select_action(self, state: str, tried_actions: set, tried_terminals: set) -> Optional[Action]:
        """
        Select the next action using cost-ordered strategy.
        
        Strategy:
        1. If no context yet, retrieve first (cheapest info gain)
        2. Try terminal actions in cost order (that haven't been tried with current context)
        3. If all terminals fail, try non-terminal actions to gather more info
        4. After non-terminal, terminal actions can be retried
        
        Args:
            state: Current state string
            tried_actions: Set of non-terminal actions already tried
            tried_terminals: Set of terminal actions tried with current context
            
        Returns:
            Next action to try, or None if exhausted
        """
        has_context = len(self._current_context) > 0
        
        # Phase 1: If no context, always retrieve first
        if not has_context:
            if Action.RETRIEVE_KEYWORD not in tried_actions:
                return Action.RETRIEVE_KEYWORD
            if Action.RETRIEVE_DENSE not in tried_actions and self.dense_retriever is not None:
                return Action.RETRIEVE_DENSE
        
        # Phase 2: Try terminal actions in cost order (that we haven't tried with current context)
        for action in self._cost_ordered_actions:
            if action in TERMINAL_ACTIONS and action not in tried_terminals:
                return action
        
        # Phase 3: All terminals tried and failed - try non-terminal to gather more context
        for action in self._cost_ordered_actions:
            if action not in TERMINAL_ACTIONS and action not in tried_actions:
                return action
        
        return None
    
    def _execute_action(
        self, 
        action: Action, 
        query: str, 
        state: str
    ) -> Tuple[str, str]:
        """
        Execute an action and return (result, compressed_observation).
        
        The worker executes the action, then generates a <50 token summary
        that captures the semantic gist for the controller.
        
        Args:
            action: Action to execute
            query: Original query
            state: Current state (for context)
            
        Returns:
            Tuple of (raw_result, compressed_observation)
        """
        if action == Action.RETRIEVE_KEYWORD:
            return self._execute_retrieve_keyword(query)
        
        elif action == Action.RETRIEVE_DENSE:
            return self._execute_retrieve_dense(query)
        
        elif action == Action.GENERATE_END_SLM:
            return self._execute_generate(query, self.slm, "SLM")
        
        elif action == Action.GENERATE_END_LLM:
            return self._execute_generate(query, self.llm, "LLM")
        
        elif action == Action.DECOMPOSE_SLM:
            return self._execute_decompose(query, self.slm, "SLM")
        
        elif action == Action.DECOMPOSE_LLM:
            return self._execute_decompose(query, self.llm, "LLM")
        
        elif action == Action.REASON_LLM:
            return self._execute_reason(query)
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _execute_retrieve_keyword(self, query: str) -> Tuple[str, str]:
        """Execute BM25 keyword retrieval."""
        try:
            passages, scores = self.retriever.retrieve(query, top_k=5)
            self._current_context.extend(passages)
            
            # Compress retrieval results
            if passages:
                titles = [p.get('title', 'Unknown')[:30] for p in passages[:3]]
                observation = f"Found {len(passages)} docs: {', '.join(titles)}. "
                if scores and scores[0] < 5.0:
                    observation += "Low relevance scores."
                observation = self._truncate_observation(observation)
            else:
                observation = "No relevant documents found."
            
            return json.dumps([p.get('text', '')[:200] for p in passages[:3]]), observation
            
        except Exception as e:
            logger.error(f"Retrieve keyword failed: {e}")
            return "", f"Retrieval failed: {str(e)[:30]}"
    
    def _execute_retrieve_dense(self, query: str) -> Tuple[str, str]:
        """Execute dense vector retrieval."""
        retriever = self.dense_retriever or self.retriever
        
        try:
            # Check if retriever supports dense retrieval
            if hasattr(retriever, 'retrieve_dense'):
                passages, scores = retriever.retrieve_dense(query, top_k=5)
            else:
                passages, scores = retriever.retrieve(query, top_k=5)
            
            self._current_context.extend(passages)
            
            # Compress retrieval results
            if passages:
                titles = [p.get('title', 'Unknown')[:30] for p in passages[:3]]
                observation = f"Dense search found {len(passages)} docs: {', '.join(titles)}."
                observation = self._truncate_observation(observation)
            else:
                observation = "Dense search found no relevant documents."
            
            return json.dumps([p.get('text', '')[:200] for p in passages[:3]]), observation
            
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
            
            # Generate answer
            result = model.generate(query, context_passages)
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
        """Decompose query into sub-questions."""
        try:
            prompt = DECOMPOSE_PROMPT.format(question=query)
            result = model.generate(prompt, [])  # No context needed for decomposition
            
            sub_qs = result.get('answer', '') if isinstance(result, dict) else str(result)
            self._sub_questions.append(sub_qs)
            
            # Compress
            observation = f"Decomposed ({model_name}): {sub_qs[:80]}"
            observation = self._truncate_observation(observation)
            
            return sub_qs, observation
            
        except Exception as e:
            logger.error(f"Decompose ({model_name}) failed: {e}")
            return "", f"Decomposition failed: {str(e)[:20]}"
    
    def _execute_reason(self, query: str) -> Tuple[str, str]:
        """Execute intermediate reasoning with LLM."""
        try:
            # Build context string
            context = self._format_context_for_reasoning()
            prompt = REASON_PROMPT.format(context=context, question=query)
            
            result = self.llm.generate(prompt, [])
            reasoning = result.get('answer', '') if isinstance(result, dict) else str(result)
            self._reasoning_notes += f" {reasoning}"
            
            # Compress
            observation = f"Reasoning: {reasoning[:80]}"
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
                step = Step(
                    state=s_dict["state"],
                    action=Action(s_dict["action"]),
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
