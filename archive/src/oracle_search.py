"""
Oracle-Guided Search for High-Quality Trajectory Generation.

Key insight: During training data generation, we can use a stronger LLM
with FULL context to make decisions. The trajectory is then recorded with
compressed observations for the student controller to learn from.

This addresses the 40% failure rate by:
1. Oracle sees full retrieved docs (not 50-token summaries)
2. Oracle can reason about action sequences before committing
3. Higher success rate = more training data = better BC model

The compressed observations are still recorded so the BC model learns
to make decisions from limited information (what it will see at inference).
"""

import json
import logging
import heapq
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class OracleAction(Enum):
    """Actions the oracle can take."""
    GENERATE_LLM = "GENERATE_LLM"
    DECOMPOSE_LLM = "DECOMPOSE_LLM"
    RETRIEVE_KEYWORD = "RETRIEVE_KEYWORD"
    RETRIEVE_DENSE = "RETRIEVE_DENSE"
    REASON_LLM = "REASON_LLM"


@dataclass
class OracleState:
    """State with both full context (for oracle) and compressed (for training)."""
    query: str
    original_query: str
    
    # Full context - what the oracle sees
    full_context: str = ""  # All retrieved docs, full reasoning
    retrieved_docs: List[str] = field(default_factory=list)
    sub_questions: List[str] = field(default_factory=list)
    reasoning_history: List[str] = field(default_factory=list)
    
    # Compressed context - what gets recorded for training
    compressed_observations: List[str] = field(default_factory=list)
    
    # Metadata
    depth: int = 0
    accumulated_cost: float = 0.0
    answer: Optional[str] = None
    is_correct: bool = False


class OracleGuidedSearch:
    """
    Oracle-guided trajectory generation.
    
    The oracle LLM sees full context to make high-quality decisions,
    but we record compressed observations for the BC training data.
    """
    
    def __init__(
        self,
        retriever,
        generator,  # Main LLM for execution
        oracle,     # Oracle LLM for decision-making (can be same or stronger)
        cost_table: Dict[str, float],
        max_depth: int = 8,
        compression_tokens: int = 50,
    ):
        self.retriever = retriever
        self.generator = generator
        self.oracle = oracle
        self.cost_table = cost_table
        self.max_depth = max_depth
        self.compression_tokens = compression_tokens
        
    def generate_trajectory(
        self, 
        query: str, 
        ground_truth: str
    ) -> Optional[Dict]:
        """
        Generate a trajectory using oracle guidance.
        
        Returns trajectory dict with compressed observations for training.
        """
        state = OracleState(
            query=query,
            original_query=query,
        )
        
        trajectory_steps = []
        
        while state.depth < self.max_depth:
            # Oracle sees FULL context to decide
            action = self._oracle_decide(state, ground_truth)
            
            if action is None:
                logger.warning(f"Oracle returned no action at depth {state.depth}")
                break
            
            # Record the training step BEFORE execution
            # (state -> action mapping with compressed observation)
            compressed_state = self._build_compressed_state(state)
            
            # Execute action, update state
            state, observation_full, observation_compressed = self._execute_action(
                state, action
            )
            
            # Record step for training
            trajectory_steps.append({
                "state": compressed_state,
                "action": action.value,
                "observation": observation_compressed,
            })
            
            # Check if we got a correct answer
            if state.answer is not None:
                is_correct = self._check_answer(state.answer, ground_truth)
                if is_correct:
                    logger.info(f"Oracle found correct answer at depth {state.depth}")
                    return {
                        "query": query,
                        "ground_truth": ground_truth,
                        "answer": state.answer,
                        "is_correct": True,
                        "total_cost": state.accumulated_cost,
                        "steps": trajectory_steps,
                        "search_depth": state.depth,
                    }
                else:
                    # Wrong answer, oracle might want to try again
                    state.answer = None  # Reset and continue
                    
        logger.warning(f"Oracle failed to find correct answer for: {query[:50]}...")
        return None
    
    def _oracle_decide(self, state: OracleState, ground_truth: str) -> Optional[OracleAction]:
        """
        Oracle makes decision with FULL context visibility.
        
        The oracle prompt includes:
        - Original query
        - All retrieved documents (full text)
        - All sub-questions and their status
        - Full reasoning history
        
        This gives the oracle maximum information to make good decisions.
        """
        # Build rich context for oracle
        oracle_context = self._build_oracle_context(state)
        
        # Check if answer is already in context - if so, generate!
        context_text = " ".join(state.retrieved_docs).lower() if state.retrieved_docs else ""
        if ground_truth.lower() in context_text:
            logger.info(f"Oracle: Answer '{ground_truth}' found in context, deciding GENERATE_LLM")
            return OracleAction.GENERATE_LLM
        
        prompt = f"""You are an expert RAG system planner. Given the current state, decide the BEST next action.

QUERY: {state.original_query}

CURRENT STATE:
{oracle_context}

TARGET ANSWER (for planning only): {ground_truth}

AVAILABLE ACTIONS:
1. GENERATE_LLM - Generate final answer (USE THIS if the target answer appears in your context!)
2. DECOMPOSE_LLM - Break query into sub-questions (for multi-hop questions)
3. RETRIEVE_KEYWORD - Search for more relevant documents  
4. REASON_LLM - Synthesize/analyze current information

CRITICAL: If you can see "{ground_truth}" in the retrieved documents above, choose GENERATE_LLM immediately!

DECISION CRITERIA:
- If the target answer is visible in context → GENERATE_LLM
- If the query asks about multiple entities → DECOMPOSE_LLM first
- If you need more information → RETRIEVE_KEYWORD
- If you have docs but need to connect them → REASON_LLM

What is the SINGLE best next action? Reply with just the action name."""

        response = self.oracle.generate(prompt, max_tokens=20)
        
        # Parse action from response
        response_upper = response.strip().upper()
        for action in OracleAction:
            if action.value in response_upper:
                return action
        
        # Default to generate if unclear
        logger.warning(f"Oracle gave unclear response: {response}, defaulting to GENERATE_LLM")
        return OracleAction.GENERATE_LLM
    
    def _build_oracle_context(self, state: OracleState) -> str:
        """Build rich context string for oracle decision-making."""
        parts = []
        
        if state.retrieved_docs:
            parts.append("RETRIEVED DOCUMENTS:")
            for i, doc in enumerate(state.retrieved_docs[-3:], 1):  # Last 3 docs
                parts.append(f"  [{i}] {doc[:500]}...")
        
        if state.sub_questions:
            parts.append("\nSUB-QUESTIONS:")
            for sq in state.sub_questions:
                parts.append(f"  - {sq}")
        
        if state.reasoning_history:
            parts.append("\nREASONING SO FAR:")
            for r in state.reasoning_history[-2:]:  # Last 2 reasoning steps
                parts.append(f"  {r[:200]}...")
        
        if state.compressed_observations:
            parts.append(f"\nSTEPS TAKEN: {len(state.compressed_observations)}")
        
        return "\n".join(parts) if parts else "(No context yet - starting fresh)"
    
    def _build_compressed_state(self, state: OracleState) -> str:
        """Build compressed state string for training data."""
        parts = [f"[CLS] {state.original_query} [SEP]"]
        for obs in state.compressed_observations:
            parts.append(f"{obs} [SEP]")
        return " ".join(parts)
    
    def _execute_action(
        self, 
        state: OracleState, 
        action: OracleAction
    ) -> Tuple[OracleState, str, str]:
        """
        Execute action and return updated state + observations.
        
        Returns:
            - Updated state
            - Full observation (for oracle)
            - Compressed observation (for training)
        """
        new_state = OracleState(
            query=state.query,
            original_query=state.original_query,
            full_context=state.full_context,
            retrieved_docs=state.retrieved_docs.copy(),
            sub_questions=state.sub_questions.copy(),
            reasoning_history=state.reasoning_history.copy(),
            compressed_observations=state.compressed_observations.copy(),
            depth=state.depth + 1,
            accumulated_cost=state.accumulated_cost,
        )
        
        if action == OracleAction.GENERATE_LLM:
            return self._execute_generate(new_state)
        elif action == OracleAction.DECOMPOSE_LLM:
            return self._execute_decompose(new_state)
        elif action == OracleAction.RETRIEVE_KEYWORD:
            return self._execute_retrieve(new_state)
        elif action == OracleAction.REASON_LLM:
            return self._execute_reason(new_state)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _execute_generate(self, state: OracleState) -> Tuple[OracleState, str, str]:
        """Generate final answer."""
        # Build context from retrieved docs
        context = "\n\n".join(state.retrieved_docs[-5:]) if state.retrieved_docs else ""
        
        prompt = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {state.original_query}

Answer (be brief and direct):"""

        answer = self.generator.generate(prompt, max_tokens=100)
        answer = answer.strip()
        
        state.answer = answer
        state.accumulated_cost += self.cost_table.get("GENERATE_LLM", 0.015)
        
        # Observations
        obs_full = f"Generated answer: {answer}"
        obs_compressed = self._compress(f"Answer: {answer[:100]}")
        state.compressed_observations.append(obs_compressed)
        
        return state, obs_full, obs_compressed
    
    def _execute_decompose(self, state: OracleState) -> Tuple[OracleState, str, str]:
        """Decompose query into sub-questions."""
        prompt = f"""Break this question into 2-3 simpler fact-finding sub-questions that can be answered independently.

Original Question: {state.original_query}

Think about what entities or facts need to be looked up separately.

Sub-questions (one per line, each should be a simple factual question):"""

        response = self.generator.generate(prompt, max_tokens=150)
        
        # Parse sub-questions
        sub_qs = []
        for line in response.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if line and ("?" in line or len(line) > 10):
                sub_qs.append(line)
        sub_qs = sub_qs[:3]
        
        if sub_qs:
            state.sub_questions.extend(sub_qs)
            # Focus on first sub-question for next retrieval
            state.query = sub_qs[0]
        
        state.accumulated_cost += self.cost_table.get("DECOMPOSE_LLM", 0.028)
        
        # Observations
        obs_full = f"Decomposed into: {sub_qs}"
        obs_compressed = self._compress(f"Split into {len(sub_qs)} sub-questions: {', '.join(sq[:30] for sq in sub_qs)}")
        state.compressed_observations.append(obs_compressed)
        
        return state, obs_full, obs_compressed
    
    def _execute_retrieve(self, state: OracleState) -> Tuple[OracleState, str, str]:
        """Retrieve documents."""
        docs = self.retriever.retrieve(state.query, k=5)
        
        # Store full docs for oracle
        doc_texts = [d.get("text", d.get("content", str(d))) for d in docs]
        state.retrieved_docs.extend(doc_texts)
        
        state.accumulated_cost += self.cost_table.get("RETRIEVE_KEYWORD", 0.009)
        
        # Observations
        obs_full = f"Retrieved {len(docs)} docs: {doc_texts}"
        
        # Compressed observation for training
        titles = [d.get("title", "doc")[:20] for d in docs[:3]]
        obs_compressed = self._compress(f"Found {len(docs)} docs on: {', '.join(titles)}")
        state.compressed_observations.append(obs_compressed)
        
        return state, obs_full, obs_compressed
    
    def _execute_reason(self, state: OracleState) -> Tuple[OracleState, str, str]:
        """Synthesize/reason about current context."""
        context = "\n".join(state.retrieved_docs[-3:]) if state.retrieved_docs else "No context yet."
        
        prompt = f"""Based on this context, what can we conclude about the question?

Context:
{context}

Question: {state.original_query}

Analysis (be concise):"""

        reasoning = self.generator.generate(prompt, max_tokens=150)
        state.reasoning_history.append(reasoning.strip())
        
        state.accumulated_cost += self.cost_table.get("REASON_LLM", 0.027)
        
        # Observations
        obs_full = f"Reasoning: {reasoning}"
        obs_compressed = self._compress(f"Analysis: {reasoning[:100]}")
        state.compressed_observations.append(obs_compressed)
        
        return state, obs_full, obs_compressed
    
    def _compress(self, text: str, max_tokens: int = None) -> str:
        """Compress text to fit token limit."""
        max_tokens = max_tokens or self.compression_tokens
        # Rough approximation: 1 token ≈ 4 chars
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars-3] + "..."
        return text
    
    def _check_answer(self, answer: str, ground_truth: str) -> bool:
        """Check if answer matches ground truth."""
        # Normalize both
        answer_norm = answer.lower().strip()
        truth_norm = ground_truth.lower().strip()
        
        # Exact match
        if truth_norm in answer_norm:
            return True
        
        # Check if key parts match (for multi-word answers)
        truth_words = set(truth_norm.split())
        answer_words = set(answer_norm.split())
        
        # If ground truth is short, require it to appear
        if len(truth_words) <= 3:
            return truth_norm in answer_norm
        
        # For longer answers, check word overlap
        overlap = len(truth_words & answer_words) / len(truth_words)
        return overlap >= 0.7


class LLMOracleWrapper:
    """
    Wrapper to use any LLM as oracle.
    Can use same model as generator, or a stronger model.
    """
    
    def __init__(self, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate response from oracle LLM."""
        import requests
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,  # Lower temp for more deterministic decisions
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            logger.error(f"Oracle LLM error: {response.status_code}")
            return ""


def generate_trajectories_with_oracle(
    queries: List[Dict],
    retriever,
    generator,
    oracle,
    cost_table: Dict[str, float],
    output_path: str,
    max_trajectories: int = 500,
):
    """
    Generate trajectories using oracle-guided search.
    
    Args:
        queries: List of {"question": str, "answer": str} dicts
        retriever: Document retriever
        generator: LLM for execution
        oracle: LLM for decision-making
        cost_table: Action costs
        output_path: Where to save trajectories
        max_trajectories: Maximum number to generate
    """
    search = OracleGuidedSearch(
        retriever=retriever,
        generator=generator,
        oracle=oracle,
        cost_table=cost_table,
    )
    
    trajectories = []
    success_count = 0
    
    for i, q in enumerate(queries[:max_trajectories]):
        query = q["question"]
        ground_truth = q["answer"]
        
        logger.info(f"[{i+1}/{min(len(queries), max_trajectories)}] Processing: {query[:50]}...")
        
        trajectory = search.generate_trajectory(query, ground_truth)
        
        if trajectory:
            trajectories.append(trajectory)
            success_count += 1
            logger.info(f"  ✓ Success ({success_count} total)")
        else:
            logger.info(f"  ✗ Failed")
        
        # Save periodically
        if (i + 1) % 50 == 0:
            _save_trajectories(trajectories, output_path, success_count, i + 1)
    
    # Final save
    _save_trajectories(trajectories, output_path, success_count, len(queries))
    
    return trajectories


def _save_trajectories(trajectories, output_path, success_count, total_attempted):
    """Save trajectories to file."""
    output = {
        "metadata": {
            "method": "oracle_guided_search",
            "success_count": success_count,
            "total_attempted": total_attempted,
            "success_rate": success_count / total_attempted if total_attempted > 0 else 0,
        },
        "trajectories": trajectories,
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Saved {len(trajectories)} trajectories to {output_path}")


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Oracle-guided search module loaded.")
    print("Usage:")
    print("  from src.oracle_search import OracleGuidedSearch, LLMOracleWrapper")
    print("  oracle = LLMOracleWrapper('llama3:8b')")
    print("  search = OracleGuidedSearch(retriever, generator, oracle, cost_table)")
    print("  trajectory = search.generate_trajectory(query, ground_truth)")
