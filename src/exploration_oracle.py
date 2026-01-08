"""
Exploration-based Oracle for generating training data.

Instead of using heuristics to "decide" what to do, this Oracle
explores all possible execution paths and selects the cheapest
one that produces the correct answer.

This follows the "Green-DeepRAG" approach where we let different
strategies compete and pick the winner based on cost-effectiveness.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

@dataclass
class OracleResult:
    """Result from trying an execution path."""
    action: str
    cost: float
    answer: str
    trace: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None

class ExplorationOracle:
    """
    Oracle that explores all possible execution paths to find
    the cheapest correct solution for training data generation.
    """
    
    def __init__(self, agent, judge, cost_calculator):
        self.agent = agent
        self.judge = judge
        self.cost_calculator = cost_calculator
        
    def find_gold_trajectory(self, query: str, ground_truth: str) -> Optional[Dict[str, Any]]:
        """
        Explores all 7 execution tiers and returns the cheapest correct one.
        
        NEW EXPANDED TIERS (fills the complexity gap):
        Tier 0: Pure SLM (cost ~$1)
        Tier 1: Fast BM25 (cost ~$5) 
        Tier 2: Smart Dense (cost ~$8)
        Tier 3: Refine Query -> Hybrid (cost ~$12) - "Decompose Lite"
        Tier 4: High-Recall -> Filter -> Generate (cost ~$18) - "Sort Carefully" 
        Tier 5: 2-Step Iterative Chain (cost ~$25) - Simple bridges
        Tier 6: Full Decomposition Tree (cost ~$50+) - Complex logic
        
        Returns the gold trajectory for training, or None if all fail.
        """
        
        candidates = []
        
        # === PHASE 1: CHEAP & FAST ===
        
        # Tier 0: The "Lazy Student" - Pure parametric knowledge
        print("🔍 Tier 0: Pure SLM (no retrieval, cost ~$1)")
        result = self._try_pure_slm(query, ground_truth)
        if result.success:
            candidates.append(result)
            print(f"✅ Tier 0 SUCCESS: {result.answer[:100]}")
        else:
            print(f"❌ Tier 0 failed: {result.error}")
            
        # Tier 1: The "Librarian" - Fast keyword retrieval  
        print("🔍 Tier 1: Fast Retrieval (BM25, cost ~$5)")
        result = self._try_fast_retrieval(query, ground_truth)
        if result.success:
            candidates.append(result)
            print(f"✅ Tier 1 SUCCESS: {result.answer[:100]}")
        else:
            print(f"❌ Tier 1 failed: {result.error}")
            
        # Tier 2: The "Scholar" - Smart semantic retrieval
        print("🔍 Tier 2: Smart Retrieval (Dense, cost ~$8)")
        result = self._try_smart_retrieval(query, ground_truth)
        if result.success:
            candidates.append(result)
            print(f"✅ Tier 2 SUCCESS: {result.answer[:100]}")
        else:
            print(f"❌ Tier 2 failed: {result.error}")

        # === PHASE 2: "DECOMPOSE LITE" & SORTING (The New Skew) ===
        
        # Tier 3: The "Refiner" - Query rewriting + hybrid retrieval
        print("🔍 Tier 3: Refine Query (Rewrite -> Hybrid, cost ~$12)")
        result = self._try_refine_query(query, ground_truth)
        if result.success:
            candidates.append(result)
            print(f"✅ Tier 3 SUCCESS: {result.answer[:100]}")
        else:
            print(f"❌ Tier 3 failed: {result.error}")
            
        # Tier 4: The "Curator" - High recall + careful sorting
        print("🔍 Tier 4: Filter Results (High-Recall -> Sort, cost ~$18)")
        result = self._try_filter_results(query, ground_truth)
        if result.success:
            candidates.append(result)
            print(f"✅ Tier 4 SUCCESS: {result.answer[:100]}")
        else:
            print(f"❌ Tier 4 failed: {result.error}")

        # === PHASE 3: HEAVY REASONING ===

        # Tier 5: The "Bridge Builder" - 2-step iterative reasoning
        print("🔍 Tier 5: Iterative Chain (2-Step Bridge, cost ~$25)")
        result = self._try_iterative_chain(query, ground_truth)
        if result.success:
            candidates.append(result)
            print(f"✅ Tier 5 SUCCESS: {result.answer[:100]}")
        else:
            print(f"❌ Tier 5 failed: {result.error}")
            
        # Tier 6: The "Detective" - Full decomposition tree
        print("🔍 Tier 6: Full Decomposition (Complex Logic, cost ~$50+)")
        result = self._try_full_decomposition(query, ground_truth)
        if result.success:
            candidates.append(result)
            print(f"✅ Tier 6 SUCCESS: {result.answer[:100]}")
        else:
            print(f"❌ Tier 6 failed: {result.error}")
            
        if not candidates:
            print("💀 All 7 tiers failed - no training data generated")
            return None
            
        # Sort by cost and pick the cheapest winner
        candidates.sort(key=lambda x: x.cost)
        winner = candidates[0]
        
        print(f"🏆 WINNER: {winner.action} (cost: {winner.cost:.3f})")
        
        # Convert to training format
        return {
            "question": query,
            "ground_truth": ground_truth, 
            "gold_action": winner.action,
            "gold_cost": winner.cost,
            "gold_answer": winner.answer,
            "gold_trace": winner.trace,
            "alternatives": [
                {
                    "action": c.action,
                    "cost": c.cost,
                    "success": c.success
                } for c in candidates
            ]
        }
        
    def _try_pure_slm(self, query: str, ground_truth: str) -> OracleResult:
        """Path 1: Try to answer with just the SLM, no retrieval."""
        try:
            start_time = time.time()
            
            # Force the agent to generate without any retrieval
            answer = self.agent.llm.generate(f"""Question: {query}

Answer (be direct and concise, use only your knowledge without retrieval):""")
            
            elapsed = time.time() - start_time
            cost = self.cost_calculator.calculate_generation_cost(
                input_tokens=len(query.split()) * 1.3,  # Rough estimate
                output_tokens=len(answer.split()) * 1.3,
                model_type="slm"
            )
            
            success = self.judge.evaluate(answer, ground_truth)
            
            trace = [{
                "action": "GENERATE_DIRECT",
                "input": query,
                "output": answer,
                "cost": cost,
                "time": elapsed
            }]
            
            return OracleResult(
                action="<ASSIGN_SLM>",
                cost=cost,
                answer=answer,
                trace=trace,
                success=success,
                error=None if success else f"Answer mismatch: got '{answer}', expected '{ground_truth}'"
            )
            
        except Exception as e:
            return OracleResult(
                action="<ASSIGN_SLM>",
                cost=float('inf'),
                answer="",
                trace=[],
                success=False,
                error=f"Exception: {str(e)}"
            )
            
    def _try_fast_retrieval(self, query: str, ground_truth: str) -> OracleResult:
        """Path 2: Try BM25 retrieval + generation."""
        try:
            start_time = time.time()
            trace = []
            
            # Force BM25 retrieval using the agent's tiered system
            docs = self.agent._retrieve_tier(query, "FAST")
            cost = 0.01  # FAST tier cost from your system
            
            trace.append({
                "action": "RETRIEVE_FAST", 
                "input": query,
                "output": f"Retrieved {len(docs)} docs",
                "cost": cost
            })
            
            # Generate with retrieved context
            context = "\n\n".join([d.get('text', '') for d in docs])
            answer = self.agent.llm.generate(f"""Question: {query}

Context:
{context}

Answer (be direct and concise):""")
            
            gen_cost = self.cost_calculator.calculate_generation_cost(
                input_tokens=len((query + context).split()) * 1.3,
                output_tokens=len(answer.split()) * 1.3,
                model_type="slm"
            )
            
            trace.append({
                "action": "GENERATE_FINAL",
                "input": query,
                "output": answer, 
                "cost": gen_cost
            })
            
            elapsed = time.time() - start_time
            total_cost = cost + gen_cost
            success = self.judge.evaluate(answer, ground_truth)
            
            return OracleResult(
                action="<RETRIEVE_FAST>",
                cost=total_cost,
                answer=answer,
                trace=trace,
                success=success,
                error=None if success else f"Answer mismatch: got '{answer}', expected '{ground_truth}'"
            )
            
        except Exception as e:
            return OracleResult(
                action="<RETRIEVE_FAST>",
                cost=float('inf'),
                answer="",
                trace=[],
                success=False,
                error=f"Exception: {str(e)}"
            )
            
    def _try_smart_retrieval(self, query: str, ground_truth: str) -> OracleResult:
        """Path 3: Try dense/embedding retrieval + generation."""
        try:
            start_time = time.time()
            trace = []
            
            # Force dense retrieval using the agent's tiered system
            docs = self.agent._retrieve_tier(query, "SMART")
            cost = 0.05  # SMART tier cost from your system
            
            trace.append({
                "action": "RETRIEVE_SMART",
                "input": query, 
                "output": f"Retrieved {len(docs)} docs",
                "cost": cost
            })
            
            # Generate with retrieved context
            context = "\n\n".join([d.get('text', '') for d in docs])
            answer = self.agent.llm.generate(f"""Question: {query}

Context:
{context}

Answer (be direct and concise):""")
            
            gen_cost = self.cost_calculator.calculate_generation_cost(
                input_tokens=len((query + context).split()) * 1.3,
                output_tokens=len(answer.split()) * 1.3,
                model_type="slm"
            )
            
            trace.append({
                "action": "GENERATE_FINAL",
                "input": query,
                "output": answer,
                "cost": gen_cost
            })
            
            elapsed = time.time() - start_time
            total_cost = cost + gen_cost
            success = self.judge.evaluate(answer, ground_truth)
            
            return OracleResult(
                action="<RETRIEVE_SMART>",
                cost=total_cost,
                answer=answer,
                trace=trace,
                success=success,
                error=None if success else f"Answer mismatch: got '{answer}', expected '{ground_truth}'"
            )
            
        except Exception as e:
            return OracleResult(
                action="<RETRIEVE_SMART>",
                cost=float('inf'),
                answer="",
                trace=[],
                success=False,
                error=f"Exception: {str(e)}"
            )
    
    def _try_refine_query(self, query: str, ground_truth: str) -> OracleResult:
        """Tier 3: Rewrite query for better retrieval, then use hybrid search."""
        try:
            start_time = time.time()
            trace = []
            
            # Step 1: Rewrite query to be more searchable
            rewrite_prompt = f"""Rewrite this question to be optimal for document search. 
Expand abbreviated terms, add context, make entities explicit.

Original: {query}

Rewritten (more searchable):"""
            
            refined_query = self.agent.llm.generate(rewrite_prompt)
            
            rewrite_cost = self.cost_calculator.calculate_generation_cost(
                input_tokens=len(rewrite_prompt.split()) * 1.3,
                output_tokens=len(refined_query.split()) * 1.3,
                model_type="slm"
            )
            
            trace.append({
                "action": "REFINE_QUERY",
                "input": query,
                "output": refined_query,
                "cost": rewrite_cost
            })
            
            # Step 2: Use both BM25 and Dense retrieval on refined query
            # Simulate hybrid retrieval by combining both methods
            fast_docs = self.agent._retrieve_tier(refined_query, "FAST")
            smart_docs = self.agent._retrieve_tier(refined_query, "SMART") 
            
            # Combine and deduplicate
            all_docs = fast_docs + smart_docs
            seen_titles = set()
            hybrid_docs = []
            for doc in all_docs:
                title = doc.get('title', doc.get('text', '')[:50])
                if title not in seen_titles:
                    hybrid_docs.append(doc)
                    seen_titles.add(title)
            
            # Take top 5 combined
            hybrid_docs = hybrid_docs[:5]
            hybrid_cost = 0.01 + 0.05  # FAST + SMART costs
            
            trace.append({
                "action": "RETRIEVE_HYBRID",
                "input": refined_query,
                "output": f"Retrieved {len(hybrid_docs)} hybrid docs",
                "cost": hybrid_cost
            })
            
            # Step 3: Generate answer
            context = "\n\n".join([d.get('text', '') for d in hybrid_docs])
            answer = self.agent.llm.generate(f"""Question: {query}

Context:
{context}

Answer (be direct and concise):""")
            
            gen_cost = self.cost_calculator.calculate_generation_cost(
                input_tokens=len((query + context).split()) * 1.3,
                output_tokens=len(answer.split()) * 1.3,
                model_type="slm"
            )
            
            trace.append({
                "action": "GENERATE_FINAL",
                "input": query,
                "output": answer,
                "cost": gen_cost
            })
            
            elapsed = time.time() - start_time
            total_cost = rewrite_cost + hybrid_cost + gen_cost
            success = self.judge.evaluate(answer, ground_truth)
            
            return OracleResult(
                action="<REFINE_QUERY>",
                cost=total_cost,
                answer=answer,
                trace=trace,
                success=success,
                error=None if success else f"Answer mismatch: got '{answer}', expected '{ground_truth}'"
            )
            
        except Exception as e:
            return OracleResult(
                action="<REFINE_QUERY>",
                cost=float('inf'),
                answer="",
                trace=[],
                success=False,
                error=f"Exception: {str(e)}"
            )
            
    def _try_full_decomposition(self, query: str, ground_truth: str) -> OracleResult:
        """Path 4: Try full decomposition loop with all the bells and whistles."""
        try:
            start_time = time.time()
            
            # Force the agent to run its full decomposition loop
            # This should include your variable substitution logic
            result = self.agent.run_full_trajectory(
                query=query,
                force_decompose=True  # Override any heuristics
            )
            
            elapsed = time.time() - start_time
            
            # Calculate cost from steps
            total_cost = 0.0
            for step in result.steps:
                if step.action in ["RETRIEVE_FAST", "RETRIEVE_SMART", "RETRIEVE_DEEP"]:
                    if "FAST" in step.action:
                        total_cost += 0.01
                    elif "SMART" in step.action:
                        total_cost += 0.05
                    elif "DEEP" in step.action:
                        total_cost += 0.15
                elif step.action in ["GENERATE_FINAL", "EXTRACT", "DECOMPOSE"]:
                    # Estimate generation cost
                    total_cost += self.cost_calculator.calculate_generation_cost(
                        input_tokens=len(step.input.split()) * 1.3,
                        output_tokens=len(step.output.split()) * 1.3,
                        model_type="slm"
                    )
            
            success = self.judge.evaluate(result.final_answer, ground_truth)
            
            # Convert steps to trace format
            trace = []
            for step in result.steps:
                trace.append({
                    "action": step.action,
                    "input": step.input,
                    "output": step.output,
                    "cost": 0.01 if "RETRIEVE" in step.action else 0.005  # Rough estimates
                })
            
            return OracleResult(
                action="<DECOMPOSE>",
                cost=total_cost,
                answer=result.final_answer,
                trace=trace,
                success=success,
                error=None if success else f"Answer mismatch: got '{result.final_answer}', expected '{ground_truth}'"
            )
            
        except Exception as e:
            return OracleResult(
                action="<DECOMPOSE>",
                cost=float('inf'),
                answer="",
                trace=[],
                success=False,
                error=f"Exception: {str(e)}"
            )
    
    def _try_filter_results(self, query: str, ground_truth: str) -> OracleResult:
        """Tier 4: High-recall retrieval + careful filtering."""
        try:
            start_time = time.time()
            trace = []
            
            # Step 1: Retrieve many documents (high recall)
            fast_docs = self.agent._retrieve_tier(query, "FAST")[:10]
            smart_docs = self.agent._retrieve_tier(query, "SMART")[:10] 
            
            # Combine all docs
            all_docs = fast_docs + smart_docs
            seen_titles = set()
            candidate_docs = []
            for doc in all_docs:
                title = doc.get('title', doc.get('text', '')[:50])
                if title not in seen_titles:
                    candidate_docs.append(doc)
                    seen_titles.add(title)
            
            retrieval_cost = 0.01 + 0.05  # FAST + SMART
            
            trace.append({
                "action": "RETRIEVE_HIGH_RECALL",
                "input": query,
                "output": f"Retrieved {len(candidate_docs)} candidate docs",
                "cost": retrieval_cost
            })
            
            # Step 2: Use SLM to filter/rank documents
            doc_summaries = []
            for i, doc in enumerate(candidate_docs):
                title = doc.get('title', 'Unknown')
                snippet = doc.get('text', '')[:200]
                doc_summaries.append(f"{i+1}. {title}: {snippet}...")
            
            filter_prompt = f"""Given this question: {query}

Here are candidate documents:
{chr(10).join(doc_summaries)}

Select the 3 most relevant document numbers (e.g., "2, 5, 8"):"""
            
            filter_response = self.agent.llm.generate(filter_prompt)
            
            filter_cost = self.cost_calculator.calculate_generation_cost(
                input_tokens=len(filter_prompt.split()) * 1.3,
                output_tokens=len(filter_response.split()) * 1.3,
                model_type="slm"
            )
            
            trace.append({
                "action": "FILTER_DOCS",
                "input": f"Filter from {len(candidate_docs)} docs",
                "output": filter_response,
                "cost": filter_cost
            })
            
            # Step 3: Extract selected doc numbers and get those documents
            import re
            numbers = re.findall(r'\d+', filter_response)
            selected_docs = []
            for num_str in numbers[:3]:  # Top 3
                try:
                    idx = int(num_str) - 1  # Convert to 0-based index
                    if 0 <= idx < len(candidate_docs):
                        selected_docs.append(candidate_docs[idx])
                except (ValueError, IndexError):
                    continue
            
            # Fallback: if filtering failed, take first 3
            if not selected_docs:
                selected_docs = candidate_docs[:3]
            
            # Step 4: Generate answer with filtered documents
            context = "\n\n".join([d.get('text', '') for d in selected_docs])
            answer = self.agent.llm.generate(f"""Question: {query}

Context:
{context}

Answer (be direct and concise):""")
            
            gen_cost = self.cost_calculator.calculate_generation_cost(
                input_tokens=len((query + context).split()) * 1.3,
                output_tokens=len(answer.split()) * 1.3,
                model_type="slm"
            )
            
            trace.append({
                "action": "GENERATE_FINAL",
                "input": query,
                "output": answer,
                "cost": gen_cost
            })
            
            elapsed = time.time() - start_time
            total_cost = retrieval_cost + filter_cost + gen_cost
            success = self.judge.evaluate(answer, ground_truth)
            
            return OracleResult(
                action="<RETRIEVE_FILTER>",
                cost=total_cost,
                answer=answer,
                trace=trace,
                success=success,
                error=None if success else f"Answer mismatch: got '{answer}', expected '{ground_truth}'"
            )
            
        except Exception as e:
            return OracleResult(
                action="<RETRIEVE_FILTER>",
                cost=float('inf'),
                answer="",
                trace=[],
                success=False,
                error=f"Exception: {str(e)}"
            )
    
    def _try_iterative_chain(self, query: str, ground_truth: str) -> OracleResult:
        """Tier 5: 2-step iterative reasoning for simple bridge questions."""
        try:
            start_time = time.time()
            trace = []
            
            # Step 1: Generate an intermediate "thought" or sub-question
            decompose_prompt = f"""Break this complex question into a simple 2-step chain.
Step 1 should find an intermediate fact, Step 2 should use that fact to get the final answer.

Question: {query}

Step 1 (find intermediate fact):"""
            
            step1_query = self.agent.llm.generate(decompose_prompt)
            
            step1_cost = self.cost_calculator.calculate_generation_cost(
                input_tokens=len(decompose_prompt.split()) * 1.3,
                output_tokens=len(step1_query.split()) * 1.3,
                model_type="slm"
            )
            
            trace.append({
                "action": "DECOMPOSE_SIMPLE",
                "input": query,
                "output": step1_query,
                "cost": step1_cost
            })
            
            # Step 2: Retrieve for the first sub-question
            docs1 = self.agent._retrieve_tier(step1_query, "SMART")
            ret1_cost = 0.05
            
            trace.append({
                "action": "RETRIEVE_STEP1",
                "input": step1_query,
                "output": f"Retrieved {len(docs1)} docs for step 1",
                "cost": ret1_cost
            })
            
            # Step 3: Get intermediate answer
            context1 = "\n\n".join([d.get('text', '') for d in docs1])
            intermediate_answer = self.agent.llm.generate(f"""Question: {step1_query}

Context:
{context1}

Answer (be direct and concise):""")
            
            int_cost = self.cost_calculator.calculate_generation_cost(
                input_tokens=len((step1_query + context1).split()) * 1.3,
                output_tokens=len(intermediate_answer.split()) * 1.3,
                model_type="slm"
            )
            
            trace.append({
                "action": "EXTRACT_INTERMEDIATE",
                "input": step1_query,
                "output": intermediate_answer,
                "cost": int_cost
            })
            
            # Step 4: Create step 2 query using intermediate answer
            step2_prompt = f"""Original question: {query}
Intermediate fact: {intermediate_answer}

Step 2 (use the intermediate fact to answer the original question):"""
            
            step2_query = self.agent.llm.generate(step2_prompt)
            
            step2_cost = self.cost_calculator.calculate_generation_cost(
                input_tokens=len(step2_prompt.split()) * 1.3,
                output_tokens=len(step2_query.split()) * 1.3,
                model_type="slm"
            )
            
            trace.append({
                "action": "GENERATE_STEP2",
                "input": f"Use: {intermediate_answer}",
                "output": step2_query,
                "cost": step2_cost
            })
            
            # Step 5: Retrieve for step 2
            docs2 = self.agent._retrieve_tier(step2_query, "SMART")
            ret2_cost = 0.05
            
            trace.append({
                "action": "RETRIEVE_STEP2", 
                "input": step2_query,
                "output": f"Retrieved {len(docs2)} docs for step 2",
                "cost": ret2_cost
            })
            
            # Step 6: Generate final answer
            context2 = "\n\n".join([d.get('text', '') for d in docs2])
            final_answer = self.agent.llm.generate(f"""Original question: {query}
Intermediate fact: {intermediate_answer}

Context:
{context2}

Final answer (be direct and concise):""")
            
            final_cost = self.cost_calculator.calculate_generation_cost(
                input_tokens=len((query + intermediate_answer + context2).split()) * 1.3,
                output_tokens=len(final_answer.split()) * 1.3,
                model_type="slm"
            )
            
            trace.append({
                "action": "GENERATE_FINAL",
                "input": query,
                "output": final_answer,
                "cost": final_cost
            })
            
            elapsed = time.time() - start_time
            total_cost = step1_cost + ret1_cost + int_cost + step2_cost + ret2_cost + final_cost
            success = self.judge.evaluate(final_answer, ground_truth)
            
            return OracleResult(
                action="<ITERATE>",
                cost=total_cost,
                answer=final_answer,
                trace=trace,
                success=success,
                error=None if success else f"Answer mismatch: got '{final_answer}', expected '{ground_truth}'"
            )
            
        except Exception as e:
            return OracleResult(
                action="<ITERATE>",
                cost=float('inf'),
                answer="",
                trace=[],
                success=False,
                error=f"Exception: {str(e)}"
            )

class CostCalculator:
    """Simple cost calculator for different operations."""
    
    def __init__(self):
        # Energy costs in Wh (from your existing system)
        self.retrieval_costs = {
            "fast": 0.01,    # BM25
            "smart": 0.05,   # Dense
            "deep": 0.15     # Hybrid
        }
        
        # Token-based costs
        self.token_costs = {
            "slm": {"input": 0.001, "output": 0.002},  # per 1k tokens
            "llm": {"input": 0.01, "output": 0.03}     # per 1k tokens  
        }
        
    def calculate_generation_cost(self, input_tokens: float, output_tokens: float, model_type: str) -> float:
        """Calculate cost for text generation."""
        costs = self.token_costs.get(model_type, self.token_costs["slm"])
        return (input_tokens / 1000 * costs["input"]) + (output_tokens / 1000 * costs["output"])