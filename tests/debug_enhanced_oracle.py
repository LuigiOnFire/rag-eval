"""
Debug-Enhanced Oracle that captures detailed context information
for each tier attempt to feed into our AdvancedDebugTracker.
"""

import sys
import os
sys.path.append('/home/wcrawford/rag_eval/src')

from exploration_oracle import ExplorationOracle, OracleResult
from typing import Dict, List, Optional, Any

class DebugEnhancedOracle(ExplorationOracle):
    """Enhanced Oracle that captures context details for debug tracking"""
    
    def __init__(self, agent, judge, cost_calculator, debug_tracker=None):
        super().__init__(agent, judge, cost_calculator)
        self.debug_tracker = debug_tracker
        
    def _try_tier_with_debug(self, tier_name: str, tier_method, query: str, ground_truth: str) -> OracleResult:
        """Wrap tier execution with debug tracking"""
        print(f"🔍 {tier_name}")
        
        try:
            result = tier_method(query, ground_truth)
            
            # Extract context if available
            context_used = ""
            retrieved_docs = []
            
            # Get context from debug tracker if available
            if self.debug_tracker and hasattr(self.debug_tracker, 'last_context_used'):
                context_used = self.debug_tracker.last_context_used or ""
                
            # Get retrieved docs from debug tracker if available  
            if self.debug_tracker and hasattr(self.debug_tracker, 'last_retrieved_docs'):
                retrieved_docs = self.debug_tracker.last_retrieved_docs or []
                        
            # Log tier attempt with debug tracker
            if self.debug_tracker:
                self.debug_tracker.log_tier_attempt(
                    tier_name=tier_name,
                    tier_cost=result.cost,
                    success=result.success,
                    answer=result.answer,
                    context_used=context_used,
                    retrieved_docs=retrieved_docs,
                    error_message=result.error or ""
                )
                
            if result.success:
                print(f"✅ {tier_name} SUCCESS: {result.answer[:100]}")
            else:
                print(f"❌ {tier_name} failed: {result.error}")
                
            return result
            
        except Exception as e:
            error_result = OracleResult(
                action=tier_name,
                cost=0.0,
                answer="",
                trace=[],
                success=False,
                error=str(e)
            )
            
            # Log error attempt
            if self.debug_tracker:
                self.debug_tracker.log_tier_attempt(
                    tier_name=tier_name,
                    tier_cost=0.0,
                    success=False,
                    answer="",
                    context_used="",
                    retrieved_docs=[],
                    error_message=str(e)
                )
                
            print(f"❌ {tier_name} error: {e}")
            return error_result
    
    def find_gold_trajectory(self, query: str, ground_truth: str) -> Optional[Dict[str, Any]]:
        """Enhanced trajectory finding with debug tracking"""
        
        candidates = []
        
        # === PHASE 1: CHEAP & FAST ===
        
        # Tier 0: Pure SLM
        result = self._try_tier_with_debug("Tier 0: Pure SLM (cost ~$1)", 
                                          self._try_pure_slm, query, ground_truth)
        if result.success:
            candidates.append(result)
            
        # Tier 1: Fast Retrieval  
        result = self._try_tier_with_debug("Tier 1: Fast Retrieval (BM25, cost ~$5)",
                                          self._try_fast_retrieval, query, ground_truth)
        if result.success:
            candidates.append(result)
            
        # Tier 2: Smart Retrieval
        result = self._try_tier_with_debug("Tier 2: Smart Retrieval (Dense, cost ~$8)",
                                          self._try_smart_retrieval, query, ground_truth)
        if result.success:
            candidates.append(result)

        # === PHASE 2: "DECOMPOSE LITE" & SORTING ===
        
        # Tier 3: Refine Query
        result = self._try_tier_with_debug("Tier 3: Refine Query (Rewrite -> Hybrid, cost ~$12)",
                                          self._try_refine_query, query, ground_truth)
        if result.success:
            candidates.append(result)
            
        # Tier 4: Filter Results
        result = self._try_tier_with_debug("Tier 4: Filter Results (High-Recall -> Sort, cost ~$18)",
                                          self._try_filter_results, query, ground_truth)
        if result.success:
            candidates.append(result)

        # === PHASE 3: HEAVY REASONING ===

        # Tier 5: Iterative Chain
        result = self._try_tier_with_debug("Tier 5: Iterative Chain (2-Step Bridge, cost ~$25)",
                                          self._try_iterative_chain, query, ground_truth)
        if result.success:
            candidates.append(result)
            
        # Tier 6: Full Decomposition
        result = self._try_tier_with_debug("Tier 6: Full Decomposition (Complex Logic, cost ~$50+)",
                                          self._try_full_decomposition, query, ground_truth)
        if result.success:
            candidates.append(result)
            
        if not candidates:
            print("💀 All 7 tiers failed - no training data generated")
            return None
            
        # Sort by cost and pick the cheapest winner
        candidates.sort(key=lambda x: x.cost)
        winner = candidates[0]
        
        print(f"🏆 WINNER: {winner.action} (cost: {winner.cost:.3f})")
        
        # Convert to training format with enhanced context
        trajectory = {
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
        
        # Add detailed context if debug tracker has current query log
        if self.debug_tracker and self.debug_tracker.current_query_log:
            trajectory["debug_context"] = {
                "tier_attempts": self.debug_tracker.current_query_log["tier_attempts"],
                "total_cost": sum(attempt["cost"] for attempt in self.debug_tracker.current_query_log["tier_attempts"]),
                "winning_tier_details": next(
                    (attempt for attempt in self.debug_tracker.current_query_log["tier_attempts"] 
                     if attempt["success"]), None
                )
            }
        
        return trajectory