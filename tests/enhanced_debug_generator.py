#!/usr/bin/env python3
"""
Enhanced Oracle Trajectory Generator with Detailed Debug State Tracking

This script generates Oracle trajectories with comprehensive debug information:
- Per-query debug files showing evolving state
- Context window management tracking  
- Subquery decomposition evolution
- Document filtering and ranking processes
- Step-by-step reasoning progression

Each run creates a timestamped directory with individual query debug files.
"""
import json
import os
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import inspect

class DebugTracker:
    """Tracks detailed debug information for Oracle execution"""
    
    def __init__(self, run_dir: str, query_id: str):
        self.run_dir = run_dir
        self.query_id = query_id
        self.debug_file = os.path.join(run_dir, f"query_{query_id}_debug.json")
        self.step_counter = 0
        self.debug_log = {
            "query_id": query_id,
            "steps": [],
            "context_evolution": [],
            "subquery_evolution": [],
            "document_filtering": [],
            "tier_attempts": []
        }
    
    def log_tier_attempt(self, tier: int, tier_name: str, description: str):
        """Log a tier attempt"""
        self.debug_log["tier_attempts"].append({
            "step": self.step_counter,
            "tier": tier,
            "tier_name": tier_name,
            "description": description,
            "timestamp": datetime.now().isoformat()
        })
        self.step_counter += 1
    
    def log_context_window(self, operation: str, context_before: str, context_after: str, 
                          added_docs: List[Dict] = None, removed_info: str = None):
        """Track context window evolution"""
        preview_length = 1600
        self.debug_log["context_evolution"].append({
            "step": self.step_counter,
            "operation": operation,
            "context_before_length": len(context_before),
            "context_after_length": len(context_after),
            "context_before_preview": context_before[:preview_length] + "..." if len(context_before) > preview_length else context_before,
            "context_after_preview": context_after[:preview_length] + "..." if len(context_after) > preview_length else context_after,
            "added_documents": len(added_docs) if added_docs else 0,
            "added_doc_titles": [doc.get("title", "No title")[:50] for doc in (added_docs or [])],
            "removed_info": removed_info,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_subquery_evolution(self, operation: str, original_query: str, 
                              subqueries_before: List[str], subqueries_after: List[str],
                              reasoning: str = ""):
        """Track subquery decomposition and evolution"""
        self.debug_log["subquery_evolution"].append({
            "step": self.step_counter,
            "operation": operation,
            "original_query": original_query,
            "subqueries_before": subqueries_before,
            "subqueries_after": subqueries_after,
            "num_subqueries_before": len(subqueries_before),
            "num_subqueries_after": len(subqueries_after),
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_document_filtering(self, operation: str, docs_before: List[Dict], 
                             docs_after: List[Dict], filter_criteria: str, 
                             scores: List[float] = None):
        """Track document filtering and ranking"""
        self.debug_log["document_filtering"].append({
            "step": self.step_counter,
            "operation": operation,
            "docs_before_count": len(docs_before),
            "docs_after_count": len(docs_after),
            "filter_criteria": filter_criteria,
            "docs_before_titles": [doc.get("title", "No title")[:50] for doc in docs_before],
            "docs_after_titles": [doc.get("title", "No title")[:50] for doc in docs_after],
            "relevance_scores": scores[:5] if scores else None,  # Top 5 scores
            "docs_removed": len(docs_before) - len(docs_after),
            "timestamp": datetime.now().isoformat()
        })
    
    def log_step(self, operation: str, input_data: Any, output_data: Any, 
                metadata: Dict = None):
        """Log a general processing step"""
        preview_length = 1600
        self.debug_log["steps"].append({
            "step": self.step_counter,
            "operation": operation,
            "input_preview": str(input_data)[:preview_length] + "..." if len(str(input_data)) > preview_length else str(input_data),
            "output_preview": str(output_data)[:preview_length] + "..." if len(str(output_data)) > preview_length else str(output_data),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
        self.step_counter += 1
    
    def save_debug_log(self):
        """Save debug log to file"""
        os.makedirs(os.path.dirname(self.debug_file), exist_ok=True)
        with open(self.debug_file, 'w') as f:
            json.dump(self.debug_log, f, indent=2)
        
        # Also create a human-readable summary
        summary_file = self.debug_file.replace('.json', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Debug Summary for Query {self.query_id}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("TIER ATTEMPTS:\n")
            for attempt in self.debug_log["tier_attempts"]:
                f.write(f"  Step {attempt['step']}: Tier {attempt['tier']} ({attempt['tier_name']})\n")
                f.write(f"    {attempt['description']}\n\n")
            
            f.write("\nCONTEXT WINDOW EVOLUTION:\n")
            for ctx in self.debug_log["context_evolution"]:
                f.write(f"  Step {ctx['step']}: {ctx['operation']}\n")
                f.write(f"    Length: {ctx['context_before_length']} → {ctx['context_after_length']}\n")
                f.write(f"    Added docs: {ctx['added_documents']}\n")
                if ctx['removed_info']:
                    f.write(f"    Removed: {ctx['removed_info']}\n")
                f.write("\n")
            
            f.write("\nSUBQUERY EVOLUTION:\n")
            for sq in self.debug_log["subquery_evolution"]:
                f.write(f"  Step {sq['step']}: {sq['operation']}\n")
                f.write(f"    Before: {sq['subqueries_before']}\n")
                f.write(f"    After: {sq['subqueries_after']}\n")
                if sq['reasoning']:
                    f.write(f"    Reasoning: {sq['reasoning']}\n")
                f.write("\n")

# Enhanced Mock Components with Debug Tracking
class DebugMockGenerator:
    def __init__(self, debug_tracker: DebugTracker = None):
        self.debug_tracker = debug_tracker
        # Same question tier mapping as before
        self.question_tiers = {
            "Were Scott Derrickson and Ed Wood": 2,  # Needs Smart retrieval
            "What government position was held": 1,  # Fast retrieval works
            "What science fantasy young adult": 5,   # Needs iterative reasoning
            "Are the Laleli Mosque and Esma": 3,     # Needs query refinement  
            'The director of the romantic comedy "Big Stone Gap"': 6  # Needs decomposition
        }
        
        self.correct_answers = {
            "Were Scott Derrickson and Ed Wood": "yes",
            "What government position was held": "Chief of Protocol", 
            "What science fantasy young adult": "Animorphs",
            "Are the Laleli Mosque and Esma": "no",
            'The director of the romantic comedy "Big Stone Gap"': "Greenwich Village, New York City"
        }
    
    def generate(self, prompt: str, context: str = "", **kwargs) -> str:
        """Generate with debug tracking"""
        # Log the generation step
        if self.debug_tracker:
            self.debug_tracker.log_step("GENERATE", 
                                      f"Prompt: {prompt[:100]}...", 
                                      "Generating answer...",
                                      {"context_length": len(context)})
        
        # Detect tier from call stack (same logic as before)
        tier = self._detect_tier_from_stack()
        
        # Find question and generate answer
        answer = self._generate_answer_for_tier(prompt, tier)
        
        # Log the result
        if self.debug_tracker:
            self.debug_tracker.log_step("GENERATE_COMPLETE", 
                                      f"Generated answer for tier {tier}",
                                      answer,
                                      {"tier": tier, "answer_length": len(answer)})
        
        return answer
    
    def _detect_tier_from_stack(self) -> int:
        """Detect tier from call stack"""
        tier = 0
        frame = inspect.currentframe()
        try:
            for i in range(10):
                frame = frame.f_back
                if frame is None:
                    break
                
                func_name = frame.f_code.co_name
                if "_try_pure_slm" in func_name:
                    tier = 0
                    break
                elif "_try_fast_retrieval" in func_name:
                    tier = 1
                    break
                elif "_try_smart_retrieval" in func_name:
                    tier = 2
                    break
                elif "_try_refine_query" in func_name:
                    tier = 3
                    break
                elif "_try_filter_results" in func_name:
                    tier = 4
                    break
                elif "_try_iterative_chain" in func_name:
                    tier = 5
                    break
                elif "_try_full_decomposition" in func_name or "run_full_trajectory" in func_name:
                    tier = 6
                    break
        finally:
            del frame
        
        return tier
    
    def _generate_answer_for_tier(self, prompt: str, tier: int) -> str:
        """Generate answer based on tier capability"""
        question_key = None
        required_tier = 99
        
        for key, tier_need in self.question_tiers.items():
            if key.lower() in prompt.lower():
                question_key = key
                required_tier = tier_need
                break
        
        if question_key and tier >= required_tier:
            answer = self.correct_answers[question_key]
            if random.random() < 0.7:
                return answer
            else:
                return f"Based on analysis: {answer}"
        else:
            wrong_answers = ["I don't have enough information", "Unable to determine", "Context unclear", "Not specified"]
            return random.choice(wrong_answers)

class DebugMockAgent:
    """Enhanced agent with comprehensive debug tracking"""
    
    def __init__(self, generator: DebugMockGenerator, debug_tracker: DebugTracker = None):
        self.generator = generator
        self.llm = generator
        self.debug_tracker = debug_tracker
    
    def _retrieve_tier(self, query: str, tier: str) -> List[Dict]:
        """Mock retrieval with detailed debug tracking"""
        if self.debug_tracker:
            self.debug_tracker.log_tier_attempt(
                {"FAST": 1, "SMART": 2}.get(tier, 3),
                tier,
                f"Attempting {tier} retrieval for: {query[:50]}..."
            )
        
        # Generate realistic documents
        docs = self._generate_realistic_docs(query, tier)
        
        # Log document filtering
        if self.debug_tracker:
            self.debug_tracker.log_document_filtering(
                f"{tier}_RETRIEVAL",
                docs,  # Before (same as after in mock)
                docs,  # After
                f"Retrieved top documents using {tier} search",
                [random.uniform(0.7, 0.95) for _ in docs]  # Mock relevance scores
            )
        
        # Log context window impact
        context_before = ""
        context_after = self._create_context_from_docs(docs)
        
        if self.debug_tracker:
            self.debug_tracker.log_context_window(
                f"{tier}_RETRIEVAL_CONTEXT",
                context_before,
                context_after,
                docs,
                None
            )
        
        # Display retrieved documents (as before)
        retrieval_type = {"FAST": "BM25 keyword search", "SMART": "Dense embedding search"}.get(tier, "Hybrid (BM25 + Dense)")
        print(f"        📋 Retrieved {len(docs)} docs via {retrieval_type}:")
        for i, doc in enumerate(docs):
            title_short = doc["title"][:60] + ("..." if len(doc["title"]) > 60 else "")
            text_short = doc["text"][:80] + ("..." if len(doc["text"]) > 80 else "")
            print(f"          {i+1}. {title_short}")
            print(f"             {text_short}")
        
        return docs
    
    def _generate_realistic_docs(self, query: str, tier: str) -> List[Dict]:
        """Generate realistic documents based on query and tier"""
        tier_num = {"FAST": 1, "SMART": 2}.get(tier, 3)
        retrieval_type = {"FAST": "BM25 keyword search", "SMART": "Dense embedding search"}.get(tier, "Hybrid (BM25 + Dense)")
        
        docs = []
        for i in range(3):
            if "Scott Derrickson" in query or "Ed Wood" in query:
                docs.append({
                    "title": f"Director Biography {i+1} ({retrieval_type})",
                    "text": f"Scott Derrickson is an American film director known for horror films. Ed Wood was also an American filmmaker, famous for his B-movies in the 1950s. Retrieved via {retrieval_type}.",
                    "score": random.uniform(0.7, 0.95)
                })
            # Add more realistic document generation for other query types...
            else:
                docs.append({
                    "title": f"Generic Document {i+1} ({retrieval_type})",
                    "text": f"This is a generic document about {query[:30]}... Retrieved using {retrieval_type} for tier {tier_num}.",
                    "score": random.uniform(0.5, 0.8)
                })
        
        return docs
    
    def _create_context_from_docs(self, docs: List[Dict]) -> str:
        """Create context string from documents"""
        context_parts = []
        for doc in docs:
            context_parts.append(f"Title: {doc['title']}\nContent: {doc['text']}\n")
        return "\n---\n".join(context_parts)
    
    def run_full_trajectory(self, query: str, force_decompose: bool = False):
        """Enhanced full decomposition with detailed debug tracking"""
        if self.debug_tracker:
            self.debug_tracker.log_tier_attempt(6, "FULL_DECOMPOSITION", 
                                              f"Starting full decomposition for: {query[:50]}...")
        
        # Simulate decomposition steps with subquery evolution
        original_subqueries = []
        
        # Step 1: Initial decomposition
        if "Scott Derrickson" in query and "Ed Wood" in query:
            initial_subqueries = [
                "What is Scott Derrickson's nationality?",
                "What is Ed Wood's nationality?", 
                "Are Scott Derrickson and Ed Wood from the same country?"
            ]
        elif "government position" in query:
            initial_subqueries = [
                "Who portrayed Corliss Archer in Kiss and Tell?",
                "What government position did this actress hold?",
                "What was the specific title of this government role?"
            ]
        else:
            initial_subqueries = [
                f"Extract key entities from: {query}",
                f"Find relationships between entities", 
                f"Synthesize final answer"
            ]
        
        if self.debug_tracker:
            self.debug_tracker.log_subquery_evolution(
                "INITIAL_DECOMPOSITION",
                query,
                original_subqueries,
                initial_subqueries,
                "Breaking down complex query into manageable subqueries"
            )
        
        # Step 2: Refine subqueries
        refined_subqueries = []
        for sq in initial_subqueries:
            if "nationality" in sq:
                refined_subqueries.append(sq.replace("nationality", "country of origin and citizenship"))
            else:
                refined_subqueries.append(sq)
        
        if self.debug_tracker:
            self.debug_tracker.log_subquery_evolution(
                "REFINE_SUBQUERIES",
                query,
                initial_subqueries,
                refined_subqueries,
                "Making subqueries more specific and searchable"
            )
        
        # Step 3: Execute subqueries with context tracking
        context_accumulation = ""
        
        for i, subquery in enumerate(refined_subqueries):
            # Simulate retrieval for each subquery
            docs = self._generate_realistic_docs(subquery, "SMART")
            new_context = self._create_context_from_docs(docs)
            
            context_before = context_accumulation
            context_accumulation += f"\n\nSubquery {i+1} Results:\n{new_context}"
            
            if self.debug_tracker:
                self.debug_tracker.log_context_window(
                    f"SUBQUERY_{i+1}_RETRIEVAL",
                    context_before,
                    context_accumulation,
                    docs,
                    None
                )
        
        # Final answer generation
        answer = self.generator.generate(f"Decomposition query: {query}")
        
        print(f"        🔬 Decomposition Steps:")
        print(f"          1. Initial decomposition: {len(initial_subqueries)} subqueries")
        print(f"          2. Subquery refinement: Enhanced specificity")
        print(f"          3. Sequential subquery execution: {len(refined_subqueries)} retrievals")
        print(f"          4. Context accumulation: {len(context_accumulation)} chars")
        print(f"          5. Final synthesis: Generated answer")
        
        # Mock result object
        class MockResult:
            def __init__(self, answer, subqueries):
                self.final_answer = f"Full decomposition result: {answer}"
                self.steps = [
                    type('obj', (object,), {
                        'action': 'DECOMPOSE', 
                        'input': query, 
                        'output': f'Generated {len(subqueries)} subqueries',
                        'retrieved_docs': []
                    })(),
                    type('obj', (object,), {
                        'action': 'RETRIEVE_SMART', 
                        'input': f'{len(subqueries)} subqueries', 
                        'output': f'Accumulated context: {len(context_accumulation)} chars',
                        'retrieved_docs': docs  # Last retrieval docs
                    })(),
                    type('obj', (object,), {
                        'action': 'GENERATE_FINAL', 
                        'input': query, 
                        'output': answer,
                        'retrieved_docs': []
                    })()
                ]
        
        return MockResult(answer, refined_subqueries)

def create_debug_run(questions: List[Dict], run_name: str = None) -> str:
    """Create a new debug run directory with enhanced trajectory generation"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"debug_run_{timestamp}"
    run_dir = f"/home/wcrawford/rag_eval/results/debug_runs/{run_name}"
    
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"🎯 Starting Enhanced Debug Trajectory Generation")
    print(f"📁 Run directory: {run_dir}")
    print(f"📊 Processing {len(questions)} questions with detailed state tracking")
    print("=" * 70)
    
    # Process each question with individual debug tracking
    all_results = []
    
    for i, q_data in enumerate(questions):
        query_id = f"{i:03d}"
        question = q_data["question"]
        expected = q_data["answer"]
        
        print(f"\n🔍 Query {query_id}: {question[:70]}...")
        print(f"🎯 Expected: {expected}")
        
        # Create debug tracker for this query
        debug_tracker = DebugTracker(run_dir, query_id)
        
        # Create components with debug tracking
        generator = DebugMockGenerator(debug_tracker)
        agent = DebugMockAgent(generator, debug_tracker)
        
        try:
            # We'll integrate with the actual Oracle here
            # For now, simulate the process
            debug_tracker.log_step("QUERY_START", question, "Starting Oracle execution")
            
            # Mock Oracle execution (we'll replace this with real Oracle)
            result = {
                "query_id": query_id,
                "question": question,
                "expected": expected,
                "debug_file": debug_tracker.debug_file
            }
            
            # Save debug information
            debug_tracker.save_debug_log()
            
            print(f"✅ Debug log saved: query_{query_id}_debug.json")
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing query {query_id}: {e}")
            debug_tracker.log_step("ERROR", str(e), "Query processing failed")
            debug_tracker.save_debug_log()
    
    # Create run summary
    summary_file = os.path.join(run_dir, "run_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            "run_name": run_name,
            "timestamp": timestamp,
            "questions_processed": len(all_results),
            "run_directory": run_dir,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n🎉 Debug run complete!")
    print(f"📁 Results directory: {run_dir}")
    print(f"📊 Processed {len(all_results)} queries with detailed debug logs")
    
    return run_dir

if __name__ == "__main__":
    # Load sample questions for testing
    questions_file = "/home/wcrawford/rag_eval/data/processed/questions.json"
    
    print("📚 Loading HotPotQA questions...")
    with open(questions_file, 'r') as f:
        all_questions = json.load(f)
    
    # Test with first 3 questions
    sample_questions = all_questions[:3]
    
    # Create enhanced debug run
    run_dir = create_debug_run(sample_questions, "enhanced_debug_test")
    
    print(f"\n🔍 Check the debug files in: {run_dir}")
    print("📋 Each query has detailed state tracking and evolution logs!")