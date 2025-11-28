"""
Evaluation module using RAGChecker framework.

This module handles:
- Loading evaluation datasets (HotpotQA, Natural Questions, etc.)
- Running RAG pipeline on evaluation queries
- Computing metrics using RAGChecker
- Generating evaluation reports
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import yaml
from datasets import load_dataset
from ragchecker import RAGResults, RAGChecker
from ragchecker.container import RetrievedDoc

from pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG pipeline using RAGChecker."""
    
    def __init__(
        self,
        pipeline: RAGPipeline,
        ragchecker_batch_size: int = 8,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        test_mode: bool = False,
        test_num_samples: int = 5,
        ragchecker_extractor: str = "ollama/mistral",
        ragchecker_checker: str = "nli"
    ):
        """
        Initialize RAG evaluator.
        
        Args:
            pipeline: Initialized RAGPipeline
            ragchecker_batch_size: Batch size for RAGChecker
            use_cache: Whether to use RAGChecker cache
            cache_dir: Cache directory for RAGChecker
            test_mode: If True, use reduced sample size for faster testing
            test_num_samples: Number of samples to use in test mode
            ragchecker_extractor: Model for extracting claims (uses litellm format)
            ragchecker_checker: Model for checking claims (uses litellm format)
        """
        self.pipeline = pipeline
        self.ragchecker_batch_size = ragchecker_batch_size
        self.test_mode = test_mode
        self.test_num_samples = test_num_samples
        
        # Initialize RAGChecker
        logger.info("Initializing RAGChecker...")
        if test_mode:
            logger.info(f"TEST MODE ENABLED: Will use only {test_num_samples} samples")
        
        logger.info(f"RAGChecker using extractor: {ragchecker_extractor}")
        logger.info(f"RAGChecker using checker: {ragchecker_checker}")
        
        # Get API key from environment for RAGChecker (only needed for API-based models)
        import os
        api_key = os.getenv("GEMINI_API_KEY")  # Used for Gemini, not needed for Ollama
        
        # Monkey patch NLI checker to handle triplet claims
        # RAGChecker's NLI implementation has a bug where triplet claims aren't converted to strings
        if ragchecker_checker == "nli":
            import refchecker.checker.nli_checker as nli_module
            original_check = nli_module.NLIChecker._check
            
            def patched_check(self, claims, references, **kwargs):
                # Check if is_joint to determine structure
                is_joint = kwargs.get('is_joint', False)
                
                # DEBUG: Log what we received
                logger.warning(f"=== NLI PATCH ENTRY ===")
                logger.warning(f"is_joint: {is_joint}")
                logger.warning(f"claims type: {type(claims)}, len: {len(claims) if hasattr(claims, '__len__') else 'N/A'}")
                logger.warning(f"references type: {type(references)}, len: {len(references) if hasattr(references, '__len__') else 'N/A'}")
                logger.warning(f"First claim: {claims[0] if len(claims) > 0 else 'empty'}")
                logger.warning(f"First reference type: {type(references[0]) if len(references) > 0 else 'empty'}")
                
                # Convert triplet claims to strings
                def convert_claim(claim):
                    if isinstance(claim, list) and len(claim) == 3:
                        # Triplet format: ['subject', 'relation', 'object'] -> 'subject relation object'
                        return f'{claim[0]} {claim[1]} {claim[2]}'
                    elif isinstance(claim, str):
                        return claim
                    else:
                        return str(claim)
                
                # Convert reference to string  
                def convert_reference(ref):
                    if isinstance(ref, str):
                        return ref
                    elif hasattr(ref, 'text'):  # RetrievedDoc object
                        return ref.text
                    elif isinstance(ref, list):
                        # List of documents or strings
                        return ' '.join([r.text if hasattr(r, 'text') else str(r) for r in ref])
                    else:
                        return str(ref)
                
                if is_joint:
                    # Joint mode: claims and references are nested lists
                    # Structure: [[claim1, claim2], [claim3], ...]
                    converted_claims = []
                    converted_references = []
                    
                    for i, claim_list in enumerate(claims):
                        reference = references[i]
                        ref_str = convert_reference(reference)
                        
                        if isinstance(claim_list, list):
                            # Convert each claim in the list
                            converted_claims.append([convert_claim(c) for c in claim_list])
                        else:
                            # Single claim
                            converted_claims.append(convert_claim(claim_list))
                        
                        converted_references.append(ref_str)
                    
                    logger.warning(f"NLI Patch (joint) - {len(converted_claims)} examples, sample: {converted_claims[0] if converted_claims else 'empty'}")
                else:
                    # Non-joint mode: claims and references are already flat lists of strings
                    # checker_base.py already flattened them, just convert triplets
                    converted_claims = [convert_claim(c) for c in claims]
                    converted_references = [convert_reference(r) for r in references]
                    
                    logger.warning(f"NLI Patch (non-joint) - {len(converted_claims)} flat claims, sample: {converted_claims[:3] if converted_claims else 'empty'}")
                
                # DEBUG: Log what we're sending
                logger.warning(f"=== NLI PATCH OUTPUT ===")
                logger.warning(f"converted_claims type: {type(converted_claims)}, len: {len(converted_claims)}")
                logger.warning(f"converted_references type: {type(converted_references)}, len: {len(converted_references)}")
                logger.warning(f"First converted claim: {converted_claims[0] if len(converted_claims) > 0 else 'empty'}")
                logger.warning(f"First converted claim type: {type(converted_claims[0]) if len(converted_claims) > 0 else 'empty'}")
                logger.warning(f"First converted reference: {converted_references[0][:100] if len(converted_references) > 0 else 'empty'}...")
                logger.warning(f"First converted reference type: {type(converted_references[0]) if len(converted_references) > 0 else 'empty'}")
                
                # Cast to expected types for the original function
                from typing import cast, List, Union
                claims_arg = cast(List[Union[str, List[str], List[List[str]]]], converted_claims)
                return original_check(self, claims_arg, converted_references, **kwargs)
            
            nli_module.NLIChecker._check = patched_check
            logger.info("Applied NLI checker monkey patch for triplet claim handling")
        
        # NLI checker cannot handle joint checking (nested structures)
        # Only LLM-based checkers support is_joint=True
        use_joint_check = ragchecker_checker != "nli"
        
        self.checker = RAGChecker(
            extractor_name=ragchecker_extractor,
            checker_name=ragchecker_checker,
            batch_size_extractor=ragchecker_batch_size,
            batch_size_checker=ragchecker_batch_size,
            cache_dir=cache_dir if use_cache else None,
            openai_api_key=api_key,  # Only used for API-based models (Gemini, etc.), not Ollama
            joint_check=use_joint_check,  # Disable for NLI, enable for LLM checkers
        )
        
    def load_evaluation_dataset(
        self,
        dataset_name: str = "hotpotqa",
        split: str = "validation",
        num_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Load evaluation dataset.
        
        Args:
            dataset_name: Name of dataset (hotpotqa, nq, triviaqa)
            split: Dataset split (train, validation, test)
            num_samples: Number of samples to load (None for all)
            
        Returns:
            List of evaluation examples with 'question' and 'answer' fields
        """
        # Override num_samples if in test mode
        if self.test_mode:
            num_samples = self.test_num_samples
            logger.info(f"TEST MODE: Limiting to {num_samples} samples")
        
        logger.info(f"Loading {dataset_name} dataset ({split} split)...")
        
        # Load dataset based on name
        if dataset_name == "hotpotqa":
            dataset = load_dataset("hotpot_qa", "fullwiki", split=split, trust_remote_code=True)
            examples = []
            for item in dataset:  # type: ignore[union-attr]
                examples.append({
                    "question": item["question"],  # type: ignore[index]
                    "answer": item["answer"],  # type: ignore[index]
                    "supporting_facts": item.get("supporting_facts", {})  # type: ignore[union-attr]
                })
        
        elif dataset_name == "nq" or dataset_name == "natural_questions":
            dataset = load_dataset("natural_questions", split=split, trust_remote_code=True)
            examples = []
            for item in dataset:  # type: ignore[union-attr]
                # Extract short answer
                annotations = item["annotations"]  # type: ignore[index]
                if annotations and annotations["short_answers"]:
                    short_answer = annotations["short_answers"][0]["text"]
                    examples.append({
                        "question": item["question"]["text"],  # type: ignore[index]
                        "answer": short_answer
                    })
        
        elif dataset_name == "triviaqa":
            dataset = load_dataset("trivia_qa", "unfiltered", split=split, trust_remote_code=True)
            examples = []
            for item in dataset:  # type: ignore[union-attr]
                examples.append({
                    "question": item["question"],  # type: ignore[index]
                    "answer": item["answer"]["value"]  # type: ignore[index]
                })
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Limit samples
        if num_samples:
            examples = examples[:num_samples]
        
        logger.info(f"Loaded {len(examples)} evaluation examples")
        return examples
    
    def run_evaluation(
        self,
        eval_examples: List[Dict],
        delay: float = 1.0,
        verbose: bool = False
    ) -> Dict:
        """
        Run evaluation on dataset.
        
        Args:
            eval_examples: List of evaluation examples
            delay: Delay between generation calls
            verbose: Whether to log progress
            
        Returns:
            Dictionary with results and metrics
        """
        # Extract questions
        questions = [ex["question"] for ex in eval_examples]
        # Ensure ground truth answers are strings (handle both string and list formats)
        ground_truth_answers = []
        for ex in eval_examples:
            answer = ex["answer"]
            if isinstance(answer, list):
                # If it's a list, join or take the first one
                ground_truth_answers.append(answer[0] if answer else "")
            else:
                ground_truth_answers.append(str(answer) if answer else "")
        
        # Run RAG pipeline
        logger.info(f"Running RAG pipeline on {len(questions)} questions...")
        rag_results = self.pipeline.batch_run(
            questions=questions,
            verbose=verbose,
            delay=delay
        )
        
        # Prepare RAGChecker input
        logger.info("Preparing RAGChecker input...")
        ragchecker_inputs = []
        
        for i, result in enumerate(rag_results):
            # Skip if error or skipped
            if "error" in result or result.get("skipped", False):
                logger.warning(f"Skipping question {i} due to error or skip: {result.get('error', 'skipped')}")
                continue
            
            # Skip if answer is None
            if result.get("answer") is None:
                logger.warning(f"Skipping question {i} due to None answer")
                continue
            
            # Format retrieved chunks as RetrievedDoc objects
            retrieved_contexts = [
                RetrievedDoc(
                    doc_id=f"{i}_{j}",
                    text=p["text"]
                )
                for j, p in enumerate(result.get("retrieved_passages", []))
            ]
            
            ragchecker_inputs.append({
                "query_id": str(i),  # Required by RAGResult
                "query": str(result["question"]),  # Ensure string
                "gt_answer": str(ground_truth_answers[i]),  # Ensure string
                "response": str(result["answer"]),  # Ensure string
                "retrieved_context": retrieved_contexts  # List of RetrievedDoc objects
            })
        
        # Check if we have any valid results
        if not ragchecker_inputs:
            logger.error("No valid results to evaluate!")
            return {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "num_evaluated": 0,
                    "num_total": len(eval_examples),
                    "num_skipped": len(eval_examples),
                    "retriever_model": self.pipeline.retriever.model_name,
                    "generator_model": self.pipeline.generator.model_name
                },
                "metrics": {},
                "detailed_results": rag_results,
                "error": "All questions failed or were skipped"
            }
        
        # Create RAGResults object
        logger.info(f"Creating RAGResults from {len(ragchecker_inputs)} valid examples...")
        from ragchecker.container import RAGResult
        rag_result_objects = [RAGResult(**r) for r in ragchecker_inputs]
        rag_results_obj = RAGResults(results=rag_result_objects)
        
        # Debug: Check what we're passing to RAGChecker
        logger.warning("=" * 60)
        logger.warning("DEBUG: Inspecting RAGResults before evaluation")
        logger.warning("=" * 60)
        if len(rag_results_obj.results) > 0:
            first = rag_results_obj.results[0]
            logger.warning(f"First result query: {first.query}")
            logger.warning(f"First result gt_answer type: {type(first.gt_answer)}, value: {repr(first.gt_answer)}")
            logger.warning(f"First result response type: {type(first.response)}, value: {repr(first.response[:100])}")
            logger.warning(f"First result retrieved_context type: {type(first.retrieved_context)}, length: {len(first.retrieved_context) if first.retrieved_context else 0}")
            if first.retrieved_context and len(first.retrieved_context) > 0:
                logger.warning(f"First retrieved doc type: {type(first.retrieved_context[0])}")
                logger.warning(f"First retrieved doc text type: {type(first.retrieved_context[0].text)}")
            logger.warning(f"First result response_claims (before extraction): {first.response_claims}")
            logger.warning(f"First result gt_answer_claims (before extraction): {first.gt_answer_claims}")
        logger.warning("=" * 60)
        
        # Run RAGChecker evaluation
        logger.info("Running RAGChecker evaluation...")
        try:
            metrics = self.checker.evaluate(rag_results_obj)
        except Exception as e:
            logger.error(f"RAGChecker evaluation failed: {e}")
            # Try to get more info about what was extracted
            if len(rag_results_obj.results) > 0:
                first_result = rag_results_obj.results[0]
                logger.error(f"First result claims - response_claims type: {type(first_result.response_claims)}")
                if first_result.response_claims:
                    logger.error(f"First response_claims value: {repr(first_result.response_claims)}")
            raise
        
        # Compile full results
        evaluation_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_evaluated": len(ragchecker_inputs),
                "num_total": len(eval_examples),
                "num_skipped": len(eval_examples) - len(ragchecker_inputs),
                "retriever_model": self.pipeline.retriever.model_name,
                "generator_model": self.pipeline.generator.model_name
            },
            "metrics": metrics,
            "detailed_results": rag_results
        }
        
        return evaluation_results
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation results to {output_file}")
    
    def print_summary(self, results: Dict):
        """
        Print evaluation summary.
        
        Args:
            results: Evaluation results dictionary
        """
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        
        metadata = results["metadata"]
        logger.info(f"Evaluated: {metadata['num_evaluated']}/{metadata['num_total']} questions")
        logger.info(f"Retriever: {metadata['retriever_model']}")
        logger.info(f"Generator: {metadata['generator_model']}")
        logger.info(f"Timestamp: {metadata['timestamp']}")
        
        logger.info("\nMetrics:")
        metrics = results["metrics"]
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"  {metric_name}: {metric_value}")
        
        logger.info("="*60)


# No standalone main() - use experiments/run_baseline.py instead
