#!/usr/bin/env python3
"""
Main script to run the baseline RAG pipeline evaluation.

This script orchestrates the full workflow:
1. Load or create corpus
2. Build retrieval index
3. Load evaluation dataset
4. Run RAG pipeline
5. Evaluate with RAGChecker
6. Generate report
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corpus import CorpusProcessor
from retriever import FaissRetriever, BM25Retriever, create_retriever
from generator import create_generator
from pipeline import RAGPipeline
from evaluator import RAGEvaluator


def setup_logging(config: dict):
    """Setup logging configuration."""
    log_level = config["logging"]["level"]
    log_file = config["logging"].get("log_file")
    log_to_console = config["logging"].get("log_to_console", True)
    
    handlers = []
    
    if log_to_console:
        handlers.append(logging.StreamHandler())
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def prepare_corpus(config: dict, force_rebuild: bool = False):
    """Prepare corpus and passages."""
    passages_path = Path(config["retriever"]["passages_path"])
    
    if passages_path.exists() and not force_rebuild:
        logging.info(f"Passages already exist at {passages_path}")
        return
    
    logging.info("Preparing corpus...")
    
    processor = CorpusProcessor(
        dataset_name=config["corpus"]["dataset_name"],
        dataset_config=config["corpus"]["dataset_config"],
        num_documents=config["corpus"]["num_documents"],
        cache_dir=config["corpus"]["cache_dir"],
        max_chunk_size=config["chunking"]["max_chunk_size"],
        overlap=config["chunking"]["overlap"],
        min_chunk_size=config["chunking"]["min_chunk_size"],
        shuffle=config["corpus"].get("shuffle", False),
        shuffle_seed=config["corpus"].get("shuffle_seed"),
        skip_first=config["corpus"].get("skip_first", 0)
    )
    
    # Load and process documents
    documents = processor.load_dataset()
    passages = processor.process_documents(documents)
    
    # Save passages
    processor.save_passages(passages, str(passages_path))
    
    logging.info(f"Corpus prepared: {len(passages)} passages from {len(documents)} documents")


def build_index(config: dict, force_rebuild: bool = False):
    """Build retrieval index (BM25 or Faiss)."""
    retriever_type = config["retriever"].get("type", "bm25")
    passages_path = Path(config["retriever"]["passages_path"])
    
    # Determine index path based on retriever type
    if retriever_type == "bm25":
        index_path = Path(config["retriever"].get("bm25_index_path", "./data/indexes/faiss.bm25.pkl"))
    else:
        index_path = Path(config["retriever"]["index_path"])
    
    if index_path.exists() and not force_rebuild:
        logging.info(f"Index already exists at {index_path}")
        return
    
    logging.info(f"Building {retriever_type} retrieval index...")
    
    # Load passages
    processor = CorpusProcessor()
    passages = processor.load_passages(str(passages_path))
    
    # Initialize retriever based on type
    if retriever_type == "bm25":
        retriever = BM25Retriever()
        retriever.build_index(passages)
        retriever.save_index(
            index_path=str(index_path),
            passages_path=str(passages_path)
        )
    else:
        retriever = FaissRetriever(
            model_name=config["retriever"]["model_name"],
            embedding_dim=config["retriever"]["embedding_dim"],
            normalize_embeddings=config["retriever"]["normalize_embeddings"],
            batch_size=config["retriever"]["batch_size"]
        )
        retriever.build_index(passages, index_type=config["retriever"]["index_type"])
        retriever.save_index(
            index_path=str(index_path),
            passages_path=str(passages_path),
            embeddings_path=config["retriever"]["embeddings_path"]
        )
    
    logging.info(f"{retriever_type.upper()} index built successfully")


def run_evaluation(config: dict, delay: float = 1.0, test_mode: bool = False, test_samples: int = 5):
    """Run full RAG evaluation."""
    logging.info("Starting RAG evaluation...")
    
    if test_mode:
        logging.info(f"TEST MODE ENABLED: Using {test_samples} samples")
    
    # Load pipeline
    retriever_type = config["retriever"].get("type", "bm25")
    logging.info(f"Loading RAG pipeline with {retriever_type} retriever...")
    
    # Create retriever using factory
    retriever = create_retriever(
        retriever_type=retriever_type,
        model_name=config["retriever"].get("model_name"),
        embedding_dim=config["retriever"].get("embedding_dim", 384),
        normalize_embeddings=config["retriever"].get("normalize_embeddings", True),
        batch_size=config["retriever"].get("batch_size", 32)
    )
    
    # Determine index path based on retriever type
    if retriever_type == "bm25":
        index_path = config["retriever"].get("bm25_index_path", "./data/indexes/faiss.bm25.pkl")
    else:
        index_path = config["retriever"]["index_path"]
    
    retriever.load_index(
        index_path=index_path,
        passages_path=config["retriever"]["passages_path"]
    )
    
    # Create generator using factory (supports Gemini and Ollama)
    generator_config = config["generator"].copy()
    generator_config["system_prompt"] = config["prompt"]["system"]
    generator = create_generator(generator_config)
    
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        top_k=config["retrieval"]["top_k"],
        score_threshold=config["retrieval"]["score_threshold"],
        prompt_template=config["prompt"]["user_template"]
    )
    
    # Initialize evaluator
    evaluator = RAGEvaluator(
        pipeline=pipeline,
        ragchecker_batch_size=config["ragchecker"]["batch_size"],
        use_cache=config["ragchecker"]["use_cache"],
        cache_dir=config["ragchecker"]["cache_dir"],
        test_mode=test_mode,
        test_num_samples=test_samples,
        ragchecker_extractor=config["ragchecker"].get("extractor_model"),
        ragchecker_checker=config["ragchecker"].get("checker_model")
    )
    
    # Load evaluation dataset
    eval_examples = evaluator.load_evaluation_dataset(
        dataset_name=config["evaluation"]["dataset"],
        split=config["evaluation"]["split"],
        num_samples=config["evaluation"]["num_samples"]
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(
        eval_examples=eval_examples,
        delay=delay,
        verbose=True
    )
    
    # Print and save results
    evaluator.print_summary(results)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(config["evaluation"]["output_dir"]) / f"evaluation_{timestamp}.json"
    evaluator.save_results(results, str(output_path))
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run baseline RAG pipeline evaluation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../config_local.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--rebuild-corpus",
        action="store_true",
        help="Force rebuild corpus and passages"
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild Faiss index"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation (only prepare data)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls (seconds) for rate limiting"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (5 samples for quick iteration)"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=5,
        help="Number of samples to use in test mode"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("RAG BASELINE EVALUATION PIPELINE")
    logger.info("="*60)
    
    try:
        # Step 1: Prepare corpus
        logger.info("\n[1/3] Preparing corpus...")
        prepare_corpus(config, force_rebuild=args.rebuild_corpus)
        
        # Step 2: Build index
        logger.info("\n[2/3] Building retrieval index...")
        build_index(config, force_rebuild=args.rebuild_index)
        
        # Step 3: Run evaluation
        if not args.skip_eval:
            logger.info("\n[3/3] Running evaluation...")
            results = run_evaluation(
                config,
                delay=args.delay,
                test_mode=args.test,
                test_samples=args.test_samples
            )
            
            logger.info("\n" + "="*60)
            logger.info("EVALUATION COMPLETE")
            logger.info("="*60)
        else:
            logger.info("\n[3/3] Skipping evaluation (--skip-eval)")
            logger.info("\nData preparation complete. Run without --skip-eval to evaluate.")
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
