"""
Generator module supporting Google Gemini API and Ollama local models.

This module handles:
- Connecting to Gemini API with secure API key management
- Connecting to Ollama for local model inference
- Formatting prompts with retrieved context
- Generating answers with configurable parameters
- Error handling and retries
"""

import logging
import os
import time
import json
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import requests

logger = logging.getLogger(__name__)

# Load environment variables from multiple possible locations

load_dotenv(Path.home() / ".env")  # Then try home directory


class GeminiGenerator:
    """Generator using Google Gemini API."""
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_p: float = 1.0,
        timeout: int = 30,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize Gemini generator.
        
        Args:
            model_name: Gemini model name
            api_key_env: Environment variable name for API key
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            timeout: API timeout in seconds
            system_prompt: Optional system prompt
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.system_prompt = system_prompt
        
        # Get API key from environment
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable '{api_key_env}'. "
                f"Please set it in your .env file or export it."
            )
        
        # Configure Gemini
        genai.configure(api_key=api_key)  # type: ignore[attr-defined]
        
        # Initialize model with generation config
        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_tokens,
        }
        
        self.model = genai.GenerativeModel(  # type: ignore[attr-defined]
            model_name=self.model_name,
            generation_config=generation_config,  # type: ignore[arg-type]
            system_instruction=self.system_prompt
        )
        
        logger.info(f"Initialized Gemini generator: {model_name}")
    
    def format_context(self, passages: List[Dict]) -> str:
        """
        Format retrieved passages into context string.
        
        Args:
            passages: List of passage dictionaries with 'text' and 'title'
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, passage in enumerate(passages, 1):
            title = passage.get('title', 'Unknown')
            text = passage.get('text', '')
            score = passage.get('retrieval_score', 0.0)
            
            context_parts.append(
                f"[{i}] {title} (relevance: {score:.3f})\n{text}"
            )
        
        return "\n\n".join(context_parts)
    
    def generate(
        self,
        question: str,
        context_passages: List[Dict],
        prompt_template: Optional[str] = None,
        max_retries: int = 3,
        max_cooldown_cycles: int = 3,
        cooldown_wait: int = 60
    ) -> Dict:
        """
        Generate answer given question and context.
        
        Args:
            question: User question
            context_passages: List of retrieved passage dictionaries
            prompt_template: Optional custom prompt template
            max_retries: Maximum number of retries on API errors (per cooldown cycle)
            max_cooldown_cycles: Maximum number of 1-minute cooldown cycles to attempt
            cooldown_wait: Seconds to wait during cooldown (default 60)
            
        Returns:
            Dictionary with 'answer', 'context', and metadata
        """
        # Format context
        context = self.format_context(context_passages)
        
        # Build prompt
        if prompt_template is None:
            prompt_template = """Context:
{context}

Question: {question}

Answer:"""
        
        prompt = prompt_template.format(context=context, question=question)
        
        # Generate with retries and cooldown cycles
        for cooldown_cycle in range(max_cooldown_cycles):
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(
                        prompt,
                        request_options={"timeout": self.timeout}
                    )
                    
                    # Extract answer
                    answer = response.text.strip()
                    
                    # Build result
                    result = {
                        "question": question,
                        "answer": answer,
                        "context": context,
                        "context_passages": context_passages,
                        "prompt": prompt,
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "finish_reason": response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
                    }
                    
                    return result
                    
                except Exception as e:
                    error_str = str(e).lower()
                    is_quota_error = "quota" in error_str or "rate limit" in error_str
                    
                    logger.warning(
                        f"Generation attempt {attempt + 1}/{max_retries} "
                        f"(cooldown cycle {cooldown_cycle + 1}/{max_cooldown_cycles}) failed: {e}"
                    )
                    
                    if attempt < max_retries - 1:
                        # Exponential backoff within a cycle
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        # All retries in this cycle failed
                        if is_quota_error and cooldown_cycle < max_cooldown_cycles - 1:
                            # Quota error: wait 1 minute and try another cycle
                            logger.warning(
                                f"Rate limit/quota exceeded. Waiting {cooldown_wait} seconds "
                                f"before cooldown cycle {cooldown_cycle + 2}/{max_cooldown_cycles}..."
                            )
                            time.sleep(cooldown_wait)
                            break  # Break inner loop to start new cooldown cycle
                        else:
                            # Not a quota error, or exhausted all cooldown cycles
                            if cooldown_cycle >= max_cooldown_cycles - 1:
                                logger.error(
                                    f"Generation failed after {max_cooldown_cycles} cooldown cycles "
                                    f"of {max_retries} attempts each"
                                )
                            else:
                                logger.error(f"Generation failed after {max_retries} attempts")
                            raise
        
        # This should never be reached, but satisfy type checker
        raise RuntimeError("Generation failed: exhausted all retry cycles")
    
    def batch_generate(
        self,
        questions: List[str],
        context_passages_list: List[List[Dict]],
        prompt_template: Optional[str] = None,
        delay: float = 0.0,
        skip_on_persistent_error: bool = True
    ) -> List[Dict]:
        """
        Generate answers for multiple questions.
        
        Args:
            questions: List of questions
            context_passages_list: List of context passage lists (one per question)
            prompt_template: Optional custom prompt template
            delay: Delay between API calls (seconds) to respect rate limits
            skip_on_persistent_error: If True, skip questions that fail after all retries
            
        Returns:
            List of result dictionaries
        """
        if len(questions) != len(context_passages_list):
            raise ValueError("Number of questions must match number of context lists")
        
        results = []
        for i, (question, context_passages) in enumerate(zip(questions, context_passages_list), 1):
            logger.info(f"Generating answer {i}/{len(questions)}")
            
            try:
                result = self.generate(question, context_passages, prompt_template)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate answer for question {i}: {e}")
                
                if skip_on_persistent_error:
                    logger.warning(f"Skipping question {i} and continuing with next question")
                    results.append({
                        "question": question,
                        "answer": None,
                        "error": str(e),
                        "skipped": True
                    })
                else:
                    # Re-raise to stop batch processing
                    raise
            
            # Rate limiting
            if delay > 0 and i < len(questions):
                time.sleep(delay)
        
        return results


class OllamaGenerator:
    """Generator using Ollama for local model inference."""
    
    def __init__(
        self,
        model_name: str = "mistral",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_p: float = 1.0,
        timeout: int = 120,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize Ollama generator.
        
        Args:
            model_name: Ollama model name (e.g., "mistral", "llama3.1:8b", "llama3.1:70b")
            base_url: Ollama API base URL
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            timeout: API timeout in seconds (higher for local models)
            system_prompt: Optional system prompt
        """
        # Remove "ollama/" prefix if present (from litellm format)
        self.model_name = model_name.replace("ollama/", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.system_prompt = system_prompt
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            available_models = [m["name"] for m in response.json().get("models", [])]
            
            if self.model_name not in available_models:
                logger.warning(
                    f"Model '{self.model_name}' not found in Ollama. "
                    f"Available models: {available_models}. "
                    f"Run: ollama pull {self.model_name}"
                )
            else:
                logger.info(f"Found Ollama model: {self.model_name}")
                
        except requests.RequestException as e:
            logger.warning(
                f"Could not connect to Ollama at {self.base_url}: {e}. "
                f"Make sure Ollama is running (ollama serve)"
            )
        
        logger.info(f"Initialized Ollama generator: {self.model_name}")
    
    def format_context(self, passages: List[Dict]) -> str:
        """
        Format retrieved passages into context string.
        
        Args:
            passages: List of passage dictionaries with 'text' and 'title'
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, passage in enumerate(passages, 1):
            title = passage.get('title', 'Unknown')
            text = passage.get('text', '')
            score = passage.get('retrieval_score', 0.0)
            
            context_parts.append(
                f"[{i}] {title} (relevance: {score:.3f})\n{text}"
            )
        
        return "\n\n".join(context_parts)
    
    def generate(
        self,
        question: str,
        context_passages: List[Dict],
        prompt_template: Optional[str] = None,
        max_retries: int = 3,
        max_cooldown_cycles: int = 1,
        cooldown_wait: int = 60
    ) -> Dict:
        """
        Generate answer given question and context.
        
        Args:
            question: User question
            context_passages: List of retrieved passage dictionaries
            prompt_template: Optional custom prompt template
            max_retries: Maximum number of retries on API errors
            max_cooldown_cycles: Maximum number of cooldown cycles (typically 1 for local)
            cooldown_wait: Seconds to wait during cooldown
            
        Returns:
            Dictionary with 'answer', 'context', and metadata
        """
        # Format context
        context = self.format_context(context_passages)
        
        # Build prompt
        if prompt_template is None:
            prompt_template = """Context:
{context}

Question: {question}

Answer:"""
        
        prompt = prompt_template.format(context=context, question=question)
        
        # Build messages for Ollama API
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Generate with retries
        for cooldown_cycle in range(max_cooldown_cycles):
            for attempt in range(max_retries):
                try:
                    # Call Ollama API
                    response = requests.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": self.model_name,
                            "messages": messages,
                            "stream": False,
                            "options": {
                                "temperature": self.temperature,
                                "top_p": self.top_p,
                                "num_predict": self.max_tokens
                            }
                        },
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    
                    # Parse response
                    result_data = response.json()
                    answer = result_data.get("message", {}).get("content", "").strip()
                    
                    # Build result
                    result = {
                        "question": question,
                        "answer": answer,
                        "context": context,
                        "context_passages": context_passages,
                        "prompt": prompt,
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "finish_reason": result_data.get("done_reason", "stop")
                    }
                    
                    return result
                    
                except Exception as e:
                    logger.warning(
                        f"Generation attempt {attempt + 1}/{max_retries} "
                        f"(cooldown cycle {cooldown_cycle + 1}/{max_cooldown_cycles}) failed: {e}"
                    )
                    
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        # All retries in this cycle failed
                        if cooldown_cycle < max_cooldown_cycles - 1:
                            logger.warning(
                                f"Waiting {cooldown_wait} seconds "
                                f"before cooldown cycle {cooldown_cycle + 2}/{max_cooldown_cycles}..."
                            )
                            time.sleep(cooldown_wait)
                            break
                        else:
                            logger.error(
                                f"Generation failed after {max_cooldown_cycles} cooldown cycles "
                                f"of {max_retries} attempts each"
                            )
                            raise
        
        # This should never be reached, but satisfy type checker
        raise RuntimeError("Generation failed: exhausted all retry cycles")
    
    def batch_generate(
        self,
        questions: List[str],
        context_passages_list: List[List[Dict]],
        prompt_template: Optional[str] = None,
        delay: float = 0.0,
        skip_on_persistent_error: bool = True
    ) -> List[Dict]:
        """
        Generate answers for multiple questions.
        
        Args:
            questions: List of questions
            context_passages_list: List of context passage lists (one per question)
            prompt_template: Optional custom prompt template
            delay: Delay between API calls (seconds)
            skip_on_persistent_error: If True, skip questions that fail after all retries
            
        Returns:
            List of result dictionaries
        """
        if len(questions) != len(context_passages_list):
            raise ValueError("Number of questions must match number of context lists")
        
        results = []
        for i, (question, context_passages) in enumerate(zip(questions, context_passages_list), 1):
            logger.info(f"Generating answer {i}/{len(questions)}")
            
            try:
                result = self.generate(question, context_passages, prompt_template)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate answer for question {i}: {e}")
                
                if skip_on_persistent_error:
                    logger.warning(f"Skipping question {i} and continuing with next question")
                    results.append({
                        "question": question,
                        "answer": None,
                        "error": str(e),
                        "skipped": True
                    })
                else:
                    raise
            
            # Rate limiting (less critical for local models but still useful)
            if delay > 0 and i < len(questions):
                time.sleep(delay)
        
        return results


def create_generator(config: Dict) -> object:
    """
    Factory function to create appropriate generator based on config.
    
    Args:
        config: Configuration dictionary with generator settings
        
    Returns:
        GeminiGenerator or OllamaGenerator instance
    """
    provider = config.get("provider", "gemini").lower()
    
    if provider == "ollama":
        return OllamaGenerator(
            model_name=config["model_name"],
            base_url=config.get("base_url", "http://localhost:11434"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 256),
            top_p=config.get("top_p", 1.0),
            timeout=config.get("timeout", 120),
            system_prompt=config.get("system_prompt")
        )
    elif provider == "gemini":
        return GeminiGenerator(
            model_name=config["model_name"],
            api_key_env=config.get("api_key_env", "GEMINI_API_KEY"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 256),
            top_p=config.get("top_p", 1.0),
            timeout=config.get("timeout", 30),
            system_prompt=config.get("system_prompt")
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: 'gemini', 'ollama'"
        )


def main():
    """Example usage of GeminiGenerator."""
    import yaml
    
    # Load config
    with open("config.yaml_local", 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=config["logging"]["level"],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize generator using factory
    generator_config = config["generator"].copy()
    generator_config["system_prompt"] = config["prompt"]["system"]
    generator = create_generator(generator_config)
    
    # Test with mock context
    test_question = "What is machine learning?"
    test_passages = [
        {
            "title": "Machine Learning",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "retrieval_score": 0.95
        },
        {
            "title": "Neural Networks",
            "text": "Neural networks are computing systems inspired by biological neural networks. They are used in machine learning to recognize patterns and solve complex problems.",
            "retrieval_score": 0.87
        }
    ]
    
    logger.info(f"Test question: {test_question}\n")
    
    # Generate answer (generator is either GeminiGenerator or OllamaGenerator)
    if hasattr(generator, 'generate'):
        result = generator.generate(  # type: ignore[union-attr]
            question=test_question,
            context_passages=test_passages,
            prompt_template=config["prompt"]["user_template"]
        )
    else:
        raise TypeError(f"Generator {type(generator)} does not have generate method")
    
    logger.info(f"Generated answer:\n{result['answer']}\n")
    logger.info(f"Finish reason: {result['finish_reason']}")


if __name__ == "__main__":
    main()
