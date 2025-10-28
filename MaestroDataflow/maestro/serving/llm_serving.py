"""
LLM (Large Language Model) serving implementations for MaestroDataflow.
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
import logging

try:
    import requests
except ImportError:
    requests = None
import logging
logging.getLogger(__name__).warning("requests library not found. API-based LLM serving will not work.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMServingABC(ABC):
    """
    Abstract base class for LLM serving implementations.
    Defines the interface that all LLM serving implementations must follow.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the model

        Returns:
            The generated text
        """
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text from multiple prompts.

        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters for the model

        Returns:
            List of generated texts
        """
        pass


class APILLMServing(LLMServingABC):
    """
    LLM serving implementation that uses external API services.
    Supports OpenAI, Azure OpenAI, and other compatible API services.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_retries: int = 3,
        retry_delay: int = 5,
        timeout: int = 60,
        max_workers: int = 10,
        api_type: str = "openai"  # "openai", "azure", "custom"
    ):
        """
        Initialize a new APILLMServing instance.

        Args:
            api_url: API endpoint URL
            api_key: API key for authentication
            model_name: Name of the model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds
            max_workers: Maximum number of concurrent workers for batch processing
            api_type: Type of API service ("openai", "azure", or "custom")
        
        Raises:
            ImportError: If requests library is not available
        """
        if requests is None:
            raise ImportError("requests library is required for API-based LLM serving. Please install it with 'pip install requests'")
            
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.max_workers = max_workers
        self.api_type = api_type

        # Set up headers based on API type
        self.headers = self._setup_headers()

    def _setup_headers(self) -> Dict[str, str]:
        """
        Set up request headers based on API type.

        Returns:
            Dictionary of HTTP headers
        """
        if self.api_type == "openai":
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        elif self.api_type == "azure":
            return {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
        else:  # custom
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

    def _prepare_request_body(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Prepare the request body for the API call.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the model

        Returns:
            Dictionary containing the request body
        """
        # Default parameters
        params = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p)
        }

        # Add messages for chat models
        if ("gpt" in self.model_name.lower() or 
            "deepseek" in self.model_name.lower() or 
            "chat" in self.model_name.lower() or 
            kwargs.get("is_chat_model", False)):
            params["messages"] = [
                {"role": "user", "content": prompt}
            ]
        else:
            params["prompt"] = prompt

        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in ["max_tokens", "temperature", "top_p", "is_chat_model"]:
                params[key] = value

        return params

    def _extract_response(self, response_data: Dict[str, Any]) -> str:
        """
        Extract the generated text from the API response.

        Args:
            response_data: API response data

        Returns:
            The generated text

        Raises:
            ValueError: If the response format is not recognized
        """
        try:
            # For chat models (OpenAI GPT)
            if "choices" in response_data and "message" in response_data["choices"][0]:
                return response_data["choices"][0]["message"]["content"]

            # For completion models (OpenAI Davinci, etc.)
            elif "choices" in response_data and "text" in response_data["choices"][0]:
                return response_data["choices"][0]["text"]

            # For Azure OpenAI
            elif "choices" in response_data and "message" in response_data["choices"][0]:
                return response_data["choices"][0]["message"]["content"]

            # For custom APIs that might have different formats
            elif "output" in response_data:
                return response_data["output"]

            else:
                # If we can't find the expected fields, return the raw response
                logger.warning(f"Unrecognized response format: {response_data}")
                return str(response_data)

        except Exception as e:
            logger.error(f"Error extracting response: {e}")
            logger.error(f"Response data: {response_data}")
            raise ValueError(f"Failed to extract response: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt using the API.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the model

        Returns:
            The generated text

        Raises:
            Exception: If the API request fails after all retries
        """
        request_body = self._prepare_request_body(prompt, **kwargs)

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=request_body,
                    timeout=self.timeout
                )

                response.raise_for_status()
                response_data = response.json()

                return self._extract_response(response_data)

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
                    raise Exception(f"API request failed after {self.max_retries} attempts: {e}")

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text from multiple prompts using the API.

        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters for the model

        Returns:
            List of generated texts
        """
        import concurrent.futures

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_prompt = {
                executor.submit(self.generate, prompt, **kwargs): prompt
                for prompt in prompts
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing prompt: {e}")
                    # Add an error message as the result
                    results.append(f"ERROR: {str(e)}")

        return results


class LocalLLMServing(LLMServingABC):
    """
    LLM serving implementation that uses locally hosted models.
    Supports various inference backends like vLLM, HuggingFace Transformers, etc.
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "vllm",  # "vllm", "transformers", "ctransformers", "llama.cpp"
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        **backend_kwargs
    ):
        """
        Initialize a new LocalLLMServing instance.

        Args:
            model_path: Path to the model weights
            backend: Inference backend to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            **backend_kwargs: Additional parameters for the backend
        """
        self.model_path = model_path
        self.backend = backend
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.backend_kwargs = backend_kwargs

        # Initialize the model based on the backend
        self.model = self._initialize_model()

    def _initialize_model(self) -> Any:
        """
        Initialize the model based on the selected backend.

        Returns:
            The initialized model

        Raises:
            ImportError: If the required packages are not installed
            ValueError: If the backend is not supported
        """
        if self.backend == "vllm":
            try:
                from vllm import LLM

                return LLM(
                    model=self.model_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    **self.backend_kwargs
                )
            except ImportError:
                logger.error("vLLM is not installed. Please install it with 'pip install vllm'.")
                raise ImportError("vLLM is not installed")

        elif self.backend == "transformers":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    **self.backend_kwargs
                )

                return {"model": model, "tokenizer": tokenizer}
            except ImportError:
                logger.error("Transformers is not installed. Please install it with 'pip install transformers'.")
                raise ImportError("Transformers is not installed")

        elif self.backend == "ctransformers":
            try:
                from ctransformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **self.backend_kwargs
                )

                return model
            except ImportError:
                logger.error("CTransformers is not installed. Please install it with 'pip install ctransformers'.")
                raise ImportError("CTransformers is not installed")

        elif self.backend == "llama.cpp":
            try:
                from llama_cpp import Llama

                model = Llama(
                    model_path=self.model_path,
                    **self.backend_kwargs
                )

                return model
            except ImportError:
                logger.error("llama-cpp-python is not installed. Please install it with 'pip install llama-cpp-python'.")
                raise ImportError("llama-cpp-python is not installed")

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt using the local model.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the model

        Returns:
            The generated text

        Raises:
            Exception: If the generation fails
        """
        try:
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)

            if self.backend == "vllm":
                try:
                    from vllm import SamplingParams
                except ImportError:
                    raise ImportError("vLLM is not installed. Please install it with 'pip install vllm'.")

                sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature", "top_p"]}
                )

                outputs = self.model.generate(prompt, sampling_params)
                return outputs[0].outputs[0].text

            elif self.backend == "transformers":
                try:
                    import torch
                except ImportError:
                    raise ImportError(
                        "PyTorch is not installed. Please install it with 'pip install torch' "
                        "or use a different backend that does not require PyTorch."
                    )

                model = self.model["model"]
                tokenizer = self.model["tokenizer"]

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature", "top_p"]}
                    )

                return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            elif self.backend == "ctransformers":
                return self.model(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature", "top_p"]}
                )

            elif self.backend == "llama.cpp":
                return self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature", "top_p"]}
                )["choices"][0]["text"]

            else:
                raise ValueError(f"Unsupported backend: {self.backend}")

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise Exception(f"Text generation failed: {e}")

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text from multiple prompts using the local model.

        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters for the model

        Returns:
            List of generated texts
        """
        try:
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)

            if self.backend == "vllm":
                try:
                    from vllm import SamplingParams
                except ImportError:
                    raise ImportError("vLLM is not installed. Please install it with 'pip install vllm'.")

                sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature", "top_p"]}
                )

                outputs = self.model.generate(prompts, sampling_params)
                return [output.outputs[0].text for output in outputs]

            else:
                # For other backends, use parallel processing
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit all tasks and maintain order
                    futures = [
                        executor.submit(self.generate, prompt, **kwargs)
                        for prompt in prompts
                    ]

                    # Collect results in order
                    results = []
                    for i, future in enumerate(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Error processing prompt {i}: {e}")
                            # Add an error message as the result
                            results.append(f"ERROR: {str(e)}")

                return results

        except Exception as e:
            logger.error(f"Error batch generating text: {e}")
            raise Exception(f"Batch text generation failed: {e}")