"""
Enhanced LLM serving implementations with Prompt system support and caching.
"""

import os
import json
import time
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from maestro.serving.llm_serving import LLMServingABC, APILLMServing
from maestro.core.prompt import PromptABC


class EnhancedLLMServing(LLMServingABC):
    """
    增强的LLM服务实现，支持Prompt系统、缓存和批处理优化。
    """

    def __init__(
        self,
        base_serving: LLMServingABC,
        enable_cache: bool = True,
        cache_dir: str = "./llm_cache",
        cache_ttl: int = 3600,  # 缓存过期时间（秒）
        enable_batch_optimization: bool = True,
        batch_size: int = 10
    ):
        """
        初始化增强的LLM服务。

        Args:
            base_serving: 基础LLM服务实现
            enable_cache: 是否启用缓存
            cache_dir: 缓存目录
            cache_ttl: 缓存过期时间（秒）
            enable_batch_optimization: 是否启用批处理优化
            batch_size: 批处理大小
        """
        self.base_serving = base_serving
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir)
        self.cache_ttl = cache_ttl
        self.enable_batch_optimization = enable_batch_optimization
        self.batch_size = batch_size

        self.logger = logging.getLogger(self.__class__.__name__)

        # 初始化缓存目录
        if self.enable_cache:
            self.cache_dir.mkdir(exist_ok=True)
            self.prompt_cache = {}  # 内存缓存

    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """
        生成缓存键。

        Args:
            prompt: 提示文本
            **kwargs: 额外参数

        Returns:
            str: 缓存键
        """
        # 创建包含prompt和参数的字符串
        cache_input = f"{prompt}_{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """
        从缓存中获取结果。

        Args:
            cache_key: 缓存键

        Returns:
            Optional[str]: 缓存的结果，如果不存在则返回None
        """
        if not self.enable_cache:
            return None

        # 检查内存缓存
        if cache_key in self.prompt_cache:
            cached_data = self.prompt_cache[cache_key]
            if cached_data['timestamp'] + self.cache_ttl > time.time():
                self.logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                return cached_data['result']
            else:
                # 缓存过期，删除
                del self.prompt_cache[cache_key]

        # 检查磁盘缓存
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data['timestamp'] + self.cache_ttl > time.time():
                        # 加载到内存缓存
                        self.prompt_cache[cache_key] = cached_data
                        self.logger.debug(f"Disk cache hit for key: {cache_key[:8]}...")
                        return cached_data['result']
                    else:
                        # 缓存过期，删除文件
                        cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to load cache file {cache_file}: {e}")

        return None

    def _save_to_cache(self, cache_key: str, result: str) -> None:
        """
        保存结果到缓存。

        Args:
            cache_key: 缓存键
            result: 要缓存的结果
        """
        if not self.enable_cache:
            return

        import time
        cached_data = {
            'result': result,
            'timestamp': time.time()
        }

        # 保存到内存缓存
        self.prompt_cache[cache_key] = cached_data

        # 保存到磁盘缓存
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache file {cache_file}: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本，支持缓存。

        Args:
            prompt: 输入提示
            **kwargs: 额外参数

        Returns:
            str: 生成的文本
        """
        # 检查缓存
        cache_key = self._get_cache_key(prompt, **kwargs)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # 调用基础服务生成
        try:
            result = self.base_serving.generate(prompt, **kwargs)
            # 保存到缓存
            self._save_to_cache(cache_key, result)
            return result
        except Exception as e:
            self.logger.error(f"Failed to generate text: {e}")
            raise

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        批量生成文本，支持缓存和优化。

        Args:
            prompts: 输入提示列表
            **kwargs: 额外参数

        Returns:
            List[str]: 生成的文本列表
        """
        results = []
        uncached_prompts = []
        uncached_indices = []

        # 检查缓存
        for i, prompt in enumerate(prompts):
            cache_key = self._get_cache_key(prompt, **kwargs)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                results.append(cached_result)
            else:
                results.append(None)  # 占位符
                uncached_prompts.append(prompt)
                uncached_indices.append(i)

        # 批量处理未缓存的提示
        if uncached_prompts:
            if self.enable_batch_optimization and len(uncached_prompts) > 1:
                # 使用基础服务的批处理
                uncached_results = self.base_serving.batch_generate(uncached_prompts, **kwargs)
            else:
                # 逐个处理
                uncached_results = []
                for prompt in uncached_prompts:
                    try:
                        result = self.base_serving.generate(prompt, **kwargs)
                        uncached_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Failed to generate text for prompt: {e}")
                        uncached_results.append("")

            # 填充结果并保存到缓存
            for i, (prompt, result) in enumerate(zip(uncached_prompts, uncached_results)):
                original_index = uncached_indices[i]
                results[original_index] = result

                # 保存到缓存
                cache_key = self._get_cache_key(prompt, **kwargs)
                self._save_to_cache(cache_key, result)

        return results

    def generate_with_prompt(self, prompt: PromptABC, **kwargs) -> str:
        """
        使用Prompt对象生成文本。

        Args:
            prompt: Prompt对象
            **kwargs: 用于格式化Prompt的参数

        Returns:
            str: 生成的文本
        """
        try:
            formatted_prompt = prompt.format(**kwargs)
            return self.generate(formatted_prompt)
        except Exception as e:
            self.logger.error(f"Failed to generate with prompt: {e}")
            raise

    def batch_generate_with_prompts(
        self,
        prompts: List[PromptABC],
        prompt_kwargs: List[Dict[str, Any]]
    ) -> List[str]:
        """
        批量使用Prompt对象生成文本。

        Args:
            prompts: Prompt对象列表
            prompt_kwargs: 每个Prompt的格式化参数列表

        Returns:
            List[str]: 生成的文本列表
        """
        if len(prompts) != len(prompt_kwargs):
            raise ValueError("prompts and prompt_kwargs must have the same length")

        try:
            formatted_prompts = []
            for prompt, kwargs in zip(prompts, prompt_kwargs):
                formatted_prompt = prompt.format(**kwargs)
                formatted_prompts.append(formatted_prompt)

            return self.batch_generate(formatted_prompts)
        except Exception as e:
            self.logger.error(f"Failed to batch generate with prompts: {e}")
            raise

    def clear_cache(self) -> None:
        """清空缓存。"""
        if not self.enable_cache:
            return

        # 清空内存缓存
        self.prompt_cache.clear()

        # 清空磁盘缓存
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.warning(f"Failed to clear disk cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息。

        Returns:
            Dict[str, Any]: 缓存统计信息
        """
        if not self.enable_cache:
            return {"cache_enabled": False}

        memory_cache_size = len(self.prompt_cache)
        disk_cache_files = list(self.cache_dir.glob("*.pkl"))
        disk_cache_size = len(disk_cache_files)

        # 计算磁盘缓存总大小
        total_disk_size = sum(f.stat().st_size for f in disk_cache_files)

        return {
            "cache_enabled": True,
            "memory_cache_entries": memory_cache_size,
            "disk_cache_entries": disk_cache_size,
            "disk_cache_size_bytes": total_disk_size,
            "cache_dir": str(self.cache_dir)
        }


class LocalLLMServing(LLMServingABC):
    """
    本地LLM服务实现，支持本地模型推理。
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: str = "auto",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0
    ):
        """
        初始化本地LLM服务。

        Args:
            model_name: 模型名称
            model_path: 模型路径（可选）
            device: 设备类型（"cpu", "cuda", "auto"）
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.logger = logging.getLogger(self.__class__.__name__)
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """延迟加载模型。"""
        if self._model is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            # 确定设备
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            # 加载tokenizer和模型
            model_path = self.model_path or self.model_name
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )

            if device == "cpu":
                self._model = self._model.to(device)

            self.logger.info(f"Loaded model {self.model_name} on {device}")

        except ImportError:
            raise ImportError(
                "Please install transformers and torch: "
                "pip install transformers torch"
            )
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本。

        Args:
            prompt: 输入提示
            **kwargs: 额外参数

        Returns:
            str: 生成的文本
        """
        self._load_model()

        try:
            try:
                import torch
            except ImportError:
                torch = None

            # 编码输入
            inputs = self._tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available() and self._model.device.type == "cuda":
                inputs = inputs.to(self._model.device)

            # 生成参数
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)

            # 生成文本
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )

            # 解码输出
            generated_text = self._tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            self.logger.error(f"Failed to generate text: {e}")
            raise

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        批量生成文本。

        Args:
            prompts: 输入提示列表
            **kwargs: 额外参数

        Returns:
            List[str]: 生成的文本列表
        """
        # 简单实现：逐个处理
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to generate text for prompt: {e}")
                results.append("")

        return results