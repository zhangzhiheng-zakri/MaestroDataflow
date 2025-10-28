"""
Text generation AI operators for MaestroDataflow.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd

from maestro.core.operator import OperatorABC
from maestro.core.prompt import PromptABC, DIYPromptABC, StandardPrompt, prompt_restrict
from maestro.serving.llm_serving import LLMServingABC
from maestro.utils.storage import MaestroStorage


@prompt_restrict(PromptABC, DIYPromptABC, StandardPrompt)
class PromptedGenerator(OperatorABC):
    """
    基于Prompt的文本生成操作符。
    支持使用自定义或标准Prompt模板生成文本。
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        prompt: PromptABC,
        input_column: str,
        output_column: str = "generated_text",
        batch_size: int = 10,
        **generation_kwargs
    ):
        """
        初始化文本生成操作符。

        Args:
            llm_serving: LLM服务实例
            prompt: Prompt对象
            input_column: 输入数据列名
            output_column: 输出结果列名
            batch_size: 批处理大小
            **generation_kwargs: 传递给LLM的额外参数
        """
        super().__init__()
        self.llm_serving = llm_serving
        self.prompt = prompt
        self.input_column = input_column
        self.output_column = output_column
        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs

        # 验证Prompt类型
        if not self.validate_prompts([self.prompt]):
            raise ValueError(f"Invalid prompt type: {type(self.prompt)}")

    def run(self, storage: Union[MaestroStorage, 'FileStorage'], **kwargs) -> Dict[str, Any]:
        """
        执行文本生成操作。

        Args:
            storage: 存储对象
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 操作结果
        """
        self.log_operation_start(**kwargs)

        try:
            # 读取数据
            if hasattr(storage, 'operator_step') and storage.operator_step == -1:
                storage = storage.step()

            df = storage.read(output_type="dataframe")

            if self.input_column not in df.columns:
                raise ValueError(f"Input column '{self.input_column}' not found in data")

            # 批量生成文本
            generated_texts = []
            input_texts = df[self.input_column].tolist()

            # 分批处理
            for i in range(0, len(input_texts), self.batch_size):
                batch_inputs = input_texts[i:i + self.batch_size]
                batch_prompts = []
                batch_kwargs = []

                for input_text in batch_inputs:
                    # 准备Prompt参数
                    prompt_kwargs = {
                        "input": input_text,
                        "text": input_text,  # 兼容不同的变量名
                        **kwargs
                    }
                    batch_prompts.append(self.prompt)
                    batch_kwargs.append(prompt_kwargs)

                # 批量生成
                if hasattr(self.llm_serving, 'batch_generate_with_prompts'):
                    batch_results = self.llm_serving.batch_generate_with_prompts(
                        batch_prompts, batch_kwargs, **self.generation_kwargs
                    )
                else:
                    # 逐个生成（兼容性）
                    batch_results = []
                    for prompt, prompt_kwargs in zip(batch_prompts, batch_kwargs):
                        formatted_prompt = prompt.format(**prompt_kwargs)
                        result = self.llm_serving.generate(formatted_prompt, **self.generation_kwargs)
                        batch_results.append(result)

                generated_texts.extend(batch_results)

                self.logger.info(f"Generated {len(batch_results)} texts in batch {i // self.batch_size + 1}")

            # 添加生成的文本列
            df[self.output_column] = generated_texts

            # 写入结果
            path = storage.write(df)

            result = {
                "path": path,
                "generated_count": len(generated_texts),
                "input_column": self.input_column,
                "output_column": self.output_column
            }

            self.log_operation_end(result)
            return result

        except Exception as e:
            self.handle_error(e, "text generation")


class TextSummarizer(OperatorABC):
    """
    文本摘要操作符。
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        input_column: str,
        output_column: str = "summary",
        max_length: int = 200,
        language: str = "中文",
        batch_size: int = 10
    ):
        """
        初始化文本摘要操作符。

        Args:
            llm_serving: LLM服务实例
            input_column: 输入文本列名
            output_column: 输出摘要列名
            max_length: 摘要最大长度
            language: 摘要语言
            batch_size: 批处理大小
        """
        super().__init__()
        self.llm_serving = llm_serving
        self.input_column = input_column
        self.output_column = output_column
        self.max_length = max_length
        self.language = language
        self.batch_size = batch_size

        # 创建摘要Prompt
        self.prompt = StandardPrompt(
            "summarize",
            max_length=max_length,
            language=language
        )

    def run(self, storage: Union[MaestroStorage, 'FileStorage'], **kwargs) -> Dict[str, Any]:
        """
        执行文本摘要操作。

        Args:
            storage: 存储对象
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 操作结果
        """
        self.log_operation_start(**kwargs)

        try:
            # 读取数据
            if hasattr(storage, 'operator_step') and storage.operator_step == -1:
                storage = storage.step()

            df = storage.read(output_type="dataframe")

            if self.input_column not in df.columns:
                raise ValueError(f"Input column '{self.input_column}' not found in data")

            # 生成摘要
            summaries = []
            input_texts = df[self.input_column].tolist()

            # 分批处理
            for i in range(0, len(input_texts), self.batch_size):
                batch_inputs = input_texts[i:i + self.batch_size]
                batch_prompts = []

                for input_text in batch_inputs:
                    formatted_prompt = self.prompt.format(content=input_text)
                    batch_prompts.append(formatted_prompt)

                # 批量生成摘要
                batch_summaries = self.llm_serving.batch_generate(batch_prompts)
                summaries.extend(batch_summaries)

                self.logger.info(f"Generated {len(batch_summaries)} summaries in batch {i // self.batch_size + 1}")

            # 添加摘要列
            df[self.output_column] = summaries

            # 写入结果
            path = storage.write(df)

            result = {
                "path": path,
                "summarized_count": len(summaries),
                "input_column": self.input_column,
                "output_column": self.output_column
            }

            self.log_operation_end(result)
            return result

        except Exception as e:
            self.handle_error(e, "text summarization")


class TextClassifier(OperatorABC):
    """
    文本分类操作符。
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        input_column: str,
        categories: List[str],
        output_column: str = "category",
        include_confidence: bool = False,
        confidence_column: str = "confidence",
        batch_size: int = 10
    ):
        """
        初始化文本分类操作符。

        Args:
            llm_serving: LLM服务实例
            input_column: 输入文本列名
            categories: 分类类别列表
            output_column: 输出分类列名
            include_confidence: 是否包含置信度
            confidence_column: 置信度列名
            batch_size: 批处理大小
        """
        super().__init__()
        self.llm_serving = llm_serving
        self.input_column = input_column
        self.categories = categories
        self.output_column = output_column
        self.include_confidence = include_confidence
        self.confidence_column = confidence_column
        self.batch_size = batch_size

        # 创建分类Prompt
        categories_str = "、".join(categories)
        self.prompt = StandardPrompt(
            "classify",
            categories=categories_str
        )

    def run(self, storage: Union[MaestroStorage, 'FileStorage'], **kwargs) -> Dict[str, Any]:
        """
        执行文本分类操作。

        Args:
            storage: 存储对象
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 操作结果
        """
        self.log_operation_start(**kwargs)

        try:
            # 读取数据
            if hasattr(storage, 'operator_step') and storage.operator_step == -1:
                storage = storage.step()

            df = storage.read(output_type="dataframe")

            if self.input_column not in df.columns:
                raise ValueError(f"Input column '{self.input_column}' not found in data")

            # 执行分类
            classifications = []
            confidences = []
            input_texts = df[self.input_column].tolist()

            # 分批处理
            for i in range(0, len(input_texts), self.batch_size):
                batch_inputs = input_texts[i:i + self.batch_size]
                batch_prompts = []

                for input_text in batch_inputs:
                    formatted_prompt = self.prompt.format(text=input_text)
                    batch_prompts.append(formatted_prompt)

                # 批量分类
                batch_results = self.llm_serving.batch_generate(batch_prompts)

                # 解析分类结果
                for result in batch_results:
                    category, confidence = self._parse_classification_result(result)
                    classifications.append(category)
                    confidences.append(confidence)

                self.logger.info(f"Classified {len(batch_results)} texts in batch {i // self.batch_size + 1}")

            # 添加分类列
            df[self.output_column] = classifications
            if self.include_confidence:
                df[self.confidence_column] = confidences

            # 写入结果
            path = storage.write(df)

            result = {
                "path": path,
                "classified_count": len(classifications),
                "input_column": self.input_column,
                "output_column": self.output_column,
                "categories": self.categories
            }

            if self.include_confidence:
                result["confidence_column"] = self.confidence_column

            self.log_operation_end(result)
            return result

        except Exception as e:
            self.handle_error(e, "text classification")

    def _parse_classification_result(self, result: str) -> tuple[str, float]:
        """
        解析分类结果，提取类别和置信度。

        Args:
            result: LLM返回的分类结果

        Returns:
            tuple[str, float]: (类别, 置信度)
        """
        # 简单的解析逻辑，可以根据需要改进
        result = result.strip()

        # 查找匹配的类别
        for category in self.categories:
            if category in result:
                # 尝试提取置信度（如果有的话）
                confidence = 1.0  # 默认置信度

                # 简单的置信度提取（可以改进）
                import re
                confidence_match = re.search(r'(\d+(?:\.\d+)?)[%％]', result)
                if confidence_match:
                    confidence = float(confidence_match.group(1)) / 100

                return category, confidence

        # 如果没有找到匹配的类别，返回第一个类别和低置信度
        return self.categories[0], 0.1