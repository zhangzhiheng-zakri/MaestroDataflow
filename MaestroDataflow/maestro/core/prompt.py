"""
Prompt system for MaestroDataflow AI capabilities.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Union
from functools import wraps
import logging


class PromptABC(ABC):
    """
    Prompt抽象基类，定义所有Prompt的统一接口。
    """

    @abstractmethod
    def format(self, **kwargs) -> str:
        """
        格式化Prompt模板，返回最终的提示文本。

        Args:
            **kwargs: 用于填充模板的变量

        Returns:
            str: 格式化后的提示文本
        """
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """
        验证输入参数是否满足Prompt的要求。

        Args:
            **kwargs: 输入参数

        Returns:
            bool: 验证是否通过
        """
        pass

    def get_required_vars(self) -> List[str]:
        """
        获取Prompt所需的变量列表。

        Returns:
            List[str]: 必需变量列表
        """
        return []


class DIYPromptABC(PromptABC):
    """
    自定义Prompt抽象基类，支持用户自定义模板。
    """

    def __init__(self, template: str, required_vars: Optional[List[str]] = None):
        """
        初始化自定义Prompt。

        Args:
            template: Prompt模板字符串
            required_vars: 必需的变量列表，如果为None则自动从模板中提取
        """
        self.template = template
        self.required_vars = required_vars or self._extract_vars_from_template(template)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _extract_vars_from_template(self, template: str) -> List[str]:
        """
        从模板中提取变量名。

        Args:
            template: 模板字符串

        Returns:
            List[str]: 提取的变量名列表
        """
        import re
        # 提取 {variable} 格式的变量
        variables = re.findall(r'\{(\w+)\}', template)
        return list(set(variables))

    def format(self, **kwargs) -> str:
        """
        格式化自定义Prompt模板。

        Args:
            **kwargs: 用于填充模板的变量

        Returns:
            str: 格式化后的提示文本

        Raises:
            ValueError: 当缺少必需变量时
        """
        if not self.validate_inputs(**kwargs):
            missing_vars = [var for var in self.required_vars if var not in kwargs]
            raise ValueError(f"Missing required variables: {missing_vars}")

        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template formatting error: {e}")

    def validate_inputs(self, **kwargs) -> bool:
        """
        验证输入参数是否包含所有必需变量。

        Args:
            **kwargs: 输入参数

        Returns:
            bool: 验证是否通过
        """
        return all(var in kwargs for var in self.required_vars)

    def get_required_vars(self) -> List[str]:
        """
        获取必需的变量列表。

        Returns:
            List[str]: 必需变量列表
        """
        return self.required_vars.copy()


class StandardPrompt(PromptABC):
    """
    标准Prompt实现，提供常用的预定义模板。
    """

    # 预定义的标准模板
    TEMPLATES = {
        "summarize": "请总结以下内容：\n\n{content}\n\n总结：",
        "translate": "请将以下{source_lang}文本翻译为{target_lang}：\n\n{text}\n\n翻译：",
        "classify": "请对以下文本进行分类，可选类别：{categories}\n\n文本：{text}\n\n分类：",
        "extract": "请从以下文本中提取{extract_type}：\n\n{text}\n\n提取结果：",
        "generate": "请根据以下要求生成内容：\n\n要求：{requirements}\n\n生成内容：",
    }

    def __init__(self, template_name: str, **default_kwargs):
        """
        初始化标准Prompt。

        Args:
            template_name: 模板名称
            **default_kwargs: 默认参数
        """
        if template_name not in self.TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(self.TEMPLATES.keys())}")

        self.template_name = template_name
        self.template = self.TEMPLATES[template_name]
        self.default_kwargs = default_kwargs
        self.required_vars = self._extract_vars_from_template(self.template)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _extract_vars_from_template(self, template: str) -> List[str]:
        """从模板中提取变量名"""
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        return list(set(variables))

    def format(self, **kwargs) -> str:
        """
        格式化标准Prompt模板。

        Args:
            **kwargs: 用于填充模板的变量

        Returns:
            str: 格式化后的提示文本
        """
        # 合并默认参数和传入参数
        merged_kwargs = {**self.default_kwargs, **kwargs}

        if not self.validate_inputs(**merged_kwargs):
            missing_vars = [var for var in self.required_vars if var not in merged_kwargs]
            raise ValueError(f"Missing required variables: {missing_vars}")

        return self.template.format(**merged_kwargs)

    def validate_inputs(self, **kwargs) -> bool:
        """
        验证输入参数。

        Args:
            **kwargs: 输入参数

        Returns:
            bool: 验证是否通过
        """
        merged_kwargs = {**self.default_kwargs, **kwargs}
        return all(var in merged_kwargs for var in self.required_vars)

    def get_required_vars(self) -> List[str]:
        """
        获取必需的变量列表。

        Returns:
            List[str]: 必需变量列表
        """
        return self.required_vars.copy()


def prompt_restrict(*allowed_prompt_types: Type[PromptABC]):
    """
    Prompt类型限制装饰器，用于限制操作符可以使用的Prompt类型。

    Args:
        *allowed_prompt_types: 允许的Prompt类型

    Returns:
        装饰器函数
    """
    def decorator(cls):
        """装饰器实现"""
        if not hasattr(cls, 'ALLOWED_PROMPTS'):
            cls.ALLOWED_PROMPTS = tuple()

        # 合并现有的和新的允许类型
        existing_types = getattr(cls, 'ALLOWED_PROMPTS', tuple())
        cls.ALLOWED_PROMPTS = tuple(set(existing_types + allowed_prompt_types))

        # 添加验证方法
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not hasattr(self, 'logger'):
                self.logger = logging.getLogger(self.__class__.__name__)

        cls.__init__ = new_init

        # 添加Prompt验证方法
        def validate_prompts(self, prompts: List[PromptABC]) -> bool:
            """
            验证Prompt类型是否被允许。

            Args:
                prompts: Prompt列表

            Returns:
                bool: 验证是否通过
            """
            if not self.ALLOWED_PROMPTS:
                return True

            for prompt in prompts:
                if not isinstance(prompt, self.ALLOWED_PROMPTS):
                    self.logger.warning(
                        f"Prompt type {type(prompt)} not allowed. Allowed types: {self.ALLOWED_PROMPTS}"
                    )
                    return False
            return True

        cls.validate_prompts = validate_prompts
        return cls

    return decorator


# 便捷函数
def create_diy_prompt(template: str, required_vars: Optional[List[str]] = None) -> DIYPromptABC:
    """
    创建自定义Prompt的便捷函数。

    Args:
        template: Prompt模板
        required_vars: 必需变量列表

    Returns:
        DIYPromptABC: 自定义Prompt实例
    """
    return DIYPromptABC(template, required_vars)


def create_standard_prompt(template_name: str, **default_kwargs) -> StandardPrompt:
    """
    创建标准Prompt的便捷函数。

    Args:
        template_name: 模板名称
        **default_kwargs: 默认参数

    Returns:
        StandardPrompt: 标准Prompt实例
    """
    return StandardPrompt(template_name, **default_kwargs)