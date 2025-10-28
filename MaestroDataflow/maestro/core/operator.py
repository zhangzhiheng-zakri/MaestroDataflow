"""
Operator abstract base for MaestroDataflow.
Enhanced with AI capabilities while maintaining backward compatibility.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Union
import logging

from maestro.utils.storage import FileStorage


class OperatorABC(ABC):
    """
    所有Operator的抽象基类，约定统一的运行接口。
    AI增强版本，支持Prompt系统和日志记录，同时保持向后兼容。
    """

    def __init__(self):
        """
        初始化操作符。
        设置日志记录器和Prompt类型限制。
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        # Prompt类型限制，默认为空元组（允许所有类型）
        if not hasattr(self, 'ALLOWED_PROMPTS'):
            self.ALLOWED_PROMPTS = tuple()

    @abstractmethod
    def run(self, storage: Union[FileStorage, 'MaestroStorage'], **kwargs) -> Dict[str, Any]:
        """
        执行该操作符并将结果写入到storage的下一步文件。

        Args:
            storage: 存储对象，支持FileStorage（向后兼容）和新的MaestroStorage
            **kwargs: 额外的参数

        Returns:
            Dict[str, Any]: 操作的输出元数据
        """
        raise NotImplementedError

    def validate_prompts(self, prompts: List[Any]) -> bool:
        """
        验证Prompt类型是否被允许。

        Args:
            prompts: Prompt对象列表

        Returns:
            bool: 验证是否通过
        """
        if not self.ALLOWED_PROMPTS:
            return True

        for prompt in prompts:
            if not isinstance(prompt, self.ALLOWED_PROMPTS):
                self.logger.warning(
                    f"Prompt type {type(prompt)} not allowed. "
                    f"Allowed types: {self.ALLOWED_PROMPTS}"
                )
                return False
        return True

    def log_operation_start(self, **kwargs):
        """
        记录操作开始的日志。

        Args:
            **kwargs: 操作参数
        """
        self.logger.info(f"Starting {self.__class__.__name__} operation")
        if kwargs:
            self.logger.debug(f"Operation parameters: {kwargs}")

    def log_operation_end(self, result: Dict[str, Any]):
        """
        记录操作结束的日志。

        Args:
            result: 操作结果
        """
        self.logger.info(f"Completed {self.__class__.__name__} operation")
        if result:
            self.logger.debug(f"Operation result: {result}")

    def handle_error(self, error: Exception, context: Optional[str] = None) -> None:
        """
        统一的错误处理方法。

        Args:
            error: 发生的异常
            context: 错误上下文信息
        """
        error_msg = f"Error in {self.__class__.__name__}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {str(error)}"

        self.logger.error(error_msg, exc_info=True)
        raise error