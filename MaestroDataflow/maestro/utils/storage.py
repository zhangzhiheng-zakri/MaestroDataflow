# -*- coding: utf-8 -*-
"""
MaestroDataflow - 数据存储工具

提供对XLSX、CSV、JSON、JSONL、Parquet和Pickle格式数据的读写支持
"""

import os
import pandas as pd
import json
import copy
import pickle
from abc import ABC, abstractmethod
from typing import Any, Literal, List, Dict, Union, Optional
import logging
import numpy as np
try:
    from .vector_db import VectorDatabaseABC, InMemoryVectorDB
except ImportError:
    from maestro.utils.vector_db import VectorDatabaseABC, InMemoryVectorDB
try:
    from .model_cache import ModelCache
except ImportError:
    from maestro.utils.model_cache import ModelCache

class MaestroStorage(ABC):
    """
    Abstract base class for MaestroDataflow storage systems.
    Extended to support AI capabilities including vector databases and model caching.
    """

    def __init__(self):
        """Initialize the storage system."""
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get_keys(self) -> list[str]:
        """
        获取数据中的键列表
        """
        pass

    @abstractmethod
    def read(self, output_type) -> Any:
        """
        读取数据
        """
        pass

    @abstractmethod
    def write(self, data: Any) -> Any:
        """
        写入数据
        """
        pass

    # AI-specific methods
    def add_vectors(
        self,
        vectors: Union[np.ndarray, List[List[float]]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add vectors to the storage system.

        Args:
            vectors: Vector embeddings
            metadata: Associated metadata for each vector
            ids: Unique identifiers for each vector

        Returns:
            bool: Success status
        """
        raise NotImplementedError("Vector storage not supported by this storage type")

    def search_vectors(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold

        Returns:
            List[Dict[str, Any]]: Search results with scores and metadata
        """
        raise NotImplementedError("Vector search not supported by this storage type")

    def cache_model_output(
        self,
        model_name: str,
        input_data: Any,
        output_data: Any,
        ttl: Optional[int] = None
    ) -> str:
        """
        Cache model output.

        Args:
            model_name: Name of the model
            input_data: Input data
            output_data: Output data
            ttl: Time to live in seconds

        Returns:
            str: Cache key
        """
        raise NotImplementedError("Model caching not supported by this storage type")

    def get_cached_model_output(
        self,
        model_name: str,
        input_data: Any
    ) -> Optional[Any]:
        """
        Get cached model output.

        Args:
            model_name: Name of the model
            input_data: Input data

        Returns:
            Optional[Any]: Cached output data if available
        """
        raise NotImplementedError("Model caching not supported by this storage type")


class FileStorage(MaestroStorage):
    """
    File-based storage implementation for MaestroDataflow.
    Enhanced with AI capabilities including vector storage and model caching.
    """

    def __init__(
        self,
        input_file_path: str,
        cache_path: str = "./cache",
        file_name_prefix: str = "maestro_cache",
        cache_type: Literal["json", "jsonl", "csv", "xlsx", "parquet", "pickle"] = "jsonl",
        enable_vector_storage: bool = False,
        enable_model_cache: bool = False,
        vector_db_config: Optional[Dict[str, Any]] = None,
        model_cache_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize FileStorage with optional AI capabilities.

        Args:
            input_file_path: 输入文件路径
            cache_path: 缓存目录路径
            file_name_prefix: 缓存文件名前缀
            cache_type: 缓存文件类型，支持json、jsonl、csv、xlsx
            enable_vector_storage: Whether to enable vector storage
            enable_model_cache: Whether to enable model caching
            vector_db_config: Configuration for vector database
            model_cache_config: Configuration for model cache
        """
        super().__init__()
        self.input_file_path = input_file_path
        self.cache_path = cache_path if cache_path else "./cache"  # 确保cache_path不为空
        self.file_name_prefix = file_name_prefix
        self.cache_type = cache_type
        self.operator_step = -1

        # 检查输入文件是否存在
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"文件 {input_file_path} 不存在，请检查路径")

        # Initialize AI capabilities
        self.vector_db = None
        self.model_cache = None

        if enable_vector_storage:
            vector_config = vector_db_config or {}
            self.vector_db = InMemoryVectorDB(**vector_config)
            self.logger.info("Vector storage enabled")

        if enable_model_cache:
            cache_config = model_cache_config or {
                "cache_type": "hybrid",
                "cache_config": {
                    "memory": {"max_size": 100, "default_ttl": 3600},
                    "disk": {
                        "cache_dir": os.path.join(cache_path, "model_cache"),
                        "max_size_mb": 500,
                        "default_ttl": 86400
                    }
                }
            }
            self.model_cache = ModelCache(**cache_config)
            self.logger.info("Model caching enabled")

    def step(self):
        """
        执行步骤，进入下一步

        Returns:
            FileStorage: 返回自身实例的副本以支持链式调用
        """
        self.operator_step += 1
        return copy.copy(self)

    def reset(self):
        """
        重置步骤计数

        Returns:
            FileStorage: 返回自身实例以支持链式调用
        """
        self.operator_step = -1
        return self

    def get_keys(self) -> list[str]:
        """
        获取数据中的键列表

        Returns:
            list[str]: 键列表
        """
        dataframe = self.read(output_type="dataframe")
        return dataframe.columns.tolist() if isinstance(dataframe, pd.DataFrame) else []

    def read(self, output_type: Literal["dataframe", "dict"] = "dataframe") -> Union[pd.DataFrame, List[Dict]]:
        """
        读取数据

        Args:
            output_type: 输出数据类型，支持dataframe和dict

        Returns:
            Union[pd.DataFrame, List[Dict]]: 读取的数据
        """
        if self.operator_step == -1:
            raise ValueError("请先调用step()方法初始化处理步骤")

        file_path = self._get_cache_file_path(self.operator_step)
        file_ext = os.path.splitext(file_path)[1].lower()[1:]

        dataframe = self._load_file(file_path, file_ext)
        return self._convert_output(dataframe, output_type)

    def write(self, data: Union[pd.DataFrame, List[Dict], Any]) -> str:
        """
        写入数据

        Args:
            data: 要写入的数据，支持DataFrame、字典列表或任意可序列化对象(pickle格式)

        Returns:
            str: 写入的文件路径
        """
        # 保存数据
        file_path = self._get_cache_file_path(self.operator_step + 1)
        # 确保目录路径不为空
        dir_path = os.path.dirname(file_path)
        if dir_path:  # 只有当目录路径不为空时才创建
            os.makedirs(dir_path, exist_ok=True)

        if self.cache_type == "pickle":
            # pickle格式直接保存原始数据
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            # 其他格式需要转换为DataFrame
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    dataframe = pd.DataFrame(data)
                else:
                    # 处理空列表或非字典列表的情况
                    if len(data) == 0:
                        dataframe = pd.DataFrame()  # 创建空DataFrame
                    else:
                        raise ValueError(f"不支持的数据类型: {type(data[0])}")
            elif isinstance(data, pd.DataFrame):
                dataframe = data
            elif isinstance(data, dict):
                # 处理单个字典，转换为包含一行的DataFrame
                dataframe = pd.DataFrame([data])
            else:
                raise ValueError(f"不支持的数据类型: {type(data)}")

            # 根据文件类型保存DataFrame
            if self.cache_type == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    dataframe.to_json(f, orient="records", force_ascii=False, indent=2)
            elif self.cache_type == "jsonl":
                with open(file_path, 'w', encoding='utf-8') as f:
                    dataframe.to_json(f, orient="records", lines=True, force_ascii=False)
            elif self.cache_type == "csv":
                dataframe.to_csv(file_path, index=False, encoding='utf-8')
            elif self.cache_type == "xlsx":
                dataframe.to_excel(file_path, index=False)
            elif self.cache_type == "parquet":
                dataframe.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"不支持的文件类型: {self.cache_type}")

        return file_path

    def _load_file(self, file_path: str, file_type: str) -> pd.DataFrame:
        """
        根据文件类型加载本地文件

        Args:
            file_path: 文件路径
            file_type: 文件类型

        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            if file_type == "json":
                return pd.read_json(file_path, encoding='utf-8')
            elif file_type == "jsonl":
                return pd.read_json(file_path, lines=True, encoding='utf-8')
            elif file_type == "csv":
                # 处理空CSV文件的情况
                try:
                    return pd.read_csv(file_path, encoding='utf-8')
                except pd.errors.EmptyDataError:
                    return pd.DataFrame()  # 返回空DataFrame
            elif file_type in ["xlsx", "xls"]:
                return pd.read_excel(file_path)
            elif file_type == "parquet":
                return pd.read_parquet(file_path)
            elif file_type == "pickle":
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                # 如果是DataFrame直接返回，否则尝试转换为DataFrame
                if isinstance(data, pd.DataFrame):
                    return data
                elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    return pd.DataFrame(data)
                else:
                    # 创建一个包含原始数据的单列DataFrame
                    return pd.DataFrame({"data": [data]})
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
        except Exception as e:
            raise ValueError(f"加载{file_type}文件失败: {str(e)}")

    def _convert_output(self, dataframe: pd.DataFrame, output_type: str) -> Any:
        """
        将DataFrame转换为请求的输出类型

        Args:
            dataframe: 要转换的DataFrame
            output_type: 输出类型

        Returns:
            Any: 转换后的数据
        """
        if output_type == "dataframe":
            return dataframe
        elif output_type == "dict":
            return dataframe.to_dict(orient="records")
        else:
            raise ValueError(f"不支持的输出类型: {output_type}")

    def _get_cache_file_path(self, step: int) -> str:
        """
        获取缓存文件路径

        Args:
            step: 步骤编号

        Returns:
            str: 缓存文件路径
        """
        if step == 0:
            # 如果是第一步，使用输入文件
            return self.input_file_path
        else:
            return os.path.join(self.cache_path, f"{self.file_name_prefix}_{step}.{self.cache_type}")


# 安全加载JSON数据
def safe_json_loads(x):
    """
    安全加载JSON数据

    Args:
        x: 要加载的数据

    Returns:
        Any: 加载后的数据
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x  # 保留原始字符串
    if pd.isna(x):
        return None
    return x  # 其它类型原样返回


# AI-enhanced FileStorage methods
def _add_ai_methods_to_filestorage():
    """Add AI methods to FileStorage class."""

    def add_vectors(
        self,
        vectors: Union[np.ndarray, List[List[float]]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add vectors to the vector database."""
        if self.vector_db is None:
            raise RuntimeError("Vector storage not enabled. Initialize with enable_vector_storage=True")

        return self.vector_db.add_vectors(vectors, metadata, ids)

    def search_vectors(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if self.vector_db is None:
            raise RuntimeError("Vector storage not enabled. Initialize with enable_vector_storage=True")

        return self.vector_db.search(query_vector, top_k, similarity_threshold)

    def cache_model_output(
        self,
        model_name: str,
        input_data: Any,
        output_data: Any,
        ttl: Optional[int] = None
    ) -> str:
        """Cache model output."""
        if self.model_cache is None:
            raise RuntimeError("Model caching not enabled. Initialize with enable_model_cache=True")

        return self.model_cache.cache_model_output(model_name, input_data, output_data, ttl)

    def get_cached_model_output(
        self,
        model_name: str,
        input_data: Any
    ) -> Optional[Any]:
        """Get cached model output."""
        if self.model_cache is None:
            raise RuntimeError("Model caching not enabled. Initialize with enable_model_cache=True")

        return self.model_cache.get_model_output(model_name, input_data)

    def get_vector_stats(self) -> Dict[str, Any]:
        """Get vector database statistics."""
        if self.vector_db is None:
            return {"vector_storage_enabled": False}

        return {
            "vector_storage_enabled": True,
            "vector_count": len(self.vector_db.vectors) if hasattr(self.vector_db, 'vectors') else 0,
            "dimension": self.vector_db.dimension if hasattr(self.vector_db, 'dimension') else None
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get model cache statistics."""
        if self.model_cache is None:
            return {"model_cache_enabled": False}

        stats = self.model_cache.get_stats()
        stats["model_cache_enabled"] = True
        return stats

    # Add methods to FileStorage class
    FileStorage.add_vectors = add_vectors
    FileStorage.search_vectors = search_vectors
    FileStorage.cache_model_output = cache_model_output
    FileStorage.get_cached_model_output = get_cached_model_output
    FileStorage.get_vector_stats = get_vector_stats
    FileStorage.get_cache_stats = get_cache_stats


# Apply AI methods to FileStorage
_add_ai_methods_to_filestorage()
