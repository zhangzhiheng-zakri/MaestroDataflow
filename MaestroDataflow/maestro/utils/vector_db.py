# -*- coding: utf-8 -*-
"""
Vector database support for MaestroDataflow.
Provides vector storage and similarity search capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import logging

if TYPE_CHECKING:
    from maestro.utils.storage import MaestroStorage


class VectorDatabaseABC(ABC):
    """向量数据库抽象基类。"""

    @abstractmethod
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        添加向量到数据库。

        Args:
            vectors: 向量数组
            metadata: 元数据列表
            ids: 向量ID列表

        Returns:
            List[str]: 添加的向量ID列表
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量。

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            metadata_filter: 元数据过滤条件

        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        pass

    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> bool:
        """
        删除向量。

        Args:
            ids: 要删除的向量ID列表

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    def get_vector_count(self) -> int:
        """
        获取向量数量。

        Returns:
            int: 向量数量
        """
        pass


class InMemoryVectorDB(VectorDatabaseABC):
    """
    内存向量数据库实现。
    适用于小规模数据和快速原型开发。
    """

    def __init__(self, similarity_metric: str = "cosine"):
        """
        初始化内存向量数据库。

        Args:
            similarity_metric: 相似度度量方式（"cosine", "euclidean", "dot"）
        """
        self.similarity_metric = similarity_metric
        self.vectors = []
        self.metadata = []
        self.ids = []
        self.logger = logging.getLogger(__name__)

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """添加向量到内存数据库。"""
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array")

        num_vectors = vectors.shape[0]

        # 生成ID
        if ids is None:
            ids = [f"vec_{len(self.vectors) + i}" for i in range(num_vectors)]
        elif len(ids) != num_vectors:
            raise ValueError("Number of IDs must match number of vectors")

        # 处理元数据
        if metadata is None:
            metadata = [{} for _ in range(num_vectors)]
        elif len(metadata) != num_vectors:
            raise ValueError("Number of metadata entries must match number of vectors")

        # 添加到存储
        for i in range(num_vectors):
            self.vectors.append(vectors[i])
            self.metadata.append(metadata[i])
            self.ids.append(ids[i])

        self.logger.info(f"Added {num_vectors} vectors to in-memory database")
        return ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """在内存数据库中搜索相似向量。"""
        if len(self.vectors) == 0:
            return []

        if query_vector.ndim != 1:
            raise ValueError("Query vector must be 1D")

        # 转换为numpy数组
        vectors_array = np.array(self.vectors)

        # 计算相似度
        similarities = self._calculate_similarities(query_vector, vectors_array)

        # 应用元数据过滤
        valid_indices = list(range(len(similarities)))
        if metadata_filter:
            valid_indices = []
            for i, meta in enumerate(self.metadata):
                if self._match_metadata_filter(meta, metadata_filter):
                    valid_indices.append(i)

        if not valid_indices:
            return []

        # 过滤相似度
        filtered_similarities = [(i, similarities[i]) for i in valid_indices]

        # 应用相似度阈值
        if similarity_threshold is not None:
            filtered_similarities = [
                (i, sim) for i, sim in filtered_similarities
                if sim >= similarity_threshold
            ]

        # 排序并取top-k
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = filtered_similarities[:top_k]

        # 构建结果
        results = []
        for idx, similarity in top_results:
            result = {
                "id": self.ids[idx],
                "similarity": float(similarity),
                "metadata": self.metadata[idx].copy(),
                "vector": self.vectors[idx].tolist()
            }
            results.append(result)

        return results

    def delete_vectors(self, ids: List[str]) -> bool:
        """从内存数据库中删除向量。"""
        try:
            indices_to_remove = []
            for target_id in ids:
                for i, vec_id in enumerate(self.ids):
                    if vec_id == target_id:
                        indices_to_remove.append(i)
                        break

            # 按降序删除，避免索引变化
            for idx in sorted(indices_to_remove, reverse=True):
                del self.vectors[idx]
                del self.metadata[idx]
                del self.ids[idx]

            self.logger.info(f"Deleted {len(indices_to_remove)} vectors from in-memory database")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete vectors: {e}")
            return False

    def get_vector_count(self) -> int:
        """获取向量数量。"""
        return len(self.vectors)

    def _calculate_similarities(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """计算相似度。"""
        if self.similarity_metric == "cosine":
            # 余弦相似度
            query_norm = query_vector / np.linalg.norm(query_vector)
            vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            similarities = np.dot(vectors_norm, query_norm)

        elif self.similarity_metric == "dot":
            # 点积相似度
            similarities = np.dot(vectors, query_vector)

        elif self.similarity_metric == "euclidean":
            # 欧几里得距离（转换为相似度）
            distances = np.linalg.norm(vectors - query_vector, axis=1)
            similarities = 1 / (1 + distances)

        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

        return similarities

    def _match_metadata_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """检查元数据是否匹配过滤条件。"""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True


class FileVectorDB(VectorDatabaseABC):
    """
    基于文件的向量数据库实现。
    将向量和元数据存储在本地文件中。
    """

    def __init__(
        self,
        storage_path: str,
        similarity_metric: str = "cosine",
        auto_save: bool = True
    ):
        """
        初始化文件向量数据库。

        Args:
            storage_path: 存储路径
            similarity_metric: 相似度度量方式
            auto_save: 是否自动保存
        """
        self.storage_path = storage_path
        self.similarity_metric = similarity_metric
        self.auto_save = auto_save
        self.logger = logging.getLogger(__name__)

        # 确保存储目录存在
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        # 内存缓存
        self._memory_db = InMemoryVectorDB(similarity_metric)

        # 加载现有数据
        self._load_from_file()

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """添加向量到文件数据库。"""
        result_ids = self._memory_db.add_vectors(vectors, metadata, ids)

        if self.auto_save:
            self._save_to_file()

        return result_ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """在文件数据库中搜索相似向量。"""
        return self._memory_db.search(query_vector, top_k, similarity_threshold, metadata_filter)

    def delete_vectors(self, ids: List[str]) -> bool:
        """从文件数据库中删除向量。"""
        result = self._memory_db.delete_vectors(ids)

        if result and self.auto_save:
            self._save_to_file()

        return result

    def get_vector_count(self) -> int:
        """获取向量数量。"""
        return self._memory_db.get_vector_count()

    def save(self):
        """手动保存到文件。"""
        self._save_to_file()

    def _save_to_file(self):
        """保存数据到文件。"""
        try:
            data = {
                "vectors": [vec.tolist() for vec in self._memory_db.vectors],
                "metadata": self._memory_db.metadata,
                "ids": self._memory_db.ids,
                "similarity_metric": self.similarity_metric,
                "created_at": datetime.now().isoformat(),
                "vector_count": len(self._memory_db.vectors)
            }

            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Saved {len(self._memory_db.vectors)} vectors to {self.storage_path}")

        except Exception as e:
            self.logger.error(f"Failed to save vector database: {e}")
            raise

    def _load_from_file(self):
        """从文件加载数据。"""
        if not os.path.exists(self.storage_path):
            self.logger.info(f"Vector database file not found: {self.storage_path}")
            return

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 恢复数据
            if data.get("vectors"):
                vectors = np.array(data["vectors"])
                metadata = data.get("metadata", [])
                ids = data.get("ids", [])

                self._memory_db.vectors = [vectors[i] for i in range(len(vectors))]
                self._memory_db.metadata = metadata
                self._memory_db.ids = ids

            self.logger.info(f"Loaded {len(self._memory_db.vectors)} vectors from {self.storage_path}")

        except Exception as e:
            self.logger.error(f"Failed to load vector database: {e}")
            # 继续使用空数据库


class VectorStorage:
    """
    向量存储系统。
    提供向量数据存储和检索功能。
    """

    def __init__(
        self,
        input_file_path: str,
        cache_path: str = "./cache",
        cache_type: str = "json",
        file_name_prefix: str = "vector_data",
        vector_db_type: str = "memory",
        vector_db_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化向量存储系统。

        Args:
            input_file_path: 输入文件路径
            cache_path: 缓存路径
            cache_type: 缓存文件类型
            file_name_prefix: 文件名前缀
            vector_db_type: 向量数据库类型（"memory", "file"）
            vector_db_config: 向量数据库配置
        """
      
        self.input_file_path = input_file_path
        self.cache_path = cache_path
        self.cache_type = cache_type
        self.file_name_prefix = file_name_prefix

        self.vector_db_type = vector_db_type
        self.vector_db_config = vector_db_config or {}
        self._vector_db = None

        self.logger = logging.getLogger(__name__)

    def get_vector_db(self) -> VectorDatabaseABC:
        """获取向量数据库实例。"""
        if self._vector_db is None:
            if self.vector_db_type == "memory":
                self._vector_db = InMemoryVectorDB(**self.vector_db_config)
            elif self.vector_db_type == "file":
                # 设置默认存储路径
                if "storage_path" not in self.vector_db_config:
                    self.vector_db_config["storage_path"] = os.path.join(
                        self.cache_path, f"{self.file_name_prefix}_vectors.json"
                    )
                self._vector_db = FileVectorDB(**self.vector_db_config)
            else:
                raise ValueError(f"Unsupported vector database type: {self.vector_db_type}")

        return self._vector_db

    def add_vectors_from_dataframe(
        self,
        df: pd.DataFrame,
        vector_column: str = "embedding",
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None
    ) -> List[str]:
        """
        从DataFrame添加向量到向量数据库。

        Args:
            df: 包含向量数据的DataFrame
            vector_column: 向量列名
            id_column: ID列名
            metadata_columns: 元数据列名列表

        Returns:
            List[str]: 添加的向量ID列表
        """
        if vector_column not in df.columns:
            raise ValueError(f"Vector column '{vector_column}' not found in DataFrame")

        # 提取向量
        vectors = np.array(df[vector_column].tolist())

        # 提取ID
        ids = None
        if id_column and id_column in df.columns:
            ids = df[id_column].astype(str).tolist()

        # 提取元数据
        metadata = None
        if metadata_columns:
            available_columns = [col for col in metadata_columns if col in df.columns]
            if available_columns:
                metadata = df[available_columns].to_dict(orient="records")

        # 添加到向量数据库
        vector_db = self.get_vector_db()
        return vector_db.add_vectors(vectors, metadata, ids)

    def search_vectors(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量。

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            metadata_filter: 元数据过滤条件

        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector)

        vector_db = self.get_vector_db()
        return vector_db.search(query_vector, top_k, similarity_threshold, metadata_filter)

    def get_vector_stats(self) -> Dict[str, Any]:
        """
        获取向量数据库统计信息。

        Returns:
            Dict[str, Any]: 统计信息
        """
        vector_db = self.get_vector_db()
        return {
            "vector_count": vector_db.get_vector_count(),
            "vector_db_type": self.vector_db_type,
            "similarity_metric": getattr(vector_db, "similarity_metric", "unknown")
        }