"""
Embedding and vectorization AI operators for MaestroDataflow.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

from maestro.core.operator import OperatorABC
from maestro.utils.storage import MaestroStorage


class EmbeddingGenerator(OperatorABC):
    """
    文本向量化操作符。
    支持多种嵌入模型生成文本向量。
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        input_column: str = "text",
        output_column: str = "embedding",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: str = "auto"
    ):
        """
        初始化向量化操作符。

        Args:
            embedding_model: 嵌入模型名称或路径
            input_column: 输入文本列名
            output_column: 输出向量列名
            batch_size: 批处理大小
            normalize_embeddings: 是否标准化向量
            device: 计算设备（"cpu", "cuda", "auto"）
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.input_column = input_column
        self.output_column = output_column
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device

        self._model = None
        self._model_loaded = False

    def _load_model(self):
        """延迟加载嵌入模型。"""
        if self._model_loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self.logger.info(f"Loading embedding model: {self.embedding_model}")
            self._model = SentenceTransformer(self.embedding_model, device=self.device)
            self._model_loaded = True

            self.logger.info(f"Model loaded successfully on device: {self._model.device}")

        except ImportError:
            raise ImportError(
                "Please install sentence-transformers: "
                "pip install sentence-transformers"
            )
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise

    def run(self, storage: Union[MaestroStorage, 'FileStorage'], **kwargs) -> Dict[str, Any]:
        """
        执行文本向量化操作。

        Args:
            storage: 存储对象
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 操作结果
        """
        self.log_operation_start(**kwargs)

        try:
            # 加载模型
            self._load_model()

            # 读取数据
            if hasattr(storage, 'operator_step') and storage.operator_step == -1:
                storage = storage.step()

            df = storage.read(output_type="dataframe")

            if self.input_column not in df.columns:
                raise ValueError(f"Input column '{self.input_column}' not found in data")

            # 获取文本数据
            texts = df[self.input_column].astype(str).tolist()

            # 生成向量
            self.logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=True
            )

            # 转换为列表格式（便于存储）
            embeddings_list = embeddings.tolist()

            # 添加向量列
            df[self.output_column] = embeddings_list

            # 写入结果
            path = storage.write(df)

            result = {
                "path": path,
                "embedded_count": len(embeddings_list),
                "embedding_dimension": embeddings.shape[1],
                "input_column": self.input_column,
                "output_column": self.output_column,
                "model_name": self.embedding_model
            }

            self.log_operation_end(result)
            return result

        except Exception as e:
            self.handle_error(e, "embedding generation")


class SimilarityCalculator(OperatorABC):
    """
    相似度计算操作符。
    计算文本或向量之间的相似度。
    """

    def __init__(
        self,
        embedding_column: str = "embedding",
        reference_embeddings: Optional[List[List[float]]] = None,
        reference_texts: Optional[List[str]] = None,
        similarity_metric: str = "cosine",
        output_column: str = "similarity_scores",
        top_k: Optional[int] = None,
        embedding_model: Optional[str] = None
    ):
        """
        初始化相似度计算操作符。

        Args:
            embedding_column: 向量列名
            reference_embeddings: 参考向量列表
            reference_texts: 参考文本列表（如果提供，会自动生成向量）
            similarity_metric: 相似度度量（"cosine", "euclidean", "dot"）
            output_column: 输出相似度列名
            top_k: 返回前k个最相似的结果
            embedding_model: 嵌入模型（用于处理reference_texts）
        """
        super().__init__()
        self.embedding_column = embedding_column
        self.reference_embeddings = reference_embeddings
        self.reference_texts = reference_texts
        self.similarity_metric = similarity_metric
        self.output_column = output_column
        self.top_k = top_k
        self.embedding_model = embedding_model

        self._model = None

        # 验证参数
        if reference_embeddings is None and reference_texts is None:
            raise ValueError("Either reference_embeddings or reference_texts must be provided")

        if reference_texts is not None and embedding_model is None:
            raise ValueError("embedding_model must be provided when using reference_texts")

    def _load_embedding_model(self):
        """加载嵌入模型（如果需要）。"""
        if self._model is not None or self.embedding_model is None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model)
            self.logger.info(f"Loaded embedding model: {self.embedding_model}")
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def _prepare_reference_embeddings(self) -> np.ndarray:
        """准备参考向量。"""
        if self.reference_embeddings is not None:
            return np.array(self.reference_embeddings)

        if self.reference_texts is not None:
            self._load_embedding_model()
            self.logger.info(f"Generating embeddings for {len(self.reference_texts)} reference texts")
            embeddings = self._model.encode(self.reference_texts)
            return embeddings

        raise ValueError("No reference embeddings available")

    def _calculate_similarity(self, embeddings: np.ndarray, reference_embeddings: np.ndarray) -> np.ndarray:
        """
        计算相似度。

        Args:
            embeddings: 查询向量
            reference_embeddings: 参考向量

        Returns:
            np.ndarray: 相似度矩阵
        """
        if self.similarity_metric == "cosine":
            # 余弦相似度
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            reference_norm = reference_embeddings / np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
            similarities = np.dot(embeddings_norm, reference_norm.T)

        elif self.similarity_metric == "dot":
            # 点积相似度
            similarities = np.dot(embeddings, reference_embeddings.T)

        elif self.similarity_metric == "euclidean":
            # 欧几里得距离（转换为相似度）
            from scipy.spatial.distance import cdist
            distances = cdist(embeddings, reference_embeddings, metric='euclidean')
            similarities = 1 / (1 + distances)  # 转换为相似度

        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

        return similarities

    def run(self, storage: Union[MaestroStorage, 'FileStorage'], **kwargs) -> Dict[str, Any]:
        """
        执行相似度计算操作。

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

            if self.embedding_column not in df.columns:
                raise ValueError(f"Embedding column '{self.embedding_column}' not found in data")

            # 获取查询向量
            query_embeddings = np.array(df[self.embedding_column].tolist())

            # 准备参考向量
            reference_embeddings = self._prepare_reference_embeddings()

            # 计算相似度
            self.logger.info(f"Calculating similarities for {len(query_embeddings)} queries against {len(reference_embeddings)} references")
            similarities = self._calculate_similarity(query_embeddings, reference_embeddings)

            # 处理结果
            if self.top_k is not None:
                # 返回top-k结果
                top_indices = np.argsort(similarities, axis=1)[:, -self.top_k:][:, ::-1]
                top_scores = np.take_along_axis(similarities, top_indices, axis=1)

                similarity_results = []
                for i in range(len(similarities)):
                    result = {
                        "indices": top_indices[i].tolist(),
                        "scores": top_scores[i].tolist()
                    }
                    if self.reference_texts is not None:
                        result["texts"] = [self.reference_texts[idx] for idx in top_indices[i]]
                    similarity_results.append(result)
            else:
                # 返回所有相似度分数
                similarity_results = similarities.tolist()

            # 添加相似度列
            df[self.output_column] = similarity_results

            # 写入结果
            path = storage.write(df)

            result = {
                "path": path,
                "calculated_count": len(similarities),
                "reference_count": len(reference_embeddings),
                "similarity_metric": self.similarity_metric,
                "embedding_column": self.embedding_column,
                "output_column": self.output_column
            }

            if self.top_k is not None:
                result["top_k"] = self.top_k

            self.log_operation_end(result)
            return result

        except Exception as e:
            self.handle_error(e, "similarity calculation")


class TextMatcher(OperatorABC):
    """
    文本匹配操作符。
    基于语义相似度进行文本匹配。
    """

    def __init__(
        self,
        input_column: str,
        reference_texts: List[str],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        output_column: str = "matched_text",
        score_column: str = "match_score",
        include_all_scores: bool = False
    ):
        """
        初始化文本匹配操作符。

        Args:
            input_column: 输入文本列名
            reference_texts: 参考文本列表
            embedding_model: 嵌入模型名称
            similarity_threshold: 相似度阈值
            output_column: 输出匹配文本列名
            score_column: 输出匹配分数列名
            include_all_scores: 是否包含所有相似度分数
        """
        super().__init__()
        self.input_column = input_column
        self.reference_texts = reference_texts
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.output_column = output_column
        self.score_column = score_column
        self.include_all_scores = include_all_scores

        self._model = None
        self._reference_embeddings = None

    def _load_model_and_embeddings(self):
        """加载模型并生成参考文本的向量。"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self.logger.info(f"Loading embedding model: {self.embedding_model}")
            self._model = SentenceTransformer(self.embedding_model)

            self.logger.info(f"Generating embeddings for {len(self.reference_texts)} reference texts")
            self._reference_embeddings = self._model.encode(self.reference_texts)

        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def run(self, storage: Union[MaestroStorage, 'FileStorage'], **kwargs) -> Dict[str, Any]:
        """
        执行文本匹配操作。

        Args:
            storage: 存储对象
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 操作结果
        """
        self.log_operation_start(**kwargs)

        try:
            # 加载模型和参考向量
            self._load_model_and_embeddings()

            # 读取数据
            if hasattr(storage, 'operator_step') and storage.operator_step == -1:
                storage = storage.step()

            df = storage.read(output_type="dataframe")

            if self.input_column not in df.columns:
                raise ValueError(f"Input column '{self.input_column}' not found in data")

            # 获取输入文本并生成向量
            input_texts = df[self.input_column].astype(str).tolist()
            input_embeddings = self._model.encode(input_texts)

            # 计算相似度
            similarities = np.dot(input_embeddings, self._reference_embeddings.T)

            # 找到最佳匹配
            matched_texts = []
            match_scores = []
            all_scores = []

            for i, sim_row in enumerate(similarities):
                best_idx = np.argmax(sim_row)
                best_score = sim_row[best_idx]

                if best_score >= self.similarity_threshold:
                    matched_texts.append(self.reference_texts[best_idx])
                    match_scores.append(float(best_score))
                else:
                    matched_texts.append(None)
                    match_scores.append(0.0)

                if self.include_all_scores:
                    all_scores.append(sim_row.tolist())

            # 添加结果列
            df[self.output_column] = matched_texts
            df[self.score_column] = match_scores

            if self.include_all_scores:
                df[f"{self.score_column}_all"] = all_scores

            # 写入结果
            path = storage.write(df)

            # 统计匹配结果
            matched_count = sum(1 for text in matched_texts if text is not None)

            result = {
                "path": path,
                "total_count": len(input_texts),
                "matched_count": matched_count,
                "match_rate": matched_count / len(input_texts) if input_texts else 0,
                "similarity_threshold": self.similarity_threshold,
                "input_column": self.input_column,
                "output_column": self.output_column,
                "score_column": self.score_column
            }

            self.log_operation_end(result)
            return result

        except Exception as e:
            self.handle_error(e, "text matching")