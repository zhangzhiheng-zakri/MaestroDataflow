"""
RAG (Retrieval-Augmented Generation) AI operators for MaestroDataflow.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from maestro.core.operator import OperatorABC
from maestro.core.prompt import PromptABC, StandardPrompt, create_diy_prompt
from maestro.serving.llm_serving import LLMServingABC
from maestro.utils.storage import MaestroStorage


class KnowledgeBaseBuilder(OperatorABC):
    """
    知识库构建操作符。
    将文档数据转换为可检索的知识库。
    """

    def __init__(
        self,
        text_column: str = "text",
        title_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_format: str = "dataframe"  # "dataframe" or "vector_db"
    ):
        """
        初始化知识库构建操作符。

        Args:
            text_column: 文本内容列名
            title_column: 标题列名（可选）
            metadata_columns: 元数据列名列表
            chunk_size: 文本分块大小
            chunk_overlap: 分块重叠大小
            embedding_model: 嵌入模型名称
            output_format: 输出格式
        """
        super().__init__()
        self.text_column = text_column
        self.title_column = title_column
        self.metadata_columns = metadata_columns or []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.output_format = output_format

        self._model = None

    def _load_embedding_model(self):
        """加载嵌入模型。"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self.logger.info(f"Loading embedding model: {self.embedding_model}")
            self._model = SentenceTransformer(self.embedding_model)
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def _chunk_text(self, text: str, title: Optional[str] = None) -> List[str]:
        """
        将文本分块。

        Args:
            text: 原始文本
            title: 文档标题

        Returns:
            List[str]: 文本块列表
        """
        if not text or len(text.strip()) == 0:
            return []

        # 简单的文本分块策略
        chunks = []
        text = text.strip()

        # 如果文本长度小于chunk_size，直接返回
        if len(text) <= self.chunk_size:
            chunk = f"{title}: {text}" if title else text
            return [chunk]

        # 按句子分割
        sentences = text.replace('\n', ' ').split('. ')

        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 检查添加当前句子是否会超过chunk_size
            potential_chunk = current_chunk + ". " + sentence if current_chunk else sentence

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # 保存当前块
                if current_chunk:
                    chunk_text = f"{title}: {current_chunk}" if title else current_chunk
                    chunks.append(chunk_text)

                # 开始新块（考虑重叠）
                if self.chunk_overlap > 0 and current_chunk:
                    # 保留最后几个字符作为重叠
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + ". " + sentence
                else:
                    current_chunk = sentence

        # 添加最后一个块
        if current_chunk:
            chunk_text = f"{title}: {current_chunk}" if title else current_chunk
            chunks.append(chunk_text)

        return chunks

    def run(self, storage: Union[MaestroStorage, 'FileStorage'], **kwargs) -> Dict[str, Any]:
        """
        执行知识库构建操作。

        Args:
            storage: 存储对象
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 操作结果
        """
        self.log_operation_start(**kwargs)

        try:
            # 加载嵌入模型
            self._load_embedding_model()

            # 读取数据
            if hasattr(storage, 'operator_step') and storage.operator_step == -1:
                storage = storage.step()

            df = storage.read(output_type="dataframe")

            if self.text_column not in df.columns:
                raise ValueError(f"Text column '{self.text_column}' not found in data")

            # 构建知识库
            knowledge_chunks = []

            for idx, row in df.iterrows():
                text = str(row[self.text_column])
                title = str(row[self.title_column]) if self.title_column and self.title_column in df.columns else None

                # 分块
                chunks = self._chunk_text(text, title)

                # 为每个块创建记录
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_record = {
                        "chunk_id": f"{idx}_{chunk_idx}",
                        "source_id": idx,
                        "chunk_text": chunk,
                        "chunk_index": chunk_idx,
                        "created_at": datetime.now().isoformat()
                    }

                    # 添加元数据
                    for meta_col in self.metadata_columns:
                        if meta_col in df.columns:
                            chunk_record[f"meta_{meta_col}"] = row[meta_col]

                    knowledge_chunks.append(chunk_record)

            # 创建知识库DataFrame
            kb_df = pd.DataFrame(knowledge_chunks)

            # 生成向量
            if len(knowledge_chunks) > 0:
                self.logger.info(f"Generating embeddings for {len(knowledge_chunks)} chunks")
                chunk_texts = [chunk["chunk_text"] for chunk in knowledge_chunks]
                embeddings = self._model.encode(chunk_texts, show_progress_bar=True)
                kb_df["embedding"] = embeddings.tolist()

            # 写入结果
            path = storage.write(kb_df)

            result = {
                "path": path,
                "total_documents": len(df),
                "total_chunks": len(knowledge_chunks),
                "avg_chunks_per_doc": len(knowledge_chunks) / len(df) if len(df) > 0 else 0,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": self.embedding_model,
                "text_column": self.text_column
            }

            self.log_operation_end(result)
            return result

        except Exception as e:
            self.handle_error(e, "knowledge base building")


class RAGRetriever(OperatorABC):
    """
    RAG检索操作符。
    从知识库中检索相关文档。
    """

    def __init__(
        self,
        query_column: str = "query",
        knowledge_base_path: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        output_column: str = "retrieved_docs",
        score_column: str = "retrieval_scores"
    ):
        """
        初始化RAG检索操作符。

        Args:
            query_column: 查询文本列名
            knowledge_base_path: 知识库文件路径
            embedding_model: 嵌入模型名称
            top_k: 检索的文档数量
            similarity_threshold: 相似度阈值
            output_column: 输出检索文档列名
            score_column: 输出检索分数列名
        """
        super().__init__()
        self.query_column = query_column
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.output_column = output_column
        self.score_column = score_column

        self._model = None
        self._knowledge_base = None
        self._kb_embeddings = None

    def _load_embedding_model(self):
        """加载嵌入模型。"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self.logger.info(f"Loading embedding model: {self.embedding_model}")
            self._model = SentenceTransformer(self.embedding_model)
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def _load_knowledge_base(self, storage: Union[MaestroStorage, 'FileStorage']):
        """加载知识库。"""
        if self._knowledge_base is not None:
            return

        if self.knowledge_base_path:
            # 从指定路径加载
            import pandas as pd
            self._knowledge_base = pd.read_csv(self.knowledge_base_path)  # 简化实现
        else:
            # 从上一步操作结果加载
            self._knowledge_base = storage.read(output_type="dataframe")

        # 提取嵌入向量
        if "embedding" in self._knowledge_base.columns:
            embeddings_list = self._knowledge_base["embedding"].tolist()
            self._kb_embeddings = np.array(embeddings_list)
        else:
            raise ValueError("Knowledge base must contain 'embedding' column")

        self.logger.info(f"Loaded knowledge base with {len(self._knowledge_base)} chunks")

    def _retrieve_documents(self, query_embedding: np.ndarray) -> Tuple[List[Dict], List[float]]:
        """
        检索相关文档。

        Args:
            query_embedding: 查询向量

        Returns:
            Tuple[List[Dict], List[float]]: 检索到的文档和分数
        """
        # 计算相似度
        similarities = np.dot(query_embedding, self._kb_embeddings.T)

        # 获取top-k结果
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        top_scores = similarities[top_indices]

        # 过滤低于阈值的结果
        valid_indices = top_scores >= self.similarity_threshold
        top_indices = top_indices[valid_indices]
        top_scores = top_scores[valid_indices]

        # 构建结果
        retrieved_docs = []
        for idx in top_indices:
            doc = self._knowledge_base.iloc[idx].to_dict()
            retrieved_docs.append(doc)

        return retrieved_docs, top_scores.tolist()

    def run(self, storage: Union[MaestroStorage, 'FileStorage'], **kwargs) -> Dict[str, Any]:
        """
        执行RAG检索操作。

        Args:
            storage: 存储对象
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 操作结果
        """
        self.log_operation_start(**kwargs)

        try:
            # 加载模型和知识库
            self._load_embedding_model()
            self._load_knowledge_base(storage)

            # 读取查询数据
            if hasattr(storage, 'operator_step') and storage.operator_step == -1:
                storage = storage.step()

            df = storage.read(output_type="dataframe")

            if self.query_column not in df.columns:
                raise ValueError(f"Query column '{self.query_column}' not found in data")

            # 处理查询
            queries = df[self.query_column].astype(str).tolist()
            query_embeddings = self._model.encode(queries)

            retrieved_docs_list = []
            retrieval_scores_list = []

            for query_embedding in query_embeddings:
                docs, scores = self._retrieve_documents(query_embedding)
                retrieved_docs_list.append(docs)
                retrieval_scores_list.append(scores)

            # 添加结果列
            df[self.output_column] = retrieved_docs_list
            df[self.score_column] = retrieval_scores_list

            # 写入结果
            path = storage.write(df)

            # 统计信息
            total_retrieved = sum(len(docs) for docs in retrieved_docs_list)
            avg_retrieved = total_retrieved / len(queries) if queries else 0

            result = {
                "path": path,
                "total_queries": len(queries),
                "total_retrieved": total_retrieved,
                "avg_retrieved_per_query": avg_retrieved,
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
                "query_column": self.query_column,
                "output_column": self.output_column
            }

            self.log_operation_end(result)
            return result

        except Exception as e:
            self.handle_error(e, "RAG retrieval")


class RAGOperator(OperatorABC):
    """
    完整的RAG操作符。
    结合检索和生成功能。
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        query_column: str = "query",
        knowledge_base_path: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        similarity_threshold: float = 0.3,
        system_prompt: Optional[str] = None,
        rag_prompt_template: Optional[str] = None,
        output_column: str = "rag_response",
        include_sources: bool = True,
        max_context_length: int = 2000
    ):
        """
        初始化RAG操作符。

        Args:
            llm_serving: LLM服务对象
            query_column: 查询文本列名
            knowledge_base_path: 知识库文件路径
            embedding_model: 嵌入模型名称
            top_k: 检索的文档数量
            similarity_threshold: 相似度阈值
            system_prompt: 系统提示词
            rag_prompt_template: RAG提示词模板
            output_column: 输出响应列名
            include_sources: 是否包含来源信息
            max_context_length: 最大上下文长度
        """
        super().__init__()
        self.llm_serving = llm_serving
        self.query_column = query_column
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.output_column = output_column
        self.include_sources = include_sources
        self.max_context_length = max_context_length

        # 设置提示词
        self.system_prompt = system_prompt or (
            "你是一个有用的AI助手。请基于提供的上下文信息回答用户的问题。"
            "如果上下文中没有相关信息，请诚实地说明你不知道答案。"
        )

        self.rag_prompt_template = rag_prompt_template or (
            "上下文信息：\n{context}\n\n"
            "基于以上上下文信息，请回答以下问题：\n{query}\n\n"
            "回答："
        )

        # 初始化检索器
        self.retriever = RAGRetriever(
            query_column=query_column,
            knowledge_base_path=knowledge_base_path,
            embedding_model=embedding_model,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

    def _format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        格式化检索到的文档作为上下文。

        Args:
            retrieved_docs: 检索到的文档列表

        Returns:
            str: 格式化的上下文
        """
        if not retrieved_docs:
            return "没有找到相关信息。"

        context_parts = []
        current_length = 0

        for i, doc in enumerate(retrieved_docs):
            chunk_text = doc.get("chunk_text", "")

            # 检查是否超过最大长度
            if current_length + len(chunk_text) > self.max_context_length:
                break

            context_parts.append(f"[文档{i+1}] {chunk_text}")
            current_length += len(chunk_text)

        return "\n\n".join(context_parts)

    def run(self, storage: Union[MaestroStorage, 'FileStorage'], **kwargs) -> Dict[str, Any]:
        """
        执行RAG操作。

        Args:
            storage: 存储对象
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 操作结果
        """
        self.log_operation_start(**kwargs)

        try:
            # 首先执行检索
            retrieval_result = self.retriever.run(storage, **kwargs)

            # 读取检索结果
            df = storage.read(output_type="dataframe")

            # 生成响应
            responses = []
            sources_list = []

            for idx, row in df.iterrows():
                query = str(row[self.query_column])
                retrieved_docs = row[self.retriever.output_column]

                # 格式化上下文
                context = self._format_context(retrieved_docs)

                # 构建提示词
                prompt = self.rag_prompt_template.format(
                    context=context,
                    query=query
                )

                # 生成响应
                try:
                    response = self.llm_serving.generate(
                        prompt=prompt,
                        system_prompt=self.system_prompt
                    )
                    responses.append(response)

                    # 收集来源信息
                    if self.include_sources:
                        sources = []
                        for doc in retrieved_docs:
                            source_info = {
                                "chunk_id": doc.get("chunk_id"),
                                "source_id": doc.get("source_id"),
                                "chunk_text": doc.get("chunk_text", "")[:200] + "..."  # 截断显示
                            }
                            sources.append(source_info)
                        sources_list.append(sources)

                except Exception as e:
                    self.logger.error(f"Failed to generate response for query {idx}: {e}")
                    responses.append(f"生成响应时出错: {str(e)}")
                    if self.include_sources:
                        sources_list.append([])

            # 添加响应列
            df[self.output_column] = responses

            if self.include_sources:
                df[f"{self.output_column}_sources"] = sources_list

            # 写入结果
            path = storage.write(df)

            result = {
                "path": path,
                "total_queries": len(responses),
                "successful_responses": len([r for r in responses if not r.startswith("生成响应时出错")]),
                "retrieval_stats": {
                    "top_k": self.top_k,
                    "similarity_threshold": self.similarity_threshold,
                    "total_retrieved": retrieval_result.get("total_retrieved", 0)
                },
                "query_column": self.query_column,
                "output_column": self.output_column,
                "include_sources": self.include_sources
            }

            self.log_operation_end(result)
            return result

        except Exception as e:
            self.handle_error(e, "RAG operation")