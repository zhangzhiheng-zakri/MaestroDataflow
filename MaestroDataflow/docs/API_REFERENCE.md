# MaestroDataflow AI操作符 API参考

## 概述

本文档提供MaestroDataflow AI操作符生态系统的完整API参考，包括所有类、方法、参数和返回值的详细说明。

## 目录

1. [基础类](#基础类)
2. [文本生成操作符](#文本生成操作符)
3. [嵌入向量操作符](#嵌入向量操作符)
4. [RAG操作符](#rag操作符)
5. [多模态操作符](#多模态操作符)
6. [智能数据处理操作符](#智能数据处理操作符)
7. [存储增强](#存储增强)
8. [模型缓存](#模型缓存)
9. [向量数据库](#向量数据库)

## 基础类

### OperatorABC

所有AI操作符的抽象基类。

```python
class OperatorABC(ABC):
    """AI操作符抽象基类"""
    
    ALLOWED_PROMPTS: Tuple[Type[PromptABC], ...] = ()
    
    def __init__(self):
        """初始化操作符"""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """
        执行操作符
        
        Args:
            storage: 存储实例
            **kwargs: 操作参数
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        pass
    
    def log_operation_start(self, operation: str, params: Dict[str, Any]) -> None:
        """记录操作开始"""
        
    def log_operation_end(self, operation: str, result: Dict[str, Any]) -> None:
        """记录操作结束"""
        
    def handle_error(self, operation: str, error: Exception) -> Dict[str, Any]:
        """处理错误"""
```

## 文本生成操作符

### PromptedGenerator

基于提示词的文本生成器。

```python
class PromptedGenerator(OperatorABC):
    """提示词驱动的文本生成器"""
    
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
        初始化生成器
        
        Args:
            llm_serving: LLM服务实例
            prompt: 提示词对象
            input_column: 输入文本列名
            output_column: 输出文本列名
            batch_size: 批处理大小
            **generation_kwargs: 传递给LLM的额外参数
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行文本生成
        
        Args:
            storage: 存储实例
            input_path: 输入数据路径
            output_path: 输出数据路径
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 包含以下键的结果字典
                - status: "success" 或 "error"
                - output_path: 输出路径
                - processed_count: 处理的记录数
                - execution_time: 执行时间（秒）
                - error: 错误信息（如果有）
        """
```

### TextSummarizer

文本摘要生成器。

```python
class TextSummarizer(OperatorABC):
    """文本摘要生成器"""
    
    def __init__(
        self,
        llm_serving: LLMServingABC,
        max_length: int = 150,
        summary_type: str = "abstractive",
        input_column: str = "text",
        output_column: str = "summary",
        batch_size: int = 10
    ):
        """
        初始化摘要器
        
        Args:
            llm_serving: LLM服务实例
            max_length: 摘要最大长度
            summary_type: 摘要类型 ("extractive" 或 "abstractive")
            input_column: 输入文本列名
            output_column: 输出摘要列名
            batch_size: 批处理大小
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """执行文本摘要"""
```

### TextClassifier

文本分类器。

```python
class TextClassifier(OperatorABC):
    """文本分类器"""
    
    def __init__(
        self,
        llm_serving: LLMServingABC,
        categories: List[str],
        target_column: str = "text",
        output_column: str = "category",
        include_confidence: bool = True,
        batch_size: int = 10
    ):
        """
        初始化分类器
        
        Args:
            llm_serving: LLM服务实例
            categories: 分类类别列表
            target_column: 输入文本列名
            output_column: 输出分类列名
            include_confidence: 是否包含置信度
            batch_size: 批处理大小
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """执行文本分类"""
```

## 嵌入向量操作符

### EmbeddingGenerator

文本嵌入向量生成器。

```python
class EmbeddingGenerator(OperatorABC):
    """文本嵌入向量生成器"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        input_column: str = "text",
        output_column: str = "embedding",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: str = "cpu"
    ):
        """
        初始化嵌入生成器
        
        Args:
            model_name: 嵌入模型名称
            input_column: 输入文本列名
            output_column: 输出向量列名
            batch_size: 批处理大小
            normalize_embeddings: 是否标准化嵌入向量
            device: 计算设备 ("cpu" 或 "cuda")
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成嵌入向量
        
        Returns:
            Dict[str, Any]: 包含以下键的结果字典
                - status: "success" 或 "error"
                - output_path: 输出路径
                - processed_count: 处理的记录数
                - embedding_dimension: 嵌入向量维度
                - model_name: 使用的模型名称
        """
```

### SimilarityCalculator

相似度计算器。

```python
class SimilarityCalculator(OperatorABC):
    """相似度计算器"""
    
    def __init__(
        self,
        metric: str = "cosine",
        target_columns: Optional[List[str]] = None,
        embedding_columns: Optional[List[str]] = None,
        output_column: str = "similarity",
        batch_size: int = 100
    ):
        """
        初始化相似度计算器
        
        Args:
            metric: 相似度度量 ("cosine", "dot_product", "euclidean")
            target_columns: 文本列名列表（用于计算文本相似度）
            embedding_columns: 嵌入列名列表（用于计算向量相似度）
            output_column: 输出相似度列名
            batch_size: 批处理大小
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """计算相似度"""
```

### TextMatcher

文本匹配器。

```python
class TextMatcher(OperatorABC):
    """文本匹配器"""
    
    def __init__(
        self,
        reference_texts: List[str],
        similarity_threshold: float = 0.5,
        target_column: str = "text",
        top_k: int = 5,
        return_scores: bool = True,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        初始化文本匹配器
        
        Args:
            reference_texts: 参考文本列表
            similarity_threshold: 相似度阈值
            target_column: 输入文本列名
            top_k: 返回最相似的k个结果
            return_scores: 是否返回相似度分数
            embedding_model: 嵌入模型名称
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """执行文本匹配"""
```

## RAG操作符

### KnowledgeBaseBuilder

知识库构建器。

```python
class KnowledgeBaseBuilder(OperatorABC):
    """知识库构建器"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        text_column: str = "text",
        metadata_columns: Optional[List[str]] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunking_strategy: str = "fixed_size"
    ):
        """
        初始化知识库构建器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小
            text_column: 文本列名
            metadata_columns: 元数据列名列表
            embedding_model: 嵌入模型名称
            chunking_strategy: 分块策略 ("fixed_size", "sentence", "paragraph")
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        构建知识库
        
        Returns:
            Dict[str, Any]: 包含以下键的结果字典
                - status: "success" 或 "error"
                - output_path: 输出路径
                - total_chunks: 总文本块数
                - total_documents: 总文档数
                - embedding_dimension: 嵌入向量维度
        """
```

### RAGRetriever

RAG检索器。

```python
class RAGRetriever(OperatorABC):
    """RAG检索器"""
    
    def __init__(
        self,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        rerank: bool = False,
        rerank_model: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        初始化RAG检索器
        
        Args:
            top_k: 检索的文档数量
            similarity_threshold: 相似度阈值
            rerank: 是否重新排序
            rerank_model: 重排序模型名称
            embedding_model: 嵌入模型名称
        """
    
    def run(
        self,
        storage: MaestroStorage,
        query_path: str,
        knowledge_base_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行文档检索
        
        Args:
            storage: 存储实例
            query_path: 查询数据路径
            knowledge_base_path: 知识库路径
            output_path: 输出路径
            
        Returns:
            Dict[str, Any]: 检索结果
        """
```

### RAGOperator

完整的RAG操作符。

```python
class RAGOperator(OperatorABC):
    """RAG操作符"""
    
    def __init__(
        self,
        llm_serving: LLMServingABC,
        retriever: RAGRetriever,
        max_context_length: int = 2000,
        include_sources: bool = True,
        response_format: str = "detailed",
        system_prompt: Optional[str] = None
    ):
        """
        初始化RAG操作符
        
        Args:
            llm_serving: LLM服务实例
            retriever: RAG检索器实例
            max_context_length: 最大上下文长度
            include_sources: 是否包含来源信息
            response_format: 响应格式 ("simple", "detailed", "structured")
            system_prompt: 系统提示词
        """
    
    def run(
        self,
        storage: MaestroStorage,
        query_path: str,
        knowledge_base_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """执行RAG问答"""
```

## 多模态操作符

### ImageProcessor

图像处理器。

```python
class ImageProcessor(OperatorABC):
    """图像处理器"""
    
    def __init__(
        self,
        llm_serving: LLMServingABC,
        processing_type: str = "description",
        target_column: str = "image_path",
        output_column: str = "result",
        batch_size: int = 5,
        max_image_size: Tuple[int, int] = (1024, 1024)
    ):
        """
        初始化图像处理器
        
        Args:
            llm_serving: LLM服务实例
            processing_type: 处理类型 ("description", "ocr", "analysis")
            target_column: 图像路径列名
            output_column: 输出结果列名
            batch_size: 批处理大小
            max_image_size: 最大图像尺寸
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """处理图像"""
```

### AudioProcessor

音频处理器。

```python
class AudioProcessor(OperatorABC):
    """音频处理器"""
    
    def __init__(
        self,
        llm_serving: LLMServingABC,
        processing_type: str = "transcription",
        target_column: str = "audio_path",
        output_column: str = "result",
        batch_size: int = 3,
        language: str = "auto"
    ):
        """
        初始化音频处理器
        
        Args:
            llm_serving: LLM服务实例
            processing_type: 处理类型 ("transcription", "analysis")
            target_column: 音频路径列名
            output_column: 输出结果列名
            batch_size: 批处理大小
            language: 语言代码 ("auto", "zh", "en", etc.)
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """处理音频"""
```

### VideoProcessor

视频处理器。

```python
class VideoProcessor(OperatorABC):
    """视频处理器"""
    
    def __init__(
        self,
        llm_serving: LLMServingABC,
        processing_type: str = "analysis",
        target_column: str = "video_path",
        output_column: str = "result",
        keyframe_interval: int = 30,
        max_keyframes: int = 10
    ):
        """
        初始化视频处理器
        
        Args:
            llm_serving: LLM服务实例
            processing_type: 处理类型 ("analysis", "keyframes", "summary")
            target_column: 视频路径列名
            output_column: 输出结果列名
            keyframe_interval: 关键帧间隔（秒）
            max_keyframes: 最大关键帧数
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """处理视频"""
```

### MultimodalFusion

多模态融合器。

```python
class MultimodalFusion(OperatorABC):
    """多模态融合器"""
    
    def __init__(
        self,
        llm_serving: LLMServingABC,
        modalities: List[str],
        fusion_strategy: str = "concatenation",
        output_column: str = "fused_result",
        weights: Optional[Dict[str, float]] = None
    ):
        """
        初始化多模态融合器
        
        Args:
            llm_serving: LLM服务实例
            modalities: 模态列表 ["text", "image", "audio"]
            fusion_strategy: 融合策略 ("concatenation", "attention", "cross_modal")
            output_column: 输出结果列名
            weights: 各模态权重
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """执行多模态融合"""
```

## 智能数据处理操作符

### AutoDataCleaner

自动数据清洗器。

```python
class AutoDataCleaner(OperatorABC):
    """自动数据清洗器"""
    
    def __init__(
        self,
        llm_serving: LLMServingABC,
        cleaning_strategies: List[str],
        confidence_threshold: float = 0.8,
        generate_report: bool = True,
        batch_size: int = 100
    ):
        """
        初始化数据清洗器
        
        Args:
            llm_serving: LLM服务实例
            cleaning_strategies: 清洗策略列表
            confidence_threshold: 置信度阈值
            generate_report: 是否生成清洗报告
            batch_size: 批处理大小
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行数据清洗
        
        Returns:
            Dict[str, Any]: 包含以下键的结果字典
                - status: "success" 或 "error"
                - output_path: 输出路径
                - cleaned_count: 清洗的记录数
                - issues_found: 发现的问题数
                - cleaning_report: 清洗报告（如果启用）
        """
```

### SmartAnnotator

智能标注器。

```python
class SmartAnnotator(OperatorABC):
    """智能标注器"""
    
    def __init__(
        self,
        llm_serving: LLMServingABC,
        annotation_type: str,
        target_column: str = "text",
        output_column: str = "annotation",
        categories: Optional[List[str]] = None,
        confidence_threshold: float = 0.7,
        batch_size: int = 20
    ):
        """
        初始化智能标注器
        
        Args:
            llm_serving: LLM服务实例
            annotation_type: 标注类型 ("sentiment", "classification", "entity", "custom")
            target_column: 输入文本列名
            output_column: 输出标注列名
            categories: 分类类别（用于分类任务）
            confidence_threshold: 置信度阈值
            batch_size: 批处理大小
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """执行智能标注"""
```

### FeatureEngineer

特征工程器。

```python
class FeatureEngineer(OperatorABC):
    """特征工程器"""
    
    def __init__(
        self,
        llm_serving: LLMServingABC,
        feature_types: List[str],
        target_column: Optional[str] = None,
        max_features: int = 100,
        feature_selection_method: str = "correlation",
        selection_threshold: float = 0.1
    ):
        """
        初始化特征工程器
        
        Args:
            llm_serving: LLM服务实例
            feature_types: 特征类型列表
            target_column: 目标列名（用于特征选择）
            max_features: 最大特征数
            feature_selection_method: 特征选择方法
            selection_threshold: 选择阈值
        """
    
    def run(
        self,
        storage: MaestroStorage,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行特征工程
        
        Returns:
            Dict[str, Any]: 包含以下键的结果字典
                - status: "success" 或 "error"
                - output_path: 输出路径
                - original_features: 原始特征数
                - generated_features: 生成的特征数
                - selected_features: 选择的特征数
                - feature_importance: 特征重要性（如果可用）
        """
```

## 存储增强

### VectorStorage

向量存储扩展。

```python
class VectorStorage(MaestroStorage):
    """向量存储"""
    
    def __init__(
        self,
        base_storage: MaestroStorage,
        vector_db: VectorDatabaseABC,
        embedding_column: str = "embedding",
        metadata_columns: Optional[List[str]] = None
    ):
        """
        初始化向量存储
        
        Args:
            base_storage: 基础存储实例
            vector_db: 向量数据库实例
            embedding_column: 嵌入向量列名
            metadata_columns: 元数据列名列表
        """
    
    def add_vectors_from_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        从DataFrame添加向量
        
        Args:
            df: 包含向量和元数据的DataFrame
            batch_size: 批处理大小
            
        Returns:
            Dict[str, Any]: 添加结果
        """
    
    def search_similar(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            filter_metadata: 元数据过滤条件
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
```

### FileStorage (增强版)

增强的文件存储。

```python
class FileStorage(MaestroStorage):
    """增强的文件存储"""
    
    def __init__(
        self,
        base_path: str = "./data",
        enable_vector_storage: bool = False,
        enable_model_cache: bool = False,
        vector_db_config: Optional[Dict[str, Any]] = None,
        cache_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化文件存储
        
        Args:
            base_path: 基础路径
            enable_vector_storage: 是否启用向量存储
            enable_model_cache: 是否启用模型缓存
            vector_db_config: 向量数据库配置
            cache_config: 缓存配置
        """
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """添加向量到向量数据库"""
    
    def search_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """在向量数据库中搜索"""
    
    def cache_model_output(
        self,
        model_name: str,
        input_data: Any,
        output_data: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """缓存模型输出"""
    
    def get_cached_model_output(
        self,
        model_name: str,
        input_data: Any
    ) -> Optional[Any]:
        """获取缓存的模型输出"""
    
    def get_vector_stats(self) -> Dict[str, Any]:
        """获取向量存储统计信息"""
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
```

## 模型缓存

### ModelCache

模型缓存管理器。

```python
class ModelCache:
    """模型缓存管理器"""
    
    def __init__(
        self,
        memory_cache: Optional[InMemoryCache] = None,
        disk_cache: Optional[DiskCache] = None,
        default_ttl: int = 3600
    ):
        """
        初始化模型缓存
        
        Args:
            memory_cache: 内存缓存实例
            disk_cache: 磁盘缓存实例
            default_ttl: 默认TTL（秒）
        """
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存值或None
        """
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置缓存项
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒）
            
        Returns:
            bool: 是否成功
        """
    
    def delete(self, key: str) -> bool:
        """删除缓存项"""
    
    def clear(self) -> None:
        """清空所有缓存"""
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 包含以下键的统计信息
                - memory_hits: 内存缓存命中数
                - memory_misses: 内存缓存未命中数
                - disk_hits: 磁盘缓存命中数
                - disk_misses: 磁盘缓存未命中数
                - total_items: 总缓存项数
                - memory_usage: 内存使用量（MB）
                - disk_usage: 磁盘使用量（MB）
        """
```

### 全局缓存函数

```python
def get_global_model_cache() -> ModelCache:
    """获取全局模型缓存实例"""

def cache_model_output(
    model_name: str,
    input_data: Any,
    output_data: Any,
    ttl: Optional[int] = None
) -> bool:
    """
    缓存模型输出
    
    Args:
        model_name: 模型名称
        input_data: 输入数据
        output_data: 输出数据
        ttl: 生存时间（秒）
        
    Returns:
        bool: 是否成功缓存
    """

def get_cached_model_output(
    model_name: str,
    input_data: Any
) -> Optional[Any]:
    """
    获取缓存的模型输出
    
    Args:
        model_name: 模型名称
        input_data: 输入数据
        
    Returns:
        Optional[Any]: 缓存的输出或None
    """

def generate_cache_key(model_name: str, input_data: Any) -> str:
    """
    生成缓存键
    
    Args:
        model_name: 模型名称
        input_data: 输入数据
        
    Returns:
        str: 缓存键
    """
```

## 向量数据库

### VectorDatabaseABC

向量数据库抽象基类。

```python
class VectorDatabaseABC(ABC):
    """向量数据库抽象基类"""
    
    @abstractmethod
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        添加向量
        
        Args:
            vectors: 向量数组
            metadata: 元数据列表
            
        Returns:
            List[str]: 向量ID列表
        """
    
    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            filter_metadata: 元数据过滤条件
            
        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
    
    @abstractmethod
    def delete_vectors(self, vector_ids: List[str]) -> bool:
        """删除向量"""
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
```

### InMemoryVectorDB

内存向量数据库。

```python
class InMemoryVectorDB(VectorDatabaseABC):
    """内存向量数据库"""
    
    def __init__(self, dimension: Optional[int] = None):
        """
        初始化内存向量数据库
        
        Args:
            dimension: 向量维度
        """
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """添加向量到内存"""
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """在内存中搜索相似向量"""
    
    def delete_vectors(self, vector_ids: List[str]) -> bool:
        """从内存中删除向量"""
    
    def get_stats(self) -> Dict[str, Any]:
        """获取内存数据库统计信息"""
```

### FileVectorDB

文件向量数据库。

```python
class FileVectorDB(VectorDatabaseABC):
    """文件向量数据库"""
    
    def __init__(
        self,
        storage_path: str,
        dimension: Optional[int] = None,
        index_type: str = "flat"
    ):
        """
        初始化文件向量数据库
        
        Args:
            storage_path: 存储路径
            dimension: 向量维度
            index_type: 索引类型
        """
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """添加向量到文件"""
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """在文件中搜索相似向量"""
    
    def delete_vectors(self, vector_ids: List[str]) -> bool:
        """从文件中删除向量"""
    
    def get_stats(self) -> Dict[str, Any]:
        """获取文件数据库统计信息"""
    
    def save_to_disk(self) -> bool:
        """保存到磁盘"""
    
    def load_from_disk(self) -> bool:
        """从磁盘加载"""
```

## 错误处理

### 异常类

```python
class MaestroAIError(Exception):
    """MaestroDataflow AI操作符基础异常"""

class ModelLoadError(MaestroAIError):
    """模型加载错误"""

class EmbeddingError(MaestroAIError):
    """嵌入生成错误"""

class VectorSearchError(MaestroAIError):
    """向量搜索错误"""

class CacheError(MaestroAIError):
    """缓存操作错误"""

class MultimodalProcessingError(MaestroAIError):
    """多模态处理错误"""
```

## 配置类

### AIOperatorConfig

AI操作符配置。

```python
@dataclass
class AIOperatorConfig:
    """AI操作符配置"""
    
    # LLM配置
    llm_api_type: str = "openai"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: str = "gpt-3.5-turbo"
    
    # 嵌入模型配置
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    
    # 缓存配置
    enable_caching: bool = True
    cache_ttl: int = 3600
    max_cache_size: int = 1000
    
    # 批处理配置
    default_batch_size: int = 10
    max_batch_size: int = 100
    
    # 向量数据库配置
    vector_db_type: str = "memory"  # "memory" 或 "file"
    vector_dimension: Optional[int] = None
    
    # 多模态配置
    max_image_size: Tuple[int, int] = (1024, 1024)
    supported_audio_formats: List[str] = [".wav", ".mp3", ".m4a"]
    supported_video_formats: List[str] = [".mp4", ".avi", ".mov"]
```

## 工具函数

### 数据处理工具

```python
def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "fixed_size"
) -> List[str]:
    """
    文本分块
    
    Args:
        text: 输入文本
        chunk_size: 块大小
        chunk_overlap: 重叠大小
        strategy: 分块策略
        
    Returns:
        List[str]: 文本块列表
    """

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """标准化嵌入向量"""

def calculate_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
    metric: str = "cosine"
) -> float:
    """计算向量相似度"""

def batch_process(
    data: List[Any],
    process_func: Callable,
    batch_size: int = 10
) -> List[Any]:
    """批处理数据"""
```

### 验证工具

```python
def validate_embedding_dimension(
    embeddings: np.ndarray,
    expected_dim: Optional[int] = None
) -> bool:
    """验证嵌入向量维度"""

def validate_model_config(config: Dict[str, Any]) -> bool:
    """验证模型配置"""

def validate_storage_path(path: str) -> bool:
    """验证存储路径"""
```

## 版本信息

```python
__version__ = "1.0.0"
__author__ = "MaestroDataflow Team"
__email__ = "support@maestrodataflow.com"

# 各模块版本
AI_OPERATORS_VERSION = "1.0.0"
STORAGE_ENHANCEMENT_VERSION = "1.0.0"
MODEL_CACHE_VERSION = "1.0.0"
VECTOR_DB_VERSION = "1.0.0"
```

---

*本API参考文档涵盖了MaestroDataflow AI操作符生态系统的所有公共接口。如需了解更多实现细节，请参考源代码和使用指南。*