# MaestroDataflow AI操作符使用指南

## 概述

MaestroDataflow AI操作符生态系统为数据处理工作流提供了强大的人工智能能力。本指南将详细介绍各种AI操作符的使用方法、配置选项和最佳实践。

## 目录

1. [快速开始](#快速开始)
2. [文本生成操作符](#文本生成操作符)
3. [嵌入向量操作符](#嵌入向量操作符)
4. [RAG操作符](#rag操作符)
5. [多模态操作符](#多模态操作符)
6. [智能数据处理操作符](#智能数据处理操作符)
7. [配置和优化](#配置和优化)
8. [最佳实践](#最佳实践)
9. [故障排除](#故障排除)

## 快速开始

### 环境设置

```python
from maestro.utils.storage import FileStorage
from maestro.serving.enhanced_llm_serving import EnhancedLLMServing
from maestro.operators.ai_ops import *

# 创建存储实例
storage = FileStorage(
    input_file_path="../data/input.csv",
    cache_path="../output/ai_example/cache",
    enable_vector_storage=True,
    enable_model_cache=True
)

# 创建LLM服务
llm_serving = EnhancedLLMServing(
    api_type="openai",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"
)
```

### 基本使用模式

```python
# 1. 准备数据
import pandas as pd
data = pd.DataFrame({"text": ["示例文本1", "示例文本2"]})
storage.write(data, "input_data")

# 2. 创建操作符
operator = TextSummarizer(llm_serving=llm_serving)

# 3. 执行操作
result = operator.run(storage, input_path="input_data", output_path="results")

# 4. 查看结果
output_data = storage.read(input_path="results")
```

## 文本生成操作符

### PromptedGenerator - 提示词生成器

用于基于自定义提示词生成文本内容。

#### 基本用法

```python
from maestro.core.prompt import DIYPrompt

# 创建自定义提示词
prompt = DIYPrompt("请为以下产品写一个营销文案：{text}")

# 创建生成器
generator = PromptedGenerator(
    llm_serving=llm_serving,
    prompt=prompt,
    max_tokens=200,
    temperature=0.7
)

# 执行生成
result = generator.run(
    storage,
    input_path="products",
    output_path="marketing_copy"
)
```

#### 配置参数

- `llm_serving`: LLM服务实例
- `prompt`: 提示词对象（StandardPrompt或DIYPrompt）
- `input_column`: 目标列名（默认："text"）
- `output_column`: 输出列名（默认："generated_text"）
- `batch_size`: 批处理大小
- `**generation_kwargs`: 传递给LLM的额外参数

### TextSummarizer - 文本摘要器

用于生成文本摘要。

#### 基本用法

```python
summarizer = TextSummarizer(
    llm_serving=llm_serving,
    max_length=100,
    summary_type="abstractive",  # "extractive" 或 "abstractive"
    target_column="content"
)

result = summarizer.run(
    storage,
    input_path="articles",
    output_path="summaries"
)
```

#### 摘要类型

- **extractive**: 抽取式摘要，从原文中选择重要句子
- **abstractive**: 生成式摘要，重新组织语言生成摘要

### TextClassifier - 文本分类器

用于对文本进行分类。

#### 基本用法

```python
classifier = TextClassifier(
    llm_serving=llm_serving,
    categories=["正面", "负面", "中性"],
    target_column="comment",
    output_column="sentiment"
)

result = classifier.run(
    storage,
    input_path="reviews",
    output_path="classified_reviews"
)
```

## 嵌入向量操作符

### EmbeddingGenerator - 嵌入生成器

用于生成文本的向量表示。

#### 基本用法

```python
embedding_generator = EmbeddingGenerator(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    input_column="text",
    output_column="embedding",
    device="cpu"  # 或 "cuda"
)

result = embedding_generator.run(
    storage,
    input_path="texts",
    output_path="embeddings"
)
```

#### 支持的模型

- `sentence-transformers/all-MiniLM-L6-v2`: 轻量级多语言模型
- `sentence-transformers/all-mpnet-base-v2`: 高质量英文模型
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: 多语言模型

### SimilarityCalculator - 相似度计算器

用于计算文本或向量之间的相似度。

#### 基本用法

```python
similarity_calculator = SimilarityCalculator(
    metric="cosine",  # "cosine", "dot_product", "euclidean"
    target_columns=["text1", "text2"],
    output_column="similarity"
)

result = similarity_calculator.run(
    storage,
    input_path="text_pairs",
    output_path="similarities"
)
```

### TextMatcher - 文本匹配器

用于在参考文本集合中找到最相似的文本。

#### 基本用法

```python
reference_texts = [
    "产品质量很好",
    "价格比较合理",
    "服务态度不错"
]

text_matcher = TextMatcher(
    reference_texts=reference_texts,
    similarity_threshold=0.7,
    target_column="query",
    top_k=3
)

result = text_matcher.run(
    storage,
    input_path="queries",
    output_path="matches"
)
```

## RAG操作符

### KnowledgeBaseBuilder - 知识库构建器

用于构建向量知识库。

#### 基本用法

```python
kb_builder = KnowledgeBaseBuilder(
    chunk_size=500,
    chunk_overlap=50,
    text_column="document",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

result = kb_builder.run(
    storage,
    input_path="documents",
    output_path="knowledge_base"
)
```

#### 分块策略

- `chunk_size`: 文本块大小
- `chunk_overlap`: 块之间的重叠
- `chunking_strategy`: 分块策略（"fixed_size", "sentence", "paragraph"）

### RAGRetriever - RAG检索器

用于从知识库中检索相关文档。

#### 基本用法

```python
retriever = RAGRetriever(
    top_k=5,
    similarity_threshold=0.3,
    rerank=True
)

result = retriever.run(
    storage,
    query_path="queries",
    knowledge_base_path="knowledge_base",
    output_path="retrieved_docs"
)
```

### RAGOperator - RAG操作符

结合检索和生成的完整RAG系统。

#### 基本用法

```python
rag_operator = RAGOperator(
    llm_serving=llm_serving,
    retriever=retriever,
    max_context_length=2000,
    include_sources=True,
    response_format="detailed"
)

result = rag_operator.run(
    storage,
    query_path="questions",
    knowledge_base_path="knowledge_base",
    output_path="answers"
)
```

## 多模态操作符

### ImageProcessor - 图像处理器

用于处理和分析图像。

#### 基本用法

```python
image_processor = ImageProcessor(
    llm_serving=llm_serving,
    processing_type="description",  # "description", "ocr", "analysis"
    output_column="image_description"
)

result = image_processor.run(
    storage,
    input_path="images",
    output_path="image_analysis"
)
```

#### 处理类型

- **description**: 生成图像描述
- **ocr**: 光学字符识别
- **analysis**: 深度图像分析

### AudioProcessor - 音频处理器

用于处理音频文件。

#### 基本用法

```python
audio_processor = AudioProcessor(
    llm_serving=llm_serving,
    processing_type="transcription",  # "transcription", "analysis"
    output_column="transcription"
)

result = audio_processor.run(
    storage,
    input_path="audio_files",
    output_path="transcriptions"
)
```

### MultimodalFusion - 多模态融合

用于融合多种模态的信息。

#### 基本用法

```python
multimodal_fusion = MultimodalFusion(
    llm_serving=llm_serving,
    modalities=["text", "image", "audio"],
    fusion_strategy="concatenation",  # "concatenation", "attention", "cross_modal"
    output_column="fused_analysis"
)

result = multimodal_fusion.run(
    storage,
    input_path="multimodal_data",
    output_path="fused_results"
)
```

## 智能数据处理操作符

### AutoDataCleaner - 自动数据清洗器

使用AI智能识别和处理数据质量问题。

#### 基本用法

```python
data_cleaner = AutoDataCleaner(
    llm_serving=llm_serving,
    cleaning_strategies=[
        "remove_duplicates",
        "handle_missing",
        "standardize_format",
        "fix_typos",
        "detect_outliers"
    ],
    confidence_threshold=0.8,
    generate_report=True
)

result = data_cleaner.run(
    storage,
    input_path="dirty_data",
    output_path="cleaned_data"
)
```

#### 清洗策略

- **remove_duplicates**: 移除重复记录
- **handle_missing**: 处理缺失值
- **standardize_format**: 标准化格式
- **fix_typos**: 修复拼写错误
- **detect_outliers**: 检测异常值

### SmartAnnotator - 智能标注器

使用AI自动为数据添加标签和注释。

#### 基本用法

```python
# 情感分析标注
sentiment_annotator = SmartAnnotator(
    llm_serving=llm_serving,
    annotation_type="sentiment",
    target_column="comment",
    output_column="sentiment"
)

# 分类标注
category_annotator = SmartAnnotator(
    llm_serving=llm_serving,
    annotation_type="classification",
    target_column="text",
    output_column="category",
    categories=["类别1", "类别2", "类别3"]
)

# 实体识别标注
entity_annotator = SmartAnnotator(
    llm_serving=llm_serving,
    annotation_type="entity",
    target_column="text",
    output_column="entities"
)
```

#### 标注类型

- **sentiment**: 情感分析
- **classification**: 文本分类
- **entity**: 命名实体识别
- **custom**: 自定义标注

### FeatureEngineer - 特征工程器

使用AI自动生成和选择特征。

#### 基本用法

```python
feature_engineer = FeatureEngineer(
    llm_serving=llm_serving,
    feature_types=[
        "statistical",
        "temporal", 
        "text",
        "categorical",
        "interaction"
    ],
    target_column="target",
    max_features=50,
    feature_selection_method="correlation"
)

result = feature_engineer.run(
    storage,
    input_path="raw_features",
    output_path="engineered_features"
)
```

#### 特征类型

- **statistical**: 统计特征（均值、方差、分位数等）
- **temporal**: 时间特征（年、月、日、星期等）
- **text**: 文本特征（长度、词数、字符数等）
- **categorical**: 分类特征（频次编码、标签编码等）
- **interaction**: 交互特征（乘积、比值等）

## 配置和优化

### LLM服务配置

#### 本地模型配置

```python
from maestro.serving.enhanced_llm_serving import LocalLLMServing

llm_serving = LocalLLMServing(
    model_name="microsoft/DialoGPT-medium",
    device="cuda",  # 或 "cpu"
    max_length=512,
    enable_caching=True,
    cache_config={
        "ttl": 3600,  # 缓存时间（秒）
        "max_size": 1000  # 最大缓存条目
    }
)
```

#### API服务配置

```python
from maestro.serving.enhanced_llm_serving import EnhancedLLMServing

llm_serving = EnhancedLLMServing(
    api_type="openai",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-3.5-turbo",
    enable_caching=True,
    enable_batching=True,
    batch_size=10,
    batch_timeout=5.0
)
```

### 存储配置

```python
storage = FileStorage(
    base_path="./data",
    enable_vector_storage=True,
    enable_model_cache=True,
    vector_db_config={
        "dimension": 384,
        "index_type": "flat"
    },
    cache_config={
        "max_memory_size": 100,  # MB
        "max_disk_size": 1000,   # MB
        "ttl": 3600             # 秒
    }
)
```

### 性能优化

#### 批处理优化

```python
# 启用批处理
operator = TextClassifier(
    llm_serving=llm_serving,
    batch_size=20,  # 增加批处理大小
    categories=["正面", "负面", "中性"]
)
```

#### 缓存优化

```python
# 启用模型输出缓存
from maestro.utils.model_cache import cache_model_output, get_cached_model_output

# 手动缓存
cache_model_output("model_name", input_data, output_data)

# 检查缓存
cached_result = get_cached_model_output("model_name", input_data)
```

#### 向量存储优化

```python
# 批量添加向量
storage.add_vectors(
    vectors=embeddings,
    metadata=metadata,
    batch_size=1000
)

# 优化搜索
results = storage.search_vectors(
    query_vector=query_embedding,
    top_k=10,
    filter_metadata={"category": "产品评论"}
)
```

## 最佳实践

### 1. 数据预处理

```python
# 在使用AI操作符之前进行基础清洗
data_cleaner = AutoDataCleaner(
    llm_serving=llm_serving,
    cleaning_strategies=["remove_duplicates", "handle_missing"]
)
cleaned_result = data_cleaner.run(storage, input_path="raw", output_path="clean")
```

### 2. 渐进式处理

```python
# 将复杂任务分解为多个步骤
# 步骤1: 基础标注
annotator = SmartAnnotator(llm_serving=llm_serving, annotation_type="sentiment")
step1_result = annotator.run(storage, input_path="clean", output_path="step1")

# 步骤2: 特征工程
engineer = FeatureEngineer(llm_serving=llm_serving)
step2_result = engineer.run(storage, input_path="step1", output_path="step2")
```

### 3. 错误处理

```python
try:
    result = operator.run(storage, input_path="data", output_path="results")
    if result["status"] == "error":
        print(f"操作失败: {result['error']}")
        # 处理错误逻辑
except Exception as e:
    print(f"执行异常: {e}")
    # 异常处理逻辑
```

### 4. 监控和日志

```python
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

# 操作符会自动记录执行日志
operator = TextSummarizer(llm_serving=llm_serving)
result = operator.run(storage, input_path="data", output_path="results")

# 检查执行统计
print(f"处理时间: {result.get('execution_time', 0)} 秒")
print(f"处理记录数: {result.get('processed_count', 0)}")
```

### 5. 资源管理

```python
# 定期清理缓存
if hasattr(storage, 'clear_cache'):
    storage.clear_cache()

# 监控资源使用
cache_stats = storage.get_cache_stats()
vector_stats = storage.get_vector_stats()

print(f"缓存使用: {cache_stats}")
print(f"向量存储: {vector_stats}")
```

## 故障排除

### 常见问题

#### 1. 内存不足

**问题**: 处理大量数据时内存溢出

**解决方案**:
```python
# 减少批处理大小
operator = TextClassifier(batch_size=5)  # 默认是10

# 启用磁盘缓存
storage = FileStorage(
    enable_model_cache=True,
    cache_config={"max_memory_size": 50}  # 减少内存缓存
)
```

#### 2. API调用限制

**问题**: API调用频率过高被限制

**解决方案**:
```python
# 启用缓存避免重复调用
llm_serving = EnhancedLLMServing(
    enable_caching=True,
    cache_config={"ttl": 7200}  # 增加缓存时间
)

# 增加批处理减少调用次数
operator = TextSummarizer(batch_size=20)
```

#### 3. 模型加载失败

**问题**: 本地模型无法加载

**解决方案**:
```python
try:
    llm_serving = LocalLLMServing(model_name="model_name")
except Exception as e:
    print(f"本地模型加载失败: {e}")
    # 回退到API服务
    llm_serving = EnhancedLLMServing(api_type="openai")
```

#### 4. 向量维度不匹配

**问题**: 嵌入向量维度不一致

**解决方案**:
```python
# 确保使用相同的嵌入模型
embedding_generator = EmbeddingGenerator(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # 固定模型
)

# 检查向量维度
vector_stats = storage.get_vector_stats()
print(f"向量维度: {vector_stats.get('dimension', 'unknown')}")
```

### 调试技巧

#### 1. 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 操作符会输出详细的执行信息
```

#### 2. 检查中间结果

```python
# 保存中间结果进行检查
result = operator.run(storage, input_path="data", output_path="intermediate")
intermediate_data = storage.read(input_path="intermediate")
print(intermediate_data.head())
```

#### 3. 使用小数据集测试

```python
# 先用小数据集测试
test_data = data.head(10)  # 只取前10条
storage.write(test_data, "test_data")
result = operator.run(storage, input_path="test_data", output_path="test_results")
```

## 扩展和自定义

### 创建自定义操作符

```python
from maestro.core.operator import OperatorABC
from maestro.core.prompt import PromptABC

class CustomAIOperator(OperatorABC):
    ALLOWED_PROMPTS = (StandardPrompt, DIYPrompt)
    
    def __init__(self, llm_serving, custom_param):
        super().__init__()
        self.llm_serving = llm_serving
        self.custom_param = custom_param
    
    def run(self, storage, **kwargs):
        self.log_operation_start("custom_operation", kwargs)
        
        try:
            # 实现自定义逻辑
            data = storage.read(**kwargs)
            
            # 处理数据
            processed_data = self._process_data(data)
            
            # 保存结果
            output_path = kwargs.get('output_path', 'custom_output')
            storage.write(processed_data, output_path)
            
            self.log_operation_end("custom_operation", {
                "processed_count": len(processed_data)
            })
            
            return {
                "status": "success",
                "output_path": output_path,
                "processed_count": len(processed_data)
            }
            
        except Exception as e:
            return self.handle_error("custom_operation", e)
    
    def _process_data(self, data):
        # 实现具体的处理逻辑
        return data
```

### 自定义提示词

```python
from maestro.core.prompt import DIYPrompt

# 创建专业领域的提示词
medical_prompt = DIYPrompt(
    "作为医学专家，请分析以下症状描述并提供初步诊断建议：\n"
    "症状: {symptoms}\n"
    "请提供：1. 可能的诊断 2. 建议的检查 3. 注意事项"
)

# 使用自定义提示词
medical_analyzer = PromptedGenerator(
    llm_serving=llm_serving,
    prompt=medical_prompt
)
```

## 版本更新和兼容性

### 版本检查

```python
import maestro
print(f"MaestroDataflow 版本: {maestro.__version__}")

# 检查操作符版本兼容性
from maestro.operators.ai_ops import __version__ as ai_ops_version
print(f"AI操作符版本: {ai_ops_version}")
```

### 迁移指南

当升级到新版本时，请注意以下变更：

1. **配置参数变更**: 检查操作符构造函数参数
2. **API变更**: 查看方法签名是否有变化
3. **依赖更新**: 更新相关依赖包版本
4. **数据格式**: 确认输入输出数据格式兼容性

## 社区和支持

### 获取帮助

- **文档**: 查看最新文档和API参考
- **示例**: 参考 `examples/` 目录中的示例代码
- **问题反馈**: 通过GitHub Issues报告问题
- **讨论**: 参与社区讨论获取帮助

### 贡献指南

欢迎为MaestroDataflow AI操作符生态系统做出贡献：

1. **报告Bug**: 详细描述问题和复现步骤
2. **功能建议**: 提出新功能需求和改进建议
3. **代码贡献**: 提交Pull Request改进代码
4. **文档改进**: 完善文档和示例

---

*本指南涵盖了MaestroDataflow AI操作符的主要功能和使用方法。随着系统的不断发展，我们会持续更新和完善本文档。*