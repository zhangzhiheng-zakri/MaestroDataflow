# MaestroDataflow AI操作符生态系统

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)

## 🚀 概述

MaestroDataflow AI操作符生态系统是一个强大的、模块化的人工智能数据处理框架，为数据科学家和开发者提供了丰富的AI能力，包括文本生成、嵌入向量、RAG（检索增强生成）、多模态处理和智能数据处理等功能。

### ✨ 核心特性

- **🤖 丰富的AI操作符**: 文本生成、分类、摘要、嵌入、RAG、多模态处理等
- **🔧 模块化设计**: 可组合的操作符，支持复杂工作流构建
- **💾 智能存储**: 集成向量数据库和模型缓存，提升性能
- **🌐 多模态支持**: 处理文本、图像、音频、视频等多种数据类型
- **⚡ 高性能**: 批处理、缓存、异步处理等优化机制
- **🔌 易于扩展**: 简单的接口设计，支持自定义操作符
- **📊 智能数据处理**: AI驱动的数据清洗、标注和特征工程

## 📦 安装

### 基础安装

```bash
pip install maestro-dataflow
```

### 完整安装（包含所有AI依赖）

```bash
pip install maestro-dataflow[ai]
```

### 开发安装

```bash
git clone https://github.com/maestro-dataflow/MaestroDataflow.git
cd MaestroDataflow
pip install -e .[dev,ai]
```

## 🚀 快速开始

### 1. 基础设置

```python
from maestro.utils.storage import FileStorage
from maestro.serving.enhanced_llm_serving import EnhancedLLMServing
from maestro.operators.ai_ops import *

# 创建存储实例
storage = FileStorage(
    input_file_path="../data/input.csv",
    cache_path="../output/ai_operators_demo/cache",
    enable_vector_storage=True,
    enable_model_cache=True
)

# 创建LLM服务
llm_serving = EnhancedLLMServing(
    api_type="openai",
    api_key="your-api-key"
)
```

### 2. 文本生成示例

```python
import pandas as pd

# 准备数据
data = pd.DataFrame({
    "product": ["智能手机", "笔记本电脑", "无线耳机"]
})
storage.write(data, "products")

# 创建文本生成器
from maestro.core.prompt import DIYPrompt
prompt = DIYPrompt("为以下产品写一个吸引人的营销文案：{product}")
generator = PromptedGenerator(llm_serving=llm_serving, prompt=prompt)

# 执行生成
result = generator.run(storage, input_path="products", output_path="marketing_copy")
print(f"生成完成，处理了 {result['processed_count']} 个产品")

# 查看结果
output = storage.read(input_path="marketing_copy")
print(output.head())
```

### 3. 智能数据清洗示例

```python
# 准备需要清洗的数据
dirty_data = pd.DataFrame({
    "name": ["张三", "李四", "王五", "张三", "赵六", None, "钱七"],
    "email": ["zhang@email.com", "li@email", "wang@email.com", "zhang@email.com", "zhao@email.com", "", "qian@email.com"],
    "age": [25, 30, None, 25, 35, 28, 40],
    "comment": ["很好的产品", "质量不错", "还可以", "很好的产品", "非常满意", "一般般", "需要改进"]
})
storage.write(dirty_data, "dirty_data")

# 数据清洗
cleaner = AutoDataCleaner(
    llm_serving=llm_serving,
    cleaning_strategies=["remove_duplicates", "handle_missing", "standardize_format"]
)
result = cleaner.run(storage, input_path="dirty_data", output_path="clean_data")

# 智能标注
annotator = SmartAnnotator(
    llm_serving=llm_serving,
    annotation_type="sentiment"
)
result = annotator.run(storage, input_path="clean_data", output_path="annotated_data")

# 查看结果
result = storage.read(input_path="annotated_data")
print(result)
```

## 🏗️ 架构概览

```
MaestroDataflow AI操作符生态系统
├── 文本生成操作符
│   ├── PromptedGenerator     # 基于提示词的文本生成
│   ├── TextSummarizer       # 文本摘要
│   └── TextClassifier       # 文本分类
├── 嵌入向量操作符
│   ├── EmbeddingGenerator   # 嵌入向量生成
│   ├── SimilarityCalculator # 相似度计算
│   └── TextMatcher         # 文本匹配
├── RAG操作符
│   ├── KnowledgeBaseBuilder # 知识库构建
│   ├── RAGRetriever        # 文档检索
│   └── RAGOperator         # 完整RAG系统
├── 多模态操作符
│   ├── ImageProcessor      # 图像处理
│   ├── AudioProcessor      # 音频处理
│   ├── VideoProcessor      # 视频处理
│   └── MultimodalFusion    # 多模态融合
├── 智能数据处理操作符
│   ├── AutoDataCleaner     # 自动数据清洗
│   ├── SmartAnnotator      # 智能标注
│   └── FeatureEngineer     # 特征工程
└── 存储增强
    ├── VectorDatabase      # 向量数据库
    ├── ModelCache         # 模型缓存
    └── EnhancedStorage    # 增强存储
```

## 📚 操作符详解

### 文本生成操作符

#### PromptedGenerator - 提示词生成器
- **功能**: 基于自定义提示词生成文本
- **适用场景**: 内容创作、文案生成、对话系统
- **特性**: 支持批处理、温度控制、长度限制

#### TextSummarizer - 文本摘要器
- **功能**: 生成文本摘要
- **摘要类型**: 抽取式、生成式
- **适用场景**: 文档摘要、新闻摘要、报告生成

#### TextClassifier - 文本分类器
- **功能**: 对文本进行分类
- **支持**: 多分类、置信度评估
- **适用场景**: 情感分析、主题分类、内容审核

### 嵌入向量操作符

#### EmbeddingGenerator - 嵌入生成器
- **功能**: 将文本转换为向量表示
- **支持模型**: Sentence-Transformers、OpenAI Embeddings
- **特性**: 批处理、GPU加速、向量标准化

#### SimilarityCalculator - 相似度计算器
- **功能**: 计算文本或向量相似度
- **度量方式**: 余弦相似度、点积、欧几里得距离
- **适用场景**: 文档检索、推荐系统、去重

#### TextMatcher - 文本匹配器
- **功能**: 在参考文本集合中找到最相似的文本
- **特性**: 阈值过滤、Top-K结果、相似度评分
- **适用场景**: 问答匹配、内容推荐

### RAG操作符

#### KnowledgeBaseBuilder - 知识库构建器
- **功能**: 构建向量化知识库
- **分块策略**: 固定大小、句子级别、段落级别
- **特性**: 重叠处理、元数据保留、批量处理

#### RAGRetriever - RAG检索器
- **功能**: 从知识库检索相关文档
- **特性**: 相似度过滤、重排序、元数据过滤
- **优化**: 缓存机制、批量检索

#### RAGOperator - RAG操作符
- **功能**: 完整的检索增强生成系统
- **特性**: 上下文管理、来源追踪、响应格式化
- **适用场景**: 智能问答、知识助手、文档查询

### 多模态操作符

#### ImageProcessor - 图像处理器
- **功能**: 图像分析和描述
- **处理类型**: 图像描述、OCR、深度分析
- **支持格式**: JPEG、PNG、WebP等

#### AudioProcessor - 音频处理器
- **功能**: 音频转录和分析
- **处理类型**: 语音转文本、音频分析
- **支持格式**: WAV、MP3、M4A等

#### VideoProcessor - 视频处理器
- **功能**: 视频分析和关键帧提取
- **特性**: 关键帧提取、视频摘要、场景分析
- **支持格式**: MP4、AVI、MOV等

#### MultimodalFusion - 多模态融合
- **功能**: 融合多种模态信息
- **融合策略**: 拼接、注意力机制、跨模态交互
- **适用场景**: 多媒体分析、内容理解

### 智能数据处理操作符

#### AutoDataCleaner - 自动数据清洗器
- **功能**: AI驱动的数据质量改善
- **清洗策略**: 去重、缺失值处理、格式标准化、拼写纠错
- **特性**: 智能检测、置信度评估、清洗报告

#### SmartAnnotator - 智能标注器
- **功能**: 自动数据标注
- **标注类型**: 情感分析、分类、实体识别、自定义标注
- **特性**: 批处理、置信度评估、增量标注

#### FeatureEngineer - 特征工程器
- **功能**: 自动特征生成和选择
- **特征类型**: 统计、时间、文本、分类、交互特征
- **选择方法**: 相关性、重要性、统计显著性

## 🔧 配置和优化

### LLM服务配置

```python
# OpenAI配置
llm_serving = EnhancedLLMServing(
    api_type="openai",
    api_key="your-api-key",
    model="gpt-3.5-turbo",
    enable_caching=True,
    enable_batching=True
)

# 本地模型配置
llm_serving = LocalLLMServing(
    model_name="microsoft/DialoGPT-medium",
    device="cuda",
    enable_caching=True
)
```

### 存储配置

```python
# FileStorage配置
storage = FileStorage(
    input_file_path="../data/input.csv",
    cache_path="../output/ai_operators_demo/cache",
    file_name_prefix="ai_demo",
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

```python
# 批处理优化
operator = TextClassifier(
    llm_serving=llm_serving,
    batch_size=20,  # 增加批处理大小
    categories=["正面", "负面", "中性"]
)

# 缓存优化
from maestro.utils.model_cache import cache_model_output
cache_model_output("model_name", input_data, output_data)

# 向量存储优化
storage.add_vectors(
    vectors=embeddings,
    metadata=metadata,
    batch_size=1000
)
```

## 📖 文档和示例

### 文档结构

```
docs/
├── AI_OPERATORS_GUIDE.md    # 详细使用指南
├── API_REFERENCE.md         # 完整API参考
├── EXAMPLES.md             # 示例集合
└── TROUBLESHOOTING.md      # 故障排除指南

examples/
├── ai_operators_demo.py         # 基础操作符演示
├── comprehensive_ai_workflow.py # 综合工作流示例
├── rag_system_example.py        # RAG系统示例
├── multimodal_processing.py     # 多模态处理示例
└── intelligent_data_processing.py # 智能数据处理示例
```

### 在线资源

- **📚 完整文档**: [docs/AI_OPERATORS_GUIDE.md](docs/AI_OPERATORS_GUIDE.md)
- **🔍 API参考**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **💡 示例代码**: [examples/](examples/)
- **🐛 问题反馈**: [GitHub Issues](https://github.com/maestro-dataflow/MaestroDataflow/issues)

## 🛠️ 开发和扩展

### 创建自定义操作符

```python
from maestro.core.operator import OperatorABC

class CustomAIOperator(OperatorABC):
    def __init__(self, llm_serving, custom_param):
        super().__init__()
        self.llm_serving = llm_serving
        self.custom_param = custom_param
    
    def run(self, storage, **kwargs):
        # 实现自定义逻辑
        pass
```

### 贡献指南

1. **Fork项目** 并创建特性分支
2. **编写代码** 并添加测试
3. **更新文档** 和示例
4. **提交PR** 并描述变更

## 🔄 版本历史

### v1.0.0 (当前版本)
- ✅ 完整的AI操作符生态系统
- ✅ 文本生成、嵌入、RAG操作符
- ✅ 多模态处理能力
- ✅ 智能数据处理操作符
- ✅ 向量数据库和模型缓存
- ✅ 综合文档和示例

### 未来规划
- 🔮 更多预训练模型支持
- 🔮 分布式处理能力
- 🔮 可视化界面
- 🔮 更多数据源连接器
- 🔮 AutoML集成

## 🤝 社区和支持

### 获取帮助

- **📖 文档**: 查看详细文档和API参考
- **💬 讨论**: 参与GitHub Discussions
- **🐛 报告问题**: 通过GitHub Issues
- **📧 联系我们**: support@maestrodataflow.com

### 贡献方式

- **代码贡献**: 提交功能改进和bug修复
- **文档改进**: 完善文档和示例
- **问题反馈**: 报告bug和提出建议
- **社区支持**: 帮助其他用户解决问题

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢所有为MaestroDataflow AI操作符生态系统做出贡献的开发者和用户！

特别感谢以下开源项目：
- [Transformers](https://github.com/huggingface/transformers)
- [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Pandas](https://github.com/pandas-dev/pandas)
- [NumPy](https://github.com/numpy/numpy)

---

**🚀 开始您的AI数据处理之旅吧！**

如果您觉得这个项目有用，请给我们一个 ⭐️ Star！