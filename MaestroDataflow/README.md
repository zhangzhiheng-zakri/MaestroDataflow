# MaestroDataflow

MaestroDataflow 是一个强大的AI增强数据处理框架，专为高效处理多种格式的数据文件而设计。它不仅提供了统一的接口来处理 XLSX、CSV、JSON、JSONL 和 Parquet 格式的数据，还集成了先进的AI功能，包括向量数据库、模型缓存和智能数据处理操作符。

## 特点

### 核心功能
- **多格式支持**：无缝处理 XLSX、CSV、JSON、JSONL 和 Parquet 格式的数据文件
- **统一接口**：使用一致的 API 处理不同格式的数据
- **简单易用**：直观的方法命名和链式调用支持
- **高效处理**：优化的数据处理流程，支持大型数据集
- **格式转换**：轻松在不同格式之间转换数据
- **缓存机制**：内置缓存系统，提高处理效率
- **管道处理**：支持复杂的数据处理管道和操作符

### AI增强功能
- **向量数据库**：内置向量存储和相似性搜索功能
- **模型缓存**：智能模型输出缓存，提高AI应用性能
- **AI操作符**：丰富的AI数据处理操作符，包括文本分析、情感分析等
- **LLM集成**：支持本地和API LLM服务集成
- **智能数据清洗**：基于AI的自动数据清洗和质量检测

### 数据分析与可视化功能 🆕
- **统计分析**：全面的描述性统计、增长率计算、趋势分析
- **数据可视化**：支持线图、柱状图、散点图、饼图、热力图等多种图表类型
- **交互式仪表板**：基于Plotly的响应式仪表板生成
- **专业报告**：自动生成HTML/PDF格式的数据分析报告
- **模板系统**：提供多种报告模板（综合报告、执行摘要、技术报告）

## 安装

```bash
pip install MaestroDataflow
```

或者从源代码安装：

```bash
git clone https://github.com/maestro-dataflow/MaestroDataflow.git
cd MaestroDataflow
pip install -e .
```

## 快速开始

### 基本用法

```python
from maestro.utils.storage import FileStorage
from maestro.pipeline.pipeline import Pipeline
from maestro.operators.basic_ops import FilterRowsOperator, SelectColumnsOperator

# 创建存储实例
storage = FileStorage(
    input_file_path="../sample_data/employees.csv",
    cache_path="../output/basic_example/cache",
    file_name_prefix="process",
    cache_type="xlsx"
)

# 创建管道和操作符
pipeline = Pipeline(storage=storage)
filter_op = FilterRowsOperator(condition=lambda row: row['salary'] > 50000)
select_op = SelectColumnsOperator(columns=['name', 'department', 'salary'])

# 添加操作符到管道
pipeline.add_operator(filter_op)
pipeline.add_operator(select_op)

# 运行管道
pipeline.run()

# 读取处理结果
result_data = storage.step().read()
print(result_data)
```

### 数据分析与可视化示例 🆕

```python
from maestro.pipeline.pipeline import Pipeline
from maestro.utils.storage import FileStorage
from maestro.operators.analytics_ops import DataAnalysisOperator, DataSummaryOperator
from maestro.operators.visualization_ops import ChartGeneratorOperator, DashboardGeneratorOperator
from maestro.operators.report_ops import HTMLReportGeneratorOperator

# 创建存储实例
storage = FileStorage(
    input_file_path="../data/sales_data.csv",
    cache_path="../output/analytics_example/cache"
)

# 创建数据分析工作流
workflow = Pipeline(storage=storage)

# 添加数据分析算子
analysis_op = DataAnalysisOperator(
    columns_to_analyze=['销售额', '利润率'],
    time_column='日期',
    include_growth_analysis=True
)

# 添加图表生成算子
chart_op = ChartGeneratorOperator(
    chart_type='line',
    x_column='日期',
    y_columns=['销售额'],
    title='销售趋势分析',
    output_file='sales_trend.png'
)

# 添加仪表板生成算子
dashboard_op = DashboardGeneratorOperator(
    dashboard_title='销售数据仪表板',
    chart_configs=[
        {
            'type': 'line',
            'x_column': '日期',
            'y_columns': ['销售额'],
            'title': '销售趋势'
        },
        {
            'type': 'bar',
            'x_column': '月份',
            'y_columns': ['利润率'],
            'title': '月度利润率'
        }
    ]
)

# 添加报告生成算子
report_op = HTMLReportGeneratorOperator(
    report_title='销售数据分析报告',
    output_file='sales_report.html',
    template_style='modern'
)

# 构建工作流
workflow.add_operator("analysis", analysis_op)
workflow.add_operator("chart", chart_op, depends_on=["analysis"])
workflow.add_operator("dashboard", dashboard_op, depends_on=["chart"])
workflow.add_operator("report", report_op, depends_on=["dashboard"])

# 执行工作流
result = workflow.run(sales_data)
```

### AI增强存储系统

```python
from maestro.utils.storage import FileStorage

# 创建AI增强的FileStorage实例
storage = FileStorage(
    input_file_path="./sample_data/employees.csv",
    cache_path="./cache",
    file_name_prefix="ai_process",
    cache_type="csv",
    enable_vector_storage=True,    # 启用向量存储
    enable_model_cache=True,       # 启用模型缓存
    vector_db_config={"similarity_metric": "cosine"},
    model_cache_config={
        "cache_type": "hybrid",
        "cache_config": {
            "memory": {"max_size": 100, "default_ttl": 3600},
            "disk": {"cache_dir": "./cache/model_cache", "max_size_mb": 500}
        }
    }
)

# 初始化并读取数据
storage.step()
data = storage.read("dataframe")

# 使用AI功能
import numpy as np

# 添加向量到向量数据库
vectors = np.random.rand(10, 128)  # 示例向量
metadata = [{"id": i, "text": f"sample_{i}"} for i in range(10)]
storage.add_vectors(vectors, metadata)

# 搜索相似向量
query_vector = np.random.rand(128)
results = storage.search_vectors(query_vector, top_k=5)
print(f"找到 {len(results)} 个相似结果")
```

### 数据分析与可视化算子 🆕
- **DataAnalysisOperator**: 全面的统计分析，包括描述性统计、增长率计算
- **DataSummaryOperator**: 数据摘要和质量评估，支持相关性分析
- **ChartGeneratorOperator**: 多种图表类型生成（线图、柱状图、散点图、饼图、热力图、箱线图）
- **DashboardGeneratorOperator**: 交互式仪表板生成，支持多图表组合
- **HTMLReportGeneratorOperator**: 专业HTML报告生成，支持多种模板样式
- **PDFReportGeneratorOperator**: PDF格式报告生成
- **ReportTemplateOperator**: 预定义报告模板（综合报告、执行摘要、技术报告）

### AI操作符使用

```python
from maestro.operators.ai_ops import TextAnalysisOperator, SentimentAnalysisOperator
from maestro.serving.llm_serving import APILLMServing

# 创建LLM服务
llm_serving = APILLMServing(
    api_key="your_api_key",
    model_name="gpt-3.5-turbo",
    base_url="https://api.openai.com/v1"
)

# 创建AI操作符
text_analysis_op = TextAnalysisOperator(
    llm_serving=llm_serving,
    analysis_type="keyword_extraction",
    target_column="content"
)

sentiment_op = SentimentAnalysisOperator(
    llm_serving=llm_serving,
    target_column="content",
    output_column="sentiment"
)

# 在管道中使用AI操作符
pipeline = Pipeline(storage=storage)
pipeline.add_operator(text_analysis_op)
pipeline.add_operator(sentiment_op)
pipeline.run()
```

### 直接使用存储系统

```python
from maestro.utils.storage import FileStorage

# 创建FileStorage实例
storage = FileStorage(
    input_file_path="path/to/your/data.xlsx",  # 支持xlsx、csv、json、jsonl
    cache_path="./cache",                       # 缓存目录
    file_name_prefix="example",                 # 缓存文件名前缀
    cache_type="xlsx"                           # 输出文件类型
)

# 初始化并读取数据
storage.step()
data = storage.read("dataframe")  # 或 "dict"

# 处理数据
processed_data = data[data['score'] > 80]

# 保存处理结果
result_path = storage.write(processed_data)
print(f"结果已保存至: {result_path}")
```

### 格式转换

```python
# 从XLSX读取
storage_in = FileStorage(
    input_file_path="data.xlsx",
    cache_path="./cache",
    file_name_prefix="original",
    cache_type="xlsx"
)

# 初始化并读取数据
storage_in.step()
data = storage_in.read("dataframe")

# 转换为JSON格式保存
storage_out = FileStorage(
    input_file_path="dummy.xlsx",  # 不会实际使用此文件
    cache_path="./cache",
    file_name_prefix="converted",
    cache_type="json"
)

json_path = storage_out.write(data)
print(f"已转换为JSON: {json_path}")
```

## 支持的文件类型

- **XLSX**: Excel文件格式
- **CSV**: 逗号分隔值文件
- **JSON**: JavaScript对象表示法
- **JSONL**: 每行一个JSON对象
- **Parquet**: 高效的列式存储格式
- **Pickle**: Python对象序列化格式

## 核心功能

MaestroDataflow 提供了丰富的数据处理功能：

### 存储系统
- **FileStorage**: 文件存储系统，支持多种格式，集成AI功能
- **DBStorage**: 数据库存储系统，支持SQLite等数据库
- **VectorStorage**: 向量数据库存储，支持相似性搜索

### 基础操作符
- **FilterRowsOperator**: 根据条件筛选数据行
- **SelectColumnsOperator**: 选择特定列
- **MapRowsOperator**: 对数据行应用自定义函数
- **AggregateOperator**: 数据聚合操作
- **SortOperator**: 数据排序操作

### AI操作符
- **TextAnalysisOperator**: 文本分析和关键词提取
- **SentimentAnalysisOperator**: 情感分析
- **DataCleaningOperator**: 智能数据清洗
- **EmbeddingOperator**: 文本向量化
- **SimilaritySearchOperator**: 相似性搜索

### LLM服务
- **APILLMServing**: API方式调用LLM服务（OpenAI、Azure等）
- **LocalLLMServing**: 本地LLM模型服务
- **EnhancedLLMServing**: 增强的LLM服务，支持缓存和重试

### 管道系统
- **Pipeline**: 数据处理管道，支持链式操作
- **BatchPipeline**: 批处理管道，支持大数据集处理
- **步骤管理**: 自动管理处理步骤和缓存
- **错误处理**: 完善的错误处理机制

## 项目结构

```
MaestroDataflow/
├── maestro/                    # 核心代码
│   ├── core/                   # 核心组件
│   │   ├── operator.py         # 操作符基类
│   │   ├── processor.py        # 数据处理器
│   │   └── prompt.py           # AI提示模板
│   ├── operators/              # 操作符
│   │   ├── basic_ops.py        # 基础操作符
│   │   ├── io_ops.py           # 输入输出操作符
│   │   ├── llm_ops.py          # LLM操作符
│   │   ├── analytics_ops.py    # 数据分析算子 🆕
│   │   ├── visualization_ops.py # 数据可视化算子 🆕
│   │   ├── report_ops.py       # 报告生成算子 🆕
│   │   └── ai_ops/             # AI操作符目录
│   │       ├── text_analysis.py      # 文本分析
│   │       ├── sentiment_analysis.py # 情感分析
│   │       ├── data_cleaning.py      # 数据清洗
│   │       └── intelligent_processing.py # 智能处理
│   ├── pipeline/               # 管道系统
│   │   ├── pipeline.py         # 管道实现
│   │   └── nodes.py            # 管道节点
│   ├── utils/                  # 工具类
│   │   ├── storage.py          # 文件存储
│   │   ├── db_storage.py       # 数据库存储
│   │   ├── vector_db.py        # 向量数据库
│   │   └── model_cache.py      # 模型缓存
│   └── serving/                # 服务组件
│       ├── llm_serving.py      # LLM服务
│       └── enhanced_llm_serving.py # 增强LLM服务
├── test/                       # 测试文件
│   ├── test_basic.py           # 基础功能测试
│   ├── test_storage.py         # 存储测试
│   ├── test_db_storage.py      # 数据库存储测试
│   ├── test_integration.py     # 集成测试
│   └── db/                     # 测试数据库
├── examples/                   # 示例代码
│   ├── advanced_pipeline_example.py    # 高级管道示例
│   ├── ai_operators_demo.py            # AI操作符演示
│   ├── comprehensive_ai_workflow.py    # 综合AI工作流
│   ├── digital_economy_analysis.py     # 数字经济数据分析示例 🆕
│   └── README_ANALYTICS.md             # 数据分析功能文档 🆕
├── docs/                       # 文档
│   ├── API_REFERENCE.md        # API参考
│   └── AI_OPERATORS_GUIDE.md   # AI操作符指南
├── sample_data/                # 示例数据
└── output/                     # 输出目录 🆕
    ├── advanced_pipeline_example/      # 高级管道示例输出
    ├── ai_operators_demo/              # AI操作符演示输出
    ├── comprehensive_ai_workflow/      # 综合工作流输出
    └── digital_economy_analysis/       # 数字经济分析输出
```

## 示例

查看 `examples/` 目录获取完整的使用示例，包括：

- **advanced_pipeline_example.py**: 高级管道处理示例，展示Pipeline框架、数据库存储和AI模型服务的集成使用
- **ai_operators_demo.py**: AI操作符功能演示，展示各种AI操作符的使用方法和功能
- **comprehensive_ai_workflow.py**: 完整的AI数据处理工作流，展示复杂的AI数据处理场景
- **digital_economy_analysis.py**: 数字经济数据分析完整示例，演示数据分析、可视化和报告生成功能 
- **integrated_column_processing_workflow.py**: 整合的三步骤列处理工作流，展示列模板生成、数据处理和LLM填充的完整流程 🆕

### 文档说明
- **README_AI_OPERATORS.md**: AI操作符详细使用指南
- **README_ANALYTICS.md**: 数据分析和可视化功能详细文档 🆕
- **README_DATA_COLUMN_PROCESS.md**: 数据列处理功能说明文档

所有示例脚本运行后的结果文件（图表、报告、数据等）都会统一保存到 `output/` 目录的相应子目录中，保持 `examples/` 目录的整洁。

## 测试

运行测试套件：

```bash
# 运行所有测试
pytest test/

# 运行特定测试
pytest test/test_basic.py -v

# 查看测试覆盖率
pytest test/ --cov=maestro
```

## 许可证

MIT License