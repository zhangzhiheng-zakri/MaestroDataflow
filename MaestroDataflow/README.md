# MaestroDataflow

MaestroDataflow 是一个现代化的AI增强数据处理框架，专为高效处理多种格式的数据文件而设计。它不仅提供了统一的接口来处理 XLSX、CSV、JSON、JSONL 和 Parquet 格式的数据，还集成了先进的AI功能，包括向量数据库、模型缓存、智能数据处理操作符、数据分析与可视化等全方位的数据科学解决方案。

**版本**: 1.0.0  
**Python要求**: 3.8+  
**许可证**: MIT

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

### 基础安装

```bash
pip install maestro-dataflow
```

### 完整安装（包含所有AI依赖）

```bash
pip install maestro-dataflow[full]
```

### 开发安装

```bash
git clone https://github.com/maestro-dataflow/MaestroDataflow.git
cd MaestroDataflow
pip install -e .[dev,full]
```

### 依赖要求

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.20.0
- openpyxl >= 3.0.0 (Excel文件支持)
- 其他依赖详见 `requirements.txt`

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

### 最新更新（输入与命名约定）
- 输入文件目录统一为 `input/datasets/`，示例脚本会自动递归查找，优先选择 `.xlsx`，其次 `.csv`。会忽略临时/隐藏文件（如以 `~$`、`.`、`._` 开头）。
- 可通过环境变量指定输入文件：`MAESTRO_INPUT_FILE`，但路径必须位于 `input/datasets/` 内。
- 标准化打包时生成的类名与类文件名使用英文简称：`Dataset<ShortSlug>`。简称规则为：
  - 先将源文件名转为 ASCII slug（移除非字母数字字符）。
  - 若存在 PascalCase 大写分词，取所有大写字母作为缩写（长度≥3优先）。
  - 否则截取前 10 个字母，并确保首字母大写；无字母则回退为 `Dataset`。
- 集成示例将数据库表名也统一使用类名形式（例如 `DatasetLCEC`），与类名保持一致，便于代码和数据库工具统一引用；中文显示名称保留为源文件名。

- 增量打包：新增数据集时，无需重跑已打包的数据集。整合示例仅处理当前输入文件并在 `output/datasets/` 下新增对应目录和类文件，不影响已生成的数据集。若输出目录同名已存在，默认会覆盖同名的 CSV/JSON/类文件；如需保留旧版本，请调整源文件名或自定义输出路径。

运行整合示例（清洗→入库→打包）：

```bash
python -m examples.integrated_packaging_workflow
```

可选环境变量：
- `DEEPSEEK_API_KEY`：启用 LLM 生成数据集简介及辅助英文简称；未设置时将使用占位/回退逻辑继续执行。
- `MAESTRO_INPUT_FILE`：指定 `input/datasets/` 内的具体文件路径（必须位于该目录）。

输出结构示例：
- `dataset_dir`: `output/datasets/上市公司能源消耗数据（2012-2024年）`
- `data_path`: `output/datasets/上市公司能源消耗数据（2012-2024年）/上市公司能源消耗数据（2012-2024年）.csv`
- `all_column_name`: `output/datasets/上市公司能源消耗数据（2012-2024年）/all_column_name.json`
- `class_file`: `output/datasets/DatasetLCEC.py`
- `class_name`: `DatasetLCEC`
- 类文件中的 `name`: 中文源名称；`info`: 若启用 LLM，将自动生成不超过 120 字的中文简介。

说明：打包流程仅生成列名意义 JSON 文件 `all_column_name.json`，类文件的 `columns_path` 指向该文件；不再生成或依赖 `column_template.json`。

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

#### API密钥配置与安全
- 强烈建议通过环境变量注入密钥，避免将真实密钥写入代码或提交到版本库。
- 支持的环境变量包括：`OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`AZURE_OPENAI_API_KEY`。

推荐用法示例：

```python
import os
from maestro.serving.llm_serving import APILLMServing

# 优先从环境变量读取密钥
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("未设置 API 密钥，请配置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")

llm_serving = APILLMServing(
    api_key=api_key,
    model_name="gpt-3.5-turbo",
    base_url="https://api.openai.com/v1"
)
```

在 Windows 终端设置环境变量：
- 临时当前会话：`$env:DEEPSEEK_API_KEY="你的密钥"`
- 永久（重启终端后生效）：`setx DEEPSEEK_API_KEY "你的密钥"`

如曾误提交真实密钥，请在对应平台及时“旋转/重置密钥”。

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
├── maestro/                    # 核心框架代码
│   ├── __init__.py
│   ├── core/                   # 核心组件
│   │   ├── __init__.py
│   │   ├── base.py            # 基础类定义
│   │   └── pipeline.py        # 管道系统
│   ├── utils/                  # 工具模块
│   │   ├── __init__.py
│   │   ├── storage.py         # 存储系统
│   │   └── ai_utils.py        # AI工具
│   ├── operators/              # 操作符
│   │   ├── __init__.py
│   │   ├── base_ops.py        # 基础操作符
│   │   ├── io_ops.py          # 输入输出操作符
│   │   ├── transform_ops.py   # 数据转换操作符
│   │   ├── ai_ops.py          # AI操作符
│   │   ├── analysis_ops.py    # 数据分析操作符
│   │   └── visualization_ops.py # 可视化操作符
│   └── services/               # 服务模块
│       ├── __init__.py
│       ├── llm_service.py     # LLM服务
│       └── vector_service.py  # 向量服务
├── examples/                   # 示例代码
│   ├── README_AI_OPERATORS.md
│   ├── basic_pipeline_example.py
│   ├── ai_pipeline_example.py
│   ├── digital_economy_pipeline_example.py
│   └── visualization_example.py
├── test/                       # 测试代码
│   ├── README.md
│   ├── test_storage.py
│   ├── test_operators.py
│   ├── test_ai_features.py
│   └── test_pipeline.py
├── sample_data/                # 示例数据
│   ├── README.md
│   ├── employees.csv
│   ├── sales_data.json
│   └── 中国数字经济发展数据（2005-2023年）.xlsx
├── docs/                       # 文档
├── output/                     # 输出目录
├── setup.py                    # 安装配置
├── requirements.txt            # 依赖列表
└── README.md                   # 项目说明
```

## 示例

### 基础示例

查看 `examples/` 目录中的示例代码：

- **basic_pipeline_example.py** - 基础数据处理管道
- **ai_pipeline_example.py** - AI增强数据处理
- **digital_economy_pipeline_example.py** - 数字经济数据分析
- **visualization_example.py** - 数据可视化示例

### 运行示例

```bash
# 运行基础管道示例
python examples/basic_pipeline_example.py

# 运行AI管道示例
python examples/ai_pipeline_example.py

# 运行数字经济分析示例
python examples/digital_economy_pipeline_example.py
```

## 测试

运行测试套件：

```bash
# 运行所有测试
python -m pytest test/

# 运行特定测试文件
python -m pytest test/test_storage.py

# 显示测试覆盖率
python -m pytest test/ --cov=maestro --cov-report=html
```

详细测试说明请参考 `test/README.md`。

## 贡献

我们欢迎社区贡献！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 联系方式

- 项目主页: https://github.com/maestro-dataflow/MaestroDataflow
- 问题反馈: https://github.com/maestro-dataflow/MaestroDataflow/issues
- 邮箱: maestro@dataflow.ai

## 更新日志

### v1.0.0 (2024-01-15)
- 🎉 首次发布
- ✨ 支持多种数据格式 (XLSX, CSV, JSON, JSONL, Parquet)
- 🤖 集成AI功能 (向量数据库, 模型缓存, AI操作符)
- 📊 数据分析与可视化功能
- 🔧 完整的管道系统
- 📚 丰富的示例和文档