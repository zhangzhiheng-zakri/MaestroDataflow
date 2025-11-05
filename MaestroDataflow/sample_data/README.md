# MaestroDataflow 示例数据

本目录包含用于测试和演示 MaestroDataflow 功能的示例数据文件。

**支持的数据格式**: CSV, JSON, XLSX  
**数据类型**: 示例数据  
**状态**: 即用型这些数据文件涵盖了不同的格式和用例，帮助用户快速上手和测试框架的各种功能。

## 📁 数据文件

### 📊 employees.csv
**用途**: 员工数据分析、文本分析和AI操作符测试

**格式**: CSV (逗号分隔值)

**内容**: 
- 员工基本信息 (姓名、部门、职位)
- 薪资数据
- 工作描述和评价文本
- 适用于文本分析、情感分析等AI功能测试

**字段说明**:
- `name`: 员工姓名
- `department`: 部门
- `position`: 职位
- `salary`: 薪资
- `description`: 工作描述
- `performance_review`: 绩效评价

### 💰 sales_data.json
**用途**: 销售数据分析、数据转换和可视化测试

**格式**: JSON (JavaScript Object Notation)

**内容**:
- 销售交易记录
- 产品信息和分类
- 时间序列数据
- 地理位置信息
- 适用于数据分析、趋势分析和可视化功能

**数据结构**:
```json
{
  "transactions": [
    {
      "id": "string",
      "date": "YYYY-MM-DD",
      "product": "string",
      "category": "string",
      "amount": "number",
      "region": "string",
      "customer_id": "string"
    }
  ]
}
```

### 🇨🇳 中国数字经济发展数据（2005-2023年）.xlsx
**用途**: 经济数据分析、时间序列分析和高级可视化

**格式**: XLSX (Excel工作簿)

**内容**:
- 中国数字经济发展指标 (2005-2023年)
- 多维度经济数据
- 时间序列分析数据
- 适用于复杂数据分析、统计建模和报告生成

**数据维度**:
- 年份: 2005-2023
- 指标: 数字经济规模、增长率、占GDP比重等
- 地区: 全国及主要省市数据
- 行业: 各行业数字化程度指标

## 🚀 使用示例

### 基础数据加载

```python
from maestro.utils.storage import FileStorage

# 加载CSV文件
csv_storage = FileStorage(input_file_path="sample_data/employees.csv")
csv_data = csv_storage.read()
print(f"CSV数据形状: {csv_data.shape}")

# 加载JSON文件
json_storage = FileStorage(input_file_path="sample_data/sales_data.json")
json_data = json_storage.read()
print(f"JSON数据形状: {json_data.shape}")

# 加载Excel文件
excel_storage = FileStorage(input_file_path="sample_data/中国数字经济发展数据（2005-2023年）.xlsx")
excel_data = excel_storage.read()
print(f"Excel数据形状: {excel_data.shape}")
```

### AI操作符测试

```python
from maestro.operators.ai_ops import TextAnalysisOperator, SentimentAnalysisOperator
from maestro.core.pipeline import Pipeline

# 使用员工数据进行文本分析
storage = FileStorage(input_file_path="sample_data/employees.csv")

# 创建AI操作符
text_analyzer = TextAnalysisOperator(
    input_column="description",
    output_column="text_analysis"
)

sentiment_analyzer = SentimentAnalysisOperator(
    input_column="performance_review",
    output_column="sentiment_score"
)

# 构建和运行管道
pipeline = Pipeline([text_analyzer, sentiment_analyzer])
result = pipeline.run(storage)
```

### 数据分析示例

```python
from maestro.operators.analysis_ops import StatisticalAnalysisOperator
from maestro.operators.visualization_ops import ChartGeneratorOperator

# 使用数字经济数据进行分析
storage = FileStorage(input_file_path="sample_data/中国数字经济发展数据（2005-2023年）.xlsx")

# 统计分析
stats_analyzer = StatisticalAnalysisOperator(
    columns=["数字经济规模", "增长率"],
    output_column="statistics"
)

# 图表生成
chart_generator = ChartGeneratorOperator(
    chart_type="line",
    x_column="年份",
    y_column="数字经济规模",
    output_path="output/digital_economy_trend.png"
)

# 执行分析
pipeline = Pipeline([stats_analyzer, chart_generator])
result = pipeline.run(storage)
```

## 📋 数据质量

### 数据完整性
- ✅ 所有文件都经过验证，确保格式正确
- ✅ 数据字段完整，无缺失关键信息
- ✅ 编码格式统一 (UTF-8)

### 数据规模
- **employees.csv**: ~100行员工记录
- **sales_data.json**: ~500条销售交易
- **中国数字经济发展数据.xlsx**: 14年×16指标的时间序列数据

### 更新频率
- 示例数据定期更新以反映最新的使用场景
- 数据结构保持向后兼容
- 新增数据文件会在版本更新中说明

## 🔧 自定义数据

### 添加新的示例数据

1. 将数据文件放入 `sample_data/` 目录
2. 确保文件格式符合 MaestroDataflow 支持的格式
3. 更新本 README 文件，添加数据描述
4. 在相应的示例代码中添加使用案例

### 支持的数据格式

- **CSV**: 逗号分隔值文件
- **JSON**: JavaScript对象表示法文件
- **JSONL**: 每行一个JSON对象
- **XLSX**: Excel工作簿文件
- **Parquet**: 列式存储格式文件

## 📞 获取帮助

如果在使用示例数据时遇到问题，请：

1. 查看主项目的 [README.md](../README.md)
2. 参考 [examples/](../examples/) 目录中的示例代码
3. 查看 [test/](../test/) 目录中的测试用例
4. 在 GitHub 上提交 [Issue](https://github.com/maestro-dataflow/MaestroDataflow/issues)

## Data Characteristics

- **Language**: Mixed Chinese and English content
- **Size**: Small datasets suitable for testing and demos
- **Quality**: Clean, well-structured data with realistic content
- **AI Content**: Includes descriptions and text suitable for NLP testing

## Adding New Sample Data

When adding new sample data files:
1. Use realistic but anonymized data
2. Include appropriate metadata and descriptions
3. Ensure data is suitable for AI/ML testing scenarios
4. Update this README with file descriptions