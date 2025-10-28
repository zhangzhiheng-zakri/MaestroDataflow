# MaestroDataflow 数据列处理功能

## 概述

数据列处理功能是MaestroDataflow的新增功能，它整合了原有data_column_process的核心能力，但使用MaestroDataflow的方法进行数据清洗。主要功能包括：

1. **数据存储到数据库** - 将处理后的数据存储到SQL数据库
2. **生成列名意义单位** - 使用AI生成数据列名的中文含义、单位和说明
3. **数据清洗** - 使用MaestroDataflow的方法进行数据预处理
4. **JSON格式输出** - 输出标准化的列名意义单位JSON格式

## 核心组件

### 1. ColumnMeaningGeneratorOperator (列名意义生成器)

基于process_column_name的prompt功能，使用AI为数据列生成详细的中文解释。

**功能特点：**
- 支持批量处理列名
- 生成中文名称、详细含义、单位信息
- 推断数据类型和可能取值
- 输出标准JSON格式

**使用示例：**
```python
from maestro.operators.column_ops import ColumnMeaningGeneratorOperator

op = ColumnMeaningGeneratorOperator(
    dataset_description="用户行为调研数据集",
    max_columns_per_batch=10,
    service=llm_service
)
```

### 2. ColumnMetadataExtractorOperator (列元数据提取器)

从数据中提取列的统计信息和元数据。

**功能特点：**
- 提取数据类型、缺失值统计
- 计算数值型列的统计指标
- 获取分类型列的唯一值信息
- 生成完整的数据概览

### 3. DataColumnProcessOperator (数据列处理器)

完整的数据列处理流程操作符，整合所有功能。

**功能特点：**
- 数据清洗（删除缺失值过多的列、去重等）
- 缺失值填充（支持多种填充方法）
- 数据库存储
- AI生成列名意义
- 输出完整的JSON报告

**使用示例：**
```python
from maestro.operators.data_column_process_ops import DataColumnProcessOperator

column_processor = DataColumnProcessOperator(
    dataset_name="用户行为数据",
    dataset_description="包含用户基本信息和行为数据的调研数据集",
    db_connection_string="sqlite:///data.db",
    service=llm_service
)
```

### 4. QuickDataColumnProcessOperator (快速数据列处理器)

简化版本，主要用于快速生成列名意义单位的JSON。

## 主要改进

相比原有的data_column_process，新的数据列处理器具有以下优势：

1. **集成化设计** - 完全集成到MaestroDataflow框架中
2. **管道化处理** - 支持与其他操作符组合使用
3. **标准化存储** - 使用MaestroDataflow的存储系统
4. **灵活配置** - 支持多种数据库和LLM服务
5. **错误处理** - 更好的异常处理和回退机制

## 输出格式

数据列处理器输出标准的JSON格式，包含以下信息：

```json
{
  "dataset_info": {
    "name": "数据集名称",
    "description": "数据集描述",
    "total_rows": 1000,
    "total_columns": 10,
    "processing_date": "2024-01-01T12:00:00",
    "database_table": "表名"
  },
  "column_meanings": [
    {
      "column_name": "age",
      "chinese_name": "年龄",
      "meaning": "用户的年龄，以年为单位",
      "unit": "年",
      "data_type": "数值型",
      "possible_values": "通常在18-65岁之间"
    }
  ],
  "column_metadata": [
    {
      "column_name": "age",
      "data_type": "int64",
      "non_null_count": 950,
      "null_count": 50,
      "null_percentage": 5.0,
      "min_value": 18,
      "max_value": 65,
      "mean_value": 35.2
    }
  ],
  "processing_summary": {
    "original_shape": [1000, 12],
    "final_shape": [950, 10],
    "na_threshold_used": 0.5,
    "fill_method_used": "median",
    "columns_dropped": 2,
    "rows_dropped": 50
  }
}
```

## 使用方法

### 基本使用

```python
from maestro.pipeline import Pipeline
from maestro.utils.storage import FileStorage
from maestro.operators.data_column_process_ops import DataColumnProcessOperator
from maestro.serving.llm_serving import APILLMServing

# 设置存储和LLM服务
storage = FileStorage(base_path="output")
llm_service = APILLMServing(api_key="your-key", model="gpt-3.5-turbo")

# 创建管道
pipeline = Pipeline(storage=storage)

# 添加数据列处理器
column_processor = DataColumnProcessOperator(
    dataset_name="我的数据集",
    dataset_description="数据集的详细描述",
    db_connection_string="sqlite:///my_data.db",
    service=llm_service
)

pipeline.add_operator("column_processor", column_processor)

# 执行
result = pipeline.run(
    na_threshold=0.3,
    fill_method="median",
    llm_service=llm_service
)
```

### 与其他操作符结合

```python
from maestro.operators.basic_ops import FilterRowsOperator, SelectColumnsOperator

# 数据预处理
filter_op = FilterRowsOperator(lambda df: df['age'] >= 18)
select_op = SelectColumnsOperator(['age', 'income', 'education'])

# 添加到管道
pipeline.add_operator("filter", filter_op)
pipeline.add_operator("select", select_op)
pipeline.add_operator("column_processor", column_processor)

# 执行完整流程
result = pipeline.run()
```

## 配置要求

1. **LLM服务** - 需要配置有效的LLM服务（OpenAI API或本地服务）
2. **数据库** - 支持SQLAlchemy兼容的数据库
3. **依赖包** - pandas, sqlalchemy, openpyxl等

## 示例文件

参考 `examples/integrated_column_processing_workflow.py` 获取完整的使用示例。

该示例展示了整合的三步骤列处理工作流：
1. 使用 `ColumnTemplateGeneratorOperator` 生成空列模板
2. 使用 `DataColumnProcessOperator` 进行数据处理
3. 使用 `ColumnMeaningGeneratorOperator` 通过LLM填充JSON

运行示例：
```bash
cd examples
python integrated_column_processing_workflow.py
```

输出文件将保存到 `../output/integrated_column_processing_workflow/` 目录。

## 注意事项

1. 确保LLM服务配置正确，否则列名意义生成会失败
2. 数据库连接字符串需要有效，确保有写入权限
3. 大数据集处理时注意内存使用
4. AI生成的列名解释可能需要人工审核和调整