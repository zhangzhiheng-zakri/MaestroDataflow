# MaestroDataflow 测试套件

这是 MaestroDataflow 项目的完整测试套件，包含基础测试、集成测试和 AI 功能测试。

**支持的Python版本**: 3.8+  
**测试框架**: pytest  
**覆盖率工具**: pytest-cov

## 📋 测试结构

### 🔧 基础测试
- **test_storage.py** - 存储系统测试
  - 文件读写功能
  - 多格式支持 (XLSX, CSV, JSON, JSONL, Parquet)
  - 缓存机制
  - 错误处理

### 🔗 集成测试  
- **test_operators.py** - 操作符测试
  - 基础操作符功能
  - 数据转换操作符
  - 输入输出操作符
  - 操作符链式调用

### 🤖 AI功能测试
- **test_ai_features.py** - AI功能测试
  - 文本分析操作符
  - 情感分析操作符
  - 向量存储功能
  - 模型缓存机制
  - LLM服务集成

### 🚀 管道测试
- **test_pipeline.py** - 管道系统测试
  - 管道构建和执行
  - 错误处理和恢复
  - 性能测试
  - 并发处理

## 测试文件说明

### 基础测试
- `test_basic.py` - 基础功能测试，验证核心组件的基本功能
- `test_storage_edge_cases.py` - 存储系统边界情况测试，验证错误处理和异常情况
- `test_db_storage.py` - 数据库存储功能测试，包括SQLite数据库操作

### 集成测试
- `test_integration.py` - 集成测试，验证各组件之间的协作，包括管道系统和存储系统的集成

### AI功能测试
目前AI操作符的测试主要通过示例代码进行验证：
- `../examples/ai_operators_demo.py` - AI操作符功能演示和测试
- `../examples/advanced_pipeline_example.py` - 高级管道示例和测试
- `../examples/digital_economy_analysis.py` - 数字经济分析示例
- `../examples/integrated_column_processing_workflow.py` - 整合列处理工作流示例

### 配置文件
- `conftest.py` - pytest 配置文件，提供通用的 fixture 和配置
- `__init__.py` - 测试模块初始化文件

## 🚀 运行测试

### 运行所有测试

```bash
# 基础运行
python -m pytest test/

# 详细输出
python -m pytest test/ -v

# 显示测试进度
python -m pytest test/ --tb=short
```

### 运行特定测试文件

```bash
# 运行存储系统测试
python -m pytest test/test_storage.py

# 运行操作符测试
python -m pytest test/test_operators.py

# 运行AI功能测试
python -m pytest test/test_ai_features.py

# 运行管道测试
python -m pytest test/test_pipeline.py
```

### 运行特定测试类或方法

```bash
# 运行特定测试类
python -m pytest test/test_storage.py::TestFileStorage

# 运行特定测试方法
python -m pytest test/test_storage.py::TestFileStorage::test_read_csv

# 运行匹配模式的测试
python -m pytest test/ -k "storage"
```

### 显示详细输出和测试覆盖率

```bash
# 显示测试覆盖率
python -m pytest test/ --cov=maestro

# 生成HTML覆盖率报告
python -m pytest test/ --cov=maestro --cov-report=html

# 显示缺失的行
python -m pytest test/ --cov=maestro --cov-report=term-missing

# 设置覆盖率阈值
python -m pytest test/ --cov=maestro --cov-fail-under=80
```

## 📊 测试配置

### pytest.ini 配置

```ini
[tool:pytest]
testpaths = test
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    ai: marks tests as AI functionality tests
```

### 环境变量

测试时可能需要设置以下环境变量：

```bash
# OpenAI API密钥（用于AI功能测试）
export OPENAI_API_KEY="your-api-key"

# 测试数据目录
export TEST_DATA_DIR="test/data"

# 启用详细日志
export MAESTRO_LOG_LEVEL="DEBUG"
```

### 运行AI功能演示测试
```bash
cd examples

# AI操作符演示（输出到 ../output/ai_operators_demo/）
python ai_operators_demo.py

# 综合AI工作流（输出到 ../output/comprehensive_ai_workflow/）
python comprehensive_ai_workflow.py

# 高级管道示例（输出到 ../output/advanced_pipeline_example/）
python advanced_pipeline_example.py

# 数字经济分析示例（输出到 ../output/digital_economy_analysis/）
python digital_economy_analysis.py

# 整合列处理工作流示例（输出到 ../output/integrated_column_processing_workflow/）
python integrated_column_processing_workflow.py

说明：整合打包流程输出的列名说明文件为 `all_column_name.json`，不再生成或依赖 `column_template.json`。
```

## 测试数据

### 基础测试数据
测试使用的样本数据包含以下字段：
- `name`: 姓名
- `city`: 城市  
- `salary`: 薪资

### AI测试数据
AI功能测试使用的数据包括：
- `../sample_data/employees.csv` - 员工数据，包含AI相关文本内容
- `../output/ai_operators_demo/cache/` - AI操作符演示缓存目录
  - `demo_cache_1.csv` - AI查询缓存数据
  - `demo_cache_2.csv` - 嵌入向量缓存数据
  - `model_cache/` - 模型缓存目录
- `../output/advanced_pipeline_example/` - 高级管道示例输出目录
- `../output/digital_economy_analysis/` - 数字经济分析输出目录
- `../output/integrated_column_processing_workflow/` - 整合列处理工作流输出目录
测试会自动创建和清理临时文件，所有输出文件统一保存到 `../output/` 目录的相应子目录中。

## 数据库文件

测试过程中生成的数据库文件统一存放在 `db/` 子目录中：

- `db/test_integration.db` - 集成测试数据库
- `db/test_compatibility.db` - 兼容性测试数据库  
- `db/test_maestro.db` - DBStorage功能测试数据库

## 测试环境配置

### AI功能测试环境
运行AI功能测试需要配置以下环境：

1. **LLM服务配置**：
   - 设置 `OPENAI_API_KEY` 环境变量（用于API服务）
   - 或安装本地模型依赖（用于本地服务）

2. **向量数据库依赖**：
   - 安装 `sentence-transformers` 用于嵌入生成
   - 安装 `numpy` 用于向量计算

3. **可选依赖**：
   - `torch` - 用于深度学习模型
   - `transformers` - 用于Transformer模型

### 环境变量
```bash
# OpenAI API配置
export OPENAI_API_KEY="your-api-key-here"

# 可选：Azure OpenAI配置
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="your-azure-endpoint"
```

## 注意事项

1. **文件清理**：所有测试都使用临时目录和文件，测试完成后会自动清理
2. **数据库管理**：数据库文件统一管理在 `test/db/` 目录中，测试后会自动清理
3. **缓存处理**：缓存目录会在测试前后自动清理
4. **导入路径**：测试文件已配置正确的导入路径，可以直接运行
5. **异常清理**：如果测试异常中断，可能需要手动清理 `db/` 目录中的残留文件
6. **AI功能测试**：AI相关功能的测试需要网络连接或本地模型，可能需要较长时间
7. **资源使用**：AI功能测试可能消耗较多内存和计算资源，建议在性能较好的机器上运行

## 测试覆盖范围

### 已覆盖功能
- ✅ 基础存储系统（FileStorage, DBStorage）
- ✅ 基础操作符（FilterRows, SelectColumns, MapRows等）
- ✅ 管道系统（Pipeline）
- ✅ 数据格式转换（CSV, JSON, XLSX, Parquet等）
- ✅ 错误处理和边界情况

### 待完善测试
- 🔄 AI操作符单元测试
- 🔄 向量数据库功能测试
- 🔄 LLM服务单元测试
- 🔄 模型缓存功能测试
- 🔄 多模态处理测试

## 贡献测试

如需添加新的测试用例：

1. **单元测试**：在相应的 `test_*.py` 文件中添加测试方法
2. **集成测试**：在 `test_integration.py` 中添加组件协作测试
3. **AI功能测试**：在 `examples/` 目录中添加演示和验证代码
4. **测试数据**：将测试数据放在适当的目录中，确保测试后清理

### 测试命名规范
- 测试文件：`test_*.py`
- 测试类：`Test*`
- 测试方法：`test_*`
- AI演示：`*_demo.py` 或 `*_example.py`