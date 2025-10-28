# 测试数据库文件夹

这个文件夹用于存放测试过程中生成的SQLite数据库文件。

## 文件说明

- `test_integration.db` - 集成测试使用的数据库，包含管道系统和存储系统的集成测试数据
- `test_compatibility.db` - 存储兼容性测试使用的数据库，验证不同数据格式的兼容性
- `test_maestro.db` - DBStorage功能测试使用的数据库，测试数据库存储的基本功能

## 数据库结构

### test_integration.db
- 包含测试用的员工数据表
- 用于验证FileStorage和DBStorage之间的数据转换
- 测试管道系统的数据流处理

### test_compatibility.db
- 测试不同数据类型的存储和读取
- 验证CSV、JSON、XLSX、Parquet等格式的兼容性
- 包含边界情况和异常数据的测试

### test_maestro.db
- 基础DBStorage功能测试
- 数据库连接和事务处理测试
- SQL查询和数据操作测试

## 注意事项

- 这些数据库文件仅用于测试，可以安全删除
- 测试运行时会自动创建和清理这些文件
- 如果测试异常中断，可能需要手动清理残留的数据库文件
- 数据库文件大小通常很小，主要包含测试样本数据
- 不要将这些文件提交到版本控制系统中
