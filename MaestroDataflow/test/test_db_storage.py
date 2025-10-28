#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DBStorage 功能测试
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from maestro.utils.db_storage import DBStorage
from maestro.operators.basic_ops import FilterRowsOperator, SelectColumnsOperator
from maestro.pipeline.pipeline import Pipeline

def test_db_storage_functionality():
    print("=== DBStorage 功能测试 ===")
    
    # 1. 创建测试数据
    test_data = pd.DataFrame({
        'name': ['张三', '李四', '王五', '赵六', '钱七'],
        'age': [25, 30, 35, 28, 32],
        'city': ['北京', '上海', '广州', '深圳', '杭州'],
        'salary': [50000, 60000, 70000, 55000, 65000]
    })
    
    # 2. 创建SQLite数据库存储实例
    db_storage = DBStorage(
        connection_string="sqlite:///test/db/test_maestro.db",
        table_name="test_data"
    )
    print("OK DBStorage 创建成功")
    
    # 3. 重置存储状态
    db_storage.reset()
    print("OK 存储状态重置成功")
    
    # 4. 写入初始数据
    result_path = db_storage.write(test_data)
    print(f"OK 数据写入成功: {result_path}")
    
    # 5. 读取数据验证
    db_storage_next = db_storage.step()
    read_data = db_storage_next.read(output_type="dataframe")
    print(f"OK 数据读取成功，形状: {read_data.shape}")
    print("读取的数据:")
    print(read_data.head())
    
    # 6. 测试Pipeline集成
    pipeline = Pipeline(storage=db_storage_next)
    print("OK Pipeline 创建成功")
    
    # 7. 添加操作符
    filter_op = FilterRowsOperator(
        lambda df: df['age'] > 27  # 筛选年龄大于27的记录
    )
    select_op = SelectColumnsOperator(['name', 'city', 'salary'])
    
    pipeline.add_operator(filter_op, "filter")
    pipeline.add_operator(select_op, "select")
    print("OK 操作符添加成功")
    
    # 8. 运行管道
    results = pipeline.run()
    print("OK Pipeline 运行成功")
    
    # 9. 查看最终结果
    db_storage_final = db_storage_next.step()  # 进入最后一步
    final_data = db_storage_final.read(output_type="dataframe")
    print("\n=== 处理结果 ===")
    print(final_data)
    
    print(f"\nOK 测试完成，处理了 {len(final_data)} 条记录")
    
    # 10. 清理测试数据库
    try:
        db_storage.engine.dispose()
        if os.path.exists("test/db/test_maestro.db"):
            os.remove("test/db/test_maestro.db")
        print("OK 测试数据库清理成功")
    except Exception as cleanup_error:
        print(f"清理警告: {cleanup_error}")
    
    print("\n🎉 DBStorage 测试通过！")
    return True

if __name__ == "__main__":
    try:
        test_db_storage_functionality()
    except Exception as e:
        print(f"\nERROR 测试失败: {e}")
        import traceback
        traceback.print_exc()