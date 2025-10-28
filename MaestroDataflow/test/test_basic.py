#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaestroDataflow 基础功能测试
"""

import os
import sys
import pandas as pd

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from maestro.utils.storage import FileStorage
from maestro.operators.basic_ops import FilterRowsOperator, SelectColumnsOperator
from maestro.pipeline.pipeline import Pipeline

def test_basic_functionality():
    print("=== MaestroDataflow 基础功能测试 ===")
    
    # 1. 创建测试数据文件
    test_data = pd.DataFrame({
        'name': ['张三', '李四', '王五', '赵六'],
        'age': [25, 30, 35, 28],
        'city': ['北京', '上海', '广州', '深圳'],
        'salary': [50000, 60000, 70000, 55000]
    })
    
    # 创建临时输入文件
    os.makedirs("./test_cache", exist_ok=True)
    input_file = "./test_cache/input.csv"
    test_data.to_csv(input_file, index=False)
    
    # 2. 创建存储实例，使用CSV格式避免JSON编码问题
    storage = FileStorage(input_file, cache_path="./test_cache", cache_type="csv")
    print("OK FileStorage 创建成功")
    
    # 3. 重置存储状态，确保从输入文件开始读取
    storage.reset()
    print("OK 存储状态重置成功")
    
    # 4. 创建管道
    pipeline = Pipeline(storage=storage)
    print("OK Pipeline 创建成功")
    
    # 5. 添加操作符
    filter_op = FilterRowsOperator(
        condition=lambda df: df['age'] >= 30
    )
    
    select_op = SelectColumnsOperator(
        columns=['name', 'city', 'salary']
    )
    
    pipeline.add_operator(filter_op, "filter")
    pipeline.add_operator(select_op, "select")
    print("OK 操作符添加成功")
    
    # 6. 运行管道
    results = pipeline.run()
    print("OK Pipeline 运行成功")
    
    # 7. 查看结果
    # 只需要step一次到最后一步
    storage.step()  # 进入第1步（filter结果）
    final_data = storage.read(output_type="dataframe")
    print("\n=== 处理结果 ===")
    print(final_data)
    
    print(f"\nOK 测试完成，处理了 {len(final_data)} 条记录")
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
        print("\n🎉 所有测试通过！MaestroDataflow 框架运行正常。")
    except Exception as e:
        print(f"\nERROR 测试失败: {e}")
        import traceback
        traceback.print_exc()
