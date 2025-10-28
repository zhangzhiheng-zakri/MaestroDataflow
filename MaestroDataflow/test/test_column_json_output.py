#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试列名处理的JSON输出功能
"""

import pandas as pd
import json
from maestro.utils.storage import FileStorage
from maestro.operators.column_ops import ColumnMeaningGeneratorOperator

# 创建一个简单的模拟LLM服务
class MockLLMService:
    def generate(self, prompt: str, **kwargs) -> str:
        """模拟LLM响应，返回JSON格式的列名解释"""
        # 根据prompt中的列名返回相应的JSON数组
        responses = []
        
        if "age" in prompt:
            responses.append({"意义": "用户的年龄，表示从出生到现在的时间长度", "单位": "年"})
        if "income" in prompt:
            responses.append({"意义": "用户的年收入，表示一年内获得的总收入", "单位": "元"})
        if "satisfaction_score" in prompt:
            responses.append({"意义": "用户满意度评分，反映用户对服务或产品的满意程度", "单位": "分"})
        if "city_code" in prompt:
            responses.append({"意义": "城市代码，用于标识不同城市的编码", "单位": "没有单位"})
        
        # 如果没有匹配的列名，返回默认响应
        if not responses:
            responses = [{"意义": "待人工补充说明", "单位": "没有单位"}]
        
        # 返回JSON字符串
        import json
        return json.dumps(responses, ensure_ascii=False)

def create_test_data():
    """创建测试数据"""
    data = {
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'satisfaction_score': [4.2, 3.8, 4.5, 4.1, 3.9],
        'city_code': ['BJ', 'SH', 'GZ', 'SZ', 'HZ']
    }
    return pd.DataFrame(data)

def main():
    print("=== 测试列名处理JSON输出 ===")
    
    # 创建测试数据
    df = create_test_data()
    print(f"测试数据列名: {list(df.columns)}")
    
    # 设置存储 - 使用正确的FileStorage初始化方式
    import tempfile
    import os
    
    # 创建临时输入文件
    temp_dir = "output/json_test"
    os.makedirs(temp_dir, exist_ok=True)
    temp_input_file = os.path.join(temp_dir, "temp_input.csv")
    df.to_csv(temp_input_file, index=False)
    
    # 初始化FileStorage
    storage = FileStorage(
        input_file_path=temp_input_file,
        cache_path=temp_dir,
        file_name_prefix="test_cache"
    )
    
    # 设置LLM服务（使用模拟服务）
    llm_service = MockLLMService()
    
    # 创建列名意义生成器
    column_generator = ColumnMeaningGeneratorOperator(
        dataset_description="测试数据集，包含用户基本信息",
        service=llm_service
    )
    
    # 运行列名处理
    result = column_generator.run(storage)
    
    print("\n=== 处理结果 ===")
    print(f"处理结果: {result}")
    
    # 提取并格式化显示列名意义
    if 'column_meanings' in result and 'columns' in result['column_meanings']:
        column_meanings = result['column_meanings']['columns']
        
        print("\n=== 列名解释（JSON格式）===")
        for meaning in column_meanings:
            print(f"\n列名: {meaning['column_name']}")
            print(f"JSON输出:")
            json_output = {
                "意义": meaning['meaning'],
                "单位": meaning['unit']
            }
            print(json.dumps(json_output, ensure_ascii=False, indent=2))
        
        # 保存完整的JSON输出到文件
        json_output_path = os.path.join(temp_dir, "column_meanings.json")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(column_meanings, f, ensure_ascii=False, indent=2)
        
        print(f"\n完整JSON输出已保存到: {json_output_path}")
    else:
        print("未找到列名解释数据")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()