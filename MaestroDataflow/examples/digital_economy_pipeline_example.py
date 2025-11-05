"""
MaestroDataflow 数字经济数据管道示例
演示如何使用Pipeline创建完整的数字经济数据分析流程
使用真实的中国数字经济发展数据（2005-2023年）
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from maestro.pipeline.pipeline import Pipeline
from maestro.utils.storage import FileStorage
from maestro.operators.io_ops import LoadDataOperator, SaveDataOperator
from maestro.operators.analytics_ops import DataAnalysisOperator


def create_digital_economy_pipeline():
    """创建数字经济数据分析管道"""
    
    # 创建存储实例
    storage = FileStorage(input_file_path="../sample_data/中国数字经济发展数据（2005-2023年）.xlsx")
    
    # 创建管道
    pipeline = Pipeline(storage=storage)
    
    # 1. 数据加载操作
    loader = LoadDataOperator(
        file_path="../sample_data/中国数字经济发展数据（2005-2023年）.xlsx"
    )
    pipeline.add_operator(loader, "loader")
    
    # 2. 数据分析操作
    analyzer = DataAnalysisOperator()
    pipeline.add_operator(analyzer, "analyzer")
    
    # 3. 结果保存操作
    saver = SaveDataOperator(
        output_path="output/digital_economy_analysis_result.csv"
    )
    pipeline.add_operator(saver, "saver")
    
    return pipeline


def run_pipeline_example():
    """运行管道示例"""
    print("=== 数字经济数据分析管道示例 ===")
    
    try:
        # 创建管道
        pipeline = create_digital_economy_pipeline()
        
        # 执行管道
        print("开始执行数据分析管道...")
        result = pipeline.run()
        
        print("✅ 管道执行成功!")
        print(f"执行结果: {result}")
        print(f"分析结果已保存到: output/digital_economy_analysis_result.csv")
        
        return result
        
    except Exception as e:
        print(f"❌ 管道执行失败: {e}")
        return None


def simple_analysis_example():
    """简单的数字经济数据分析示例"""
    print("\n=== 简单数字经济数据分析示例 ===")
    
    try:
        # 1. 加载数据
        print("1. 加载数字经济数据...")
        storage = FileStorage(input_file_path="../sample_data/中国数字经济发展数据（2005-2023年）.xlsx")
        storage.step()
        data = storage.read(output_type="dataframe")
        
        print(f"   数据形状: {data.shape}")
        print(f"   数据列: {list(data.columns)}")
        
        # 2. 数据分析
        print("2. 执行数据分析...")
        analyzer = DataAnalysisOperator()
        analysis_result = analyzer.run(storage)
        
        print("   分析完成!")
        print(f"   分析结果: {analysis_result}")
        
        # 3. 保存结果
        print("3. 保存分析结果...")
        saver = SaveDataOperator(output_path="output/simple_digital_economy_analysis.csv")
        save_result = saver.run(storage)
        
        print("✅ 简单分析完成!")
        print(f"   结果已保存到: output/simple_digital_economy_analysis.csv")
        print(f"   保存结果: {save_result}")
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ 简单分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("MaestroDataflow 数字经济数据分析示例")
    print("=" * 50)
    
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    
    # 运行管道示例
    pipeline_result = run_pipeline_example()
    
    # 运行简单分析示例
    simple_result = simple_analysis_example()
    
    print("\n" + "=" * 50)
    print("所有示例执行完成!")