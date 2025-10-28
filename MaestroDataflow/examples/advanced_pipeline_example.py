# -*- coding: utf-8 -*-
"""
MaestroDataflow示例：展示Pipeline框架、数据库存储和AI模型服务的集成使用
这个示例展示了如何：
1. 创建一个数据处理Pipeline
2. 使用数据库存储数据
3. 集成AI模型服务进行文本生成
4. 使用多种文件格式保存结果
"""

import os
import pandas as pd
from typing import Dict, Any, List

# 导入MaestroDataflow组件
from maestro import FileStorage, Pipeline
from maestro.core import OperatorABC


class DataLoader(OperatorABC):
    """加载数据的操作符"""

    def __init__(self, data_path: str):
        self.data_path = data_path

    def run(self, storage: FileStorage, **kwargs) -> Dict[str, Any]:
        """加载数据并返回"""
        print(f"加载数据: {self.data_path}")

        # 初始化处理步骤并读取数据
        data = storage.step().read(output_type="dataframe")

        # 写入到下一步
        storage.write(data)

        return {"data_shape": data.shape}


class DataPreprocessor(OperatorABC):
    """数据预处理操作符"""

    def run(self, storage: FileStorage, **kwargs) -> Dict[str, Any]:
        """预处理数据"""
        # 读取上一步的数据
        data = storage.step().read(output_type="dataframe")
        print(f"预处理数据 ({len(data)} 行)")

        # 示例预处理：填充缺失值，删除重复行
        processed_data = data.copy()
        processed_data = processed_data.fillna("未知")
        processed_data = processed_data.drop_duplicates()

        # 写入到下一步
        storage.write(processed_data)

        return {"processed_rows": len(processed_data)}


class TextGenerator(OperatorABC):
    """文本生成操作符"""

    def __init__(self, llm_service):
        self.llm_service = llm_service

    def run(self, storage: FileStorage, **kwargs) -> Dict[str, Any]:
        """为每行数据生成摘要文本"""
        # 读取上一步的数据
        processed_data = storage.step().read(output_type="dataframe")
        print(f"生成文本摘要 ({len(processed_data)} 行)")

        summaries = []
        for _, row in processed_data.iterrows():
            # 构建提示词
            prompt = f"请为以下数据生成简短摘要：{row.to_dict()}"

            # 生成文本
            summary = self.llm_service.generate(prompt, max_tokens=100)
            summaries.append(summary)

        # 添加摘要列
        result_data = processed_data.copy()
        result_data['ai_summary'] = summaries

        # 写入到下一步
        storage.write(result_data)

        return {"generated_summaries": len(summaries)}


class DataExporter(OperatorABC):
    """数据导出操作符"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, storage: FileStorage, **kwargs) -> Dict[str, Any]:
        """导出数据到多种格式"""
        # 读取上一步的数据
        enriched_data = storage.step().read(output_type="dataframe")
        print(f"导出数据到 {self.output_dir}")

        export_paths = {}

        # 导出为CSV
        csv_path = os.path.join(self.output_dir, "results.csv")
        enriched_data.to_csv(csv_path, index=False, encoding='utf-8')
        export_paths['csv'] = csv_path

        # 导出为JSON
        json_path = os.path.join(self.output_dir, "results.json")
        enriched_data.to_json(json_path, orient='records', force_ascii=False, indent=2)
        export_paths['json'] = json_path

        # 导出为Excel
        xlsx_path = os.path.join(self.output_dir, "results.xlsx")
        enriched_data.to_excel(xlsx_path, index=False)
        export_paths['xlsx'] = xlsx_path

        return {"export_paths": export_paths}


def create_sample_data():
    """创建示例数据"""
    data = pd.DataFrame({
        'id': range(1, 6),
        'name': ['张三', '李四', '王五', '赵六', '钱七'],
        'age': [25, 30, 35, 28, 32],
        'city': ['北京', '上海', '广州', '深圳', '杭州'],
        'profession': ['工程师', '设计师', '产品经理', '数据分析师', '运营专员']
    })

    # 保存示例数据
    os.makedirs("../output/advanced_pipeline_example/data", exist_ok=True)
    sample_path = "../output/advanced_pipeline_example/data/employees.csv"
    data.to_csv(sample_path, index=False, encoding='utf-8')

    return sample_path


def main():
    """主函数：演示完整的数据处理流程"""
    print("=== MaestroDataflow 高级示例 ===")

    # 1. 创建示例数据
    sample_data_path = create_sample_data()
    print(f"创建示例数据: {sample_data_path}")

    # 2. 初始化存储
    storage = FileStorage(
        input_file_path=sample_data_path,
        cache_path="../output/advanced_pipeline_example/cache",
        file_name_prefix="advanced_example",
        cache_type="csv"
    )

    # 3. 初始化AI服务（这里使用模拟服务）
    class MockLLMService:
        def generate(self, prompt: str, max_tokens: int = 100) -> str:
            return f"AI生成的摘要：基于提供的数据生成的智能摘要内容"

    llm_service = MockLLMService()

    # 4. 创建操作符实例
    loader = DataLoader(sample_data_path)
    preprocessor = DataPreprocessor()
    text_generator = TextGenerator(llm_service)
    exporter = DataExporter("../output/advanced_pipeline_example/results")

    # 5. 构建Pipeline
    pipeline = Pipeline(storage=storage)

    # 添加操作符（使用add_operator方法）
    pipeline.add_operator(loader, "loader")
    pipeline.add_operator(preprocessor, "preprocessor")
    pipeline.add_operator(text_generator, "text_generator")
    pipeline.add_operator(exporter, "exporter")

    # 6. 执行Pipeline
    print("\n开始执行Pipeline...")
    result = pipeline.run()

    # 7. 获取最终结果
    final_data = storage.read(output_type="dataframe")

    print("\nPipeline执行完成！")
    print(f"结果导出到: {result['exporter']['export_paths']}")
    print(f"最终数据形状: {final_data.shape}")

    # 显示最终数据的前几行
    print("\n最终数据预览:")
    print(final_data.head())


if __name__ == "__main__":
    main()

