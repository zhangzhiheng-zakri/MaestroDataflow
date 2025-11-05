"""
示例：将处理后的数据打包成 datasets 规范结构。

运行方式：
    python -m examples.package_dataset_example
"""

from maestro.operators import DatasetPackagingOperator
from maestro.utils.storage import FileStorage
from maestro.serving.llm_serving import APILLMServing
from maestro.serving.enhanced_llm_serving import EnhancedLLMServing
import os


def main():
    # 使用已存在的示例数据作为输入文件
    input_csv = "input/上市公司能源消耗数据（2012-2024年）/上市公司能源消耗数据（2012-2024年）.xlsx"

    # 初始化文件存储（需提供 input_file_path）
    storage = FileStorage(
        input_file_path=input_csv,
        cache_path="./cache",
        file_name_prefix="pack_demo",
        cache_type="csv"
    )

    # 初始化LLM服务：DeepSeek API
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("⚠️ 未设置DEEPSEEK_API_KEY环境变量，将按演示继续但可能返回占位说明")
        api_key = "demo-key-placeholder"
    base_serving = APILLMServing(
        api_url="https://api.deepseek.com/v1/chat/completions",
        api_key=api_key,
        model_name="deepseek-chat",
        api_type="openai"
    )
    llm_service = EnhancedLLMServing(base_serving=base_serving, enable_cache=True)
    print("✅ 使用DeepSeek API LLM服务")

    # 打包为 datasets 结构，并通过LLM生成列名“意义/单位”
    packer = DatasetPackagingOperator(dataset_name="示例数据集2025")
    result = packer.run(
        storage=storage,
        service=llm_service,
        normalized_data_filename="示例数据集2025_标准化.csv",
        dataset_description="示例数据集2025，包含年份、GDP、城市代码、满意度评分等字段"
    )

    print("打包完成：")
    for k, v in result.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()