"""
MaestroDataflow 示例 - 多格式数据处理

本示例展示如何使用MaestroDataflow处理不同格式的数据文件（XLSX、CSV、JSON等）
"""

import os
import sys
import pandas as pd

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from maestro.utils.storage import FileStorage

def create_sample_files():
    """创建示例数据文件（XLSX、CSV、JSON）"""
    # 创建示例数据
    sample_data = {
        'name': ['张三', '李四', '王五', '赵六', '钱七'],
        'age': [25, 30, 35, 40, 45],
        'department': ['研发', '市场', '销售', '人事', '财务'],
        'salary': [10000, 12000, 15000, 8000, 20000],
        'join_date': ['2020-01-01', '2019-05-10', '2021-03-15', '2018-07-22', '2022-02-28']
    }

    df = pd.DataFrame(sample_data)

    # 确保目录存在
    os.makedirs("maestro/example/DataProcessing/data", exist_ok=True)
    
    # 创建示例数据文件
    xlsx_path = "maestro/example/DataProcessing/data/employees.xlsx"
    df.to_excel(xlsx_path, index=False)

    # 保存为CSV文件
    csv_path = "maestro/example/DataProcessing/data/employees.csv"
    df.to_csv(csv_path, index=False)
    
    # 保存为JSON文件
    json_path = "maestro/example/DataProcessing/data/employees.json"
    df.to_json(json_path, orient='records', indent=2)
    
    # 保存为JSONL文件
    jsonl_path = "maestro/example/DataProcessing/data/employees.jsonl"
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    print(f"创建示例文件成功:")
    print(f"- XLSX: {xlsx_path}")
    print(f"- CSV: {csv_path}")
    print(f"- JSON: {json_path}")
    print(f"- JSONL: {jsonl_path}")

    return {
        'xlsx': xlsx_path,
        'csv': csv_path,
        'json': json_path,
        'jsonl': jsonl_path
    }

def example_xlsx_usage():
    """XLSX文件处理示例"""
    print("\n=== XLSX文件处理示例 ===")

    # 创建示例文件
    files = create_sample_files()
    xlsx_file = files['xlsx']

    try:
        # 创建FileStorage实例
        storage = FileStorage(
            input_file_path=xlsx_file,
            cache_path="./cache",
            file_name_prefix="xlsx_example",
            cache_type="xlsx",
        )

        # 初始化storage
        storage.step()

        # 读取数据为DataFrame格式
        dataframe = storage.read("dataframe")
        print(f"数据形状: {dataframe.shape}")
        print(f"列名: {list(dataframe.columns)}")
        print("\n数据预览:")
        print(dataframe.head(3))

        # 数据处理示例：筛选高薪员工
        high_salary = dataframe[dataframe['salary'] > 12000]
        print("\n高薪员工:")
        print(high_salary)

        # 将处理后的数据写回storage
        result_path = storage.write(high_salary)
        print(f"\n处理结果已保存至: {result_path}")

    except Exception as e:
        print(f"错误: {e}")

def example_csv_usage():
    """CSV文件处理示例"""
    print("\n=== CSV文件处理示例 ===")

    # 获取示例文件
    files = create_sample_files()
    csv_file = files['csv']

    try:
        # 创建FileStorage实例
        storage = FileStorage(
            input_file_path=csv_file,
            cache_path="./cache",
            file_name_prefix="csv_example",
            cache_type="csv",
        )

        # 初始化storage
        storage.step()

        # 读取数据为字典格式
        dict_data = storage.read("dict")
        print(f"记录数: {len(dict_data)}")
        print("第一条记录:")
        print(dict_data[0])

        # 数据处理示例：计算平均薪资
        dataframe = storage.read("dataframe")
        avg_salary = dataframe['salary'].mean()
        print(f"\n平均薪资: {avg_salary:.2f}")

        # 添加新列：是否高于平均薪资
        dataframe['above_average'] = dataframe['salary'] > avg_salary

        # 将处理后的数据写回storage
        result_path = storage.write(dataframe)
        print(f"\n处理结果已保存至: {result_path}")

    except Exception as e:
        print(f"错误: {e}")

def example_json_usage():
    """JSON文件处理示例"""
    print("\n=== JSON文件处理示例 ===")

    # 获取示例文件
    files = create_sample_files()
    json_file = files['json']

    try:
        # 创建FileStorage实例
        storage = FileStorage(
            input_file_path=json_file,
            cache_path="./cache",
            file_name_prefix="json_example",
            cache_type="json",
        )

        # 初始化storage
        storage.step()

        # 读取数据
        dataframe = storage.read("dataframe")
        print(f"数据形状: {dataframe.shape}")

        # 数据处理示例：按部门分组统计
        dept_stats = dataframe.groupby('department').agg({
            'salary': ['mean', 'min', 'max'],
            'age': 'mean'
        })

        print("\n部门统计:")
        print(dept_stats)

        # 将处理后的数据写回storage
        result_path = storage.write(dept_stats.reset_index())
        print(f"\n处理结果已保存至: {result_path}")

    except Exception as e:
        print(f"错误: {e}")

def example_format_conversion():
    """格式转换示例"""
    print("\n=== 格式转换示例 ===")

    # 获取示例文件
    files = create_sample_files()
    xlsx_file = files['xlsx']

    try:
        # 从XLSX读取
        storage_in = FileStorage(
            input_file_path=xlsx_file,
            cache_path="./cache",
            file_name_prefix="format_conversion",
            cache_type="xlsx",
        )

        # 初始化storage并读取数据
        storage_in.step()
        dataframe = storage_in.read("dataframe")
        print(f"从XLSX读取数据: {dataframe.shape}")

        # 转换为CSV
        storage_csv = FileStorage(
            input_file_path=xlsx_file,  # 实际上不会使用这个文件
            cache_path="./cache",
            file_name_prefix="converted_to_csv",
            cache_type="csv",
        )

        # 写入数据
        csv_path = storage_csv.write(dataframe)
        print(f"转换为CSV: {csv_path}")

        # 转换为JSON
        storage_json = FileStorage(
            input_file_path=xlsx_file,  # 实际上不会使用这个文件
            cache_path="./cache",
            file_name_prefix="converted_to_json",
            cache_type="json",
        )

        # 写入数据
        json_path = storage_json.write(dataframe)
        print(f"转换为JSON: {json_path}")

    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    print("MaestroDataflow 使用示例")
    print("=" * 50)

    # 运行示例
    example_xlsx_usage()
    example_csv_usage()
    example_json_usage()
    example_format_conversion()

    print("\n" + "=" * 50)
    print("所有示例运行完成！")
    print("\n使用说明:")
    print("1. 创建FileStorage实例，指定输入文件路径")
    print("2. 调用step()方法初始化")
    print("3. 使用read()方法读取数据，支持dataframe和dict格式")
    print("4. 处理数据后，使用write()方法保存结果")
    print("5. 支持XLSX、CSV、JSON和JSONL格式")