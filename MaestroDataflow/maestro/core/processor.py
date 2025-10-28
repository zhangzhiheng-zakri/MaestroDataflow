"""
MaestroDataflow - 数据处理核心模块
"""

import pandas as pd
from typing import Callable, List, Dict, Any, Union
from maestro.utils.storage import FileStorage


class DataProcessor:
    """
    数据处理器类，提供数据转换和处理功能
    """

    def __init__(self, storage: FileStorage):
        """
        初始化数据处理器

        Args:
            storage: 文件存储实例
        """
        self.storage = storage

    def process(self, transform_func: Callable[[pd.DataFrame], pd.DataFrame]) -> str:
        """
        处理数据并保存结果

        Args:
            transform_func: 数据转换函数，接收DataFrame并返回处理后的DataFrame

        Returns:
            str: 处理后数据的保存路径
        """
        # 读取数据
        data = self.storage.step().read(output_type="dataframe")

        # 应用转换函数
        result = transform_func(data)

        # 保存结果
        return self.storage.write(result)

    def filter(self, condition: Callable[[pd.DataFrame], pd.Series]) -> str:
        """
        根据条件过滤数据

        Args:
            condition: 过滤条件函数，接收DataFrame并返回布尔Series

        Returns:
            str: 过滤后数据的保存路径
        """
        return self.process(lambda df: df[condition(df)])

    def select_columns(self, columns: List[str]) -> str:
        """
        选择指定列

        Args:
            columns: 要选择的列名列表

        Returns:
            str: 处理后数据的保存路径
        """
        return self.process(lambda df: df[columns])

    def rename_columns(self, rename_dict: Dict[str, str]) -> str:
        """
        重命名列

        Args:
            rename_dict: 列重命名字典，键为原列名，值为新列名

        Returns:
            str: 处理后数据的保存路径
        """
        return self.process(lambda df: df.rename(columns=rename_dict))

    def sort_values(self, by: Union[str, List[str]], ascending: bool = True) -> str:
        """
        对数据进行排序

        Args:
            by: 排序依据的列名或列名列表
            ascending: 是否升序排序

        Returns:
            str: 处理后数据的保存路径
        """
        return self.process(lambda df: df.sort_values(by=by, ascending=ascending))

    def group_by(self, by: Union[str, List[str]], agg_dict: Dict[str, str]) -> str:
        """
        分组聚合数据

        Args:
            by: 分组依据的列名或列名列表
            agg_dict: 聚合方法字典，键为列名，值为聚合方法

        Returns:
            str: 处理后数据的保存路径
        """
        return self.process(lambda df: df.groupby(by).agg(agg_dict).reset_index())

    def apply_function(self, func: Callable, column: str, new_column: str = None) -> str:
        """
        对指定列应用函数

        Args:
            func: 要应用的函数
            column: 要处理的列名
            new_column: 结果存储的新列名，默认覆盖原列

        Returns:
            str: 处理后数据的保存路径
        """
        def transform(df):
            target_column = new_column if new_column else column
            df_copy = df.copy()
            df_copy[target_column] = df_copy[column].apply(func)
            return df_copy

        return self.process(transform)

    def convert_format(self, output_path: str, format_type: str) -> str:
        """
        转换数据格式并保存

        Args:
            output_path: 输出文件路径
            format_type: 输出格式类型，支持"csv", "json", "jsonl", "xlsx"

        Returns:
            str: 输出文件路径
        """
        data = self.storage.step().read(output_type="dataframe")

        if format_type == "csv":
            data.to_csv(output_path, index=False)
        elif format_type == "json":
            data.to_json(output_path, orient="records", force_ascii=False, indent=2)
        elif format_type == "jsonl":
            data.to_json(output_path, orient="records", lines=True, force_ascii=False)
        elif format_type == "xlsx":
            data.to_excel(output_path, index=False)
        else:
            raise ValueError(f"不支持的输出格式: {format_type}")

        return output_path