"""
IO-related operators for saving data.
"""
from __future__ import annotations
from typing import Dict, Any, Literal
import os
import pandas as pd

from maestro.core import OperatorABC
from maestro.utils.storage import FileStorage
from maestro.utils.db_storage import DBStorage


class LoadDataOperator(OperatorABC):
    """从文件加载数据到存储系统"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def run(self, storage: FileStorage, **kwargs) -> Dict[str, Any]:
        """加载数据文件并写入存储系统"""
        # 对于FileStorage，需要先调用step()来初始化处理步骤
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
            
        # 根据文件扩展名确定读取方法
        if self.file_path.endswith('.csv'):
            df = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
            df = pd.read_excel(self.file_path, engine='openpyxl')
        elif self.file_path.endswith('.json'):
            df = pd.read_json(self.file_path)
        elif self.file_path.endswith('.parquet'):
            df = pd.read_parquet(self.file_path)
        else:
            raise ValueError(f"不支持的文件格式: {self.file_path}")
        
        # 写入存储系统
        path = storage.write(df)
        return {"path": path, "loaded_from": self.file_path, "shape": df.shape}


class SaveDataOperator(OperatorABC):
    """保存数据操作符的别名，指向SaveToFileOperator"""
    def __init__(self, output_path: str, format_type: str = "csv"):
        self.save_operator = SaveToFileOperator(output_path, format_type)
    
    def run(self, storage: FileStorage, **kwargs) -> Dict[str, Any]:
        return self.save_operator.run(storage, **kwargs)


class SaveToFileOperator(OperatorABC):
    """将当前DataFrame保存到指定路径与格式"""
    def __init__(
        self, 
        output_path: str, 
        format_type: Literal["csv", "json", "jsonl", "xlsx", "parquet", "pickle"] = "csv"
    ):
        self.output_path = output_path
        self.format_type = format_type
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def run(self, storage: FileStorage, **kwargs) -> Dict[str, Any]:
        df = storage.step().read(output_type="dataframe")

        if self.format_type == "csv":
            df.to_csv(self.output_path, index=False)
        elif self.format_type == "json":
            df.to_json(self.output_path, orient="records", force_ascii=False, indent=2)
        elif self.format_type == "jsonl":
            df.to_json(self.output_path, orient="records", lines=True, force_ascii=False)
        elif self.format_type == "xlsx":
            df.to_excel(self.output_path, index=False)
        elif self.format_type == "parquet":
            df.to_parquet(self.output_path, index=False)
        elif self.format_type == "pickle":
            import pickle
            with open(self.output_path, 'wb') as f:
                pickle.dump(df, f)
        else:
            raise ValueError(f"不支持的输出格式: {self.format_type}")

        path = storage.write(df)
        return {"path": path, "saved_to": self.output_path}


class SaveToDBOperator(OperatorABC):
    """将当前DataFrame保存到数据库表"""
    def __init__(self, table_name: str):
        self.table_name = table_name

    def run(self, storage: FileStorage, **kwargs) -> Dict[str, Any]:
        df = storage.step().read(output_type="dataframe")
        db_storage: DBStorage = kwargs.get("db_storage")
        if db_storage is None:
            raise ValueError("SaveToDBOperator 需要传入 db_storage 实例")

        db_storage.write(df, self.table_name)
        path = storage.write(df)
        return {"path": path, "saved_table": self.table_name}