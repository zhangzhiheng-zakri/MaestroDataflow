"""
Basic data transformation operators.
"""
from __future__ import annotations
from typing import Dict, Any, List, Callable
import pandas as pd

from maestro.core import OperatorABC
from maestro.utils.storage import MaestroStorage


class FilterRowsOperator(OperatorABC):
    """根据条件函数过滤行"""
    def __init__(self, condition: Callable[[pd.DataFrame], pd.Series]):
        self.condition = condition

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        # 对于FileStorage，需要先调用step()来初始化处理步骤
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        df = storage.read(output_type="dataframe")
        result = df[self.condition(df)]
        path = storage.write(result)
        return {"path": path, "rows": len(result)}


class SelectColumnsOperator(OperatorABC):
    """选择指定列"""
    def __init__(self, columns: List[str]):
        self.columns = columns

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        # 对于FileStorage，需要先调用step()来初始化处理步骤
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        df = storage.read(output_type="dataframe")
        result = df[self.columns]
        path = storage.write(result)
        return {"path": path, "columns": len(result.columns)}


class MapRowsOperator(OperatorABC):
    """对某列应用函数生成新列或覆盖"""
    def __init__(self, column: str, func: Callable[[Any], Any], new_column: str | None = None):
        self.column = column
        self.func = func
        self.new_column = new_column or column

    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        # 对于FileStorage，需要先调用step()来初始化处理步骤
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        df = storage.read(output_type="dataframe")
        df_copy = df.copy()
        df_copy[self.new_column] = df_copy[self.column].apply(self.func)
        path = storage.write(df_copy)
        return {"path": path, "updated_column": self.new_column}