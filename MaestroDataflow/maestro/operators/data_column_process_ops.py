"""
数据列处理操作符 - 整合数据存储到数据库和列名意义单位生成功能

"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import json
from pathlib import Path
import numpy as np

from maestro.core import OperatorABC
from maestro.utils.storage import MaestroStorage
from maestro.utils.db_storage import DBStorage
from maestro.serving.llm_serving import APILLMServing, LocalLLMServing
from maestro.operators.column_ops import ColumnMeaningGeneratorOperator, ColumnMetadataExtractorOperator


class DataColumnProcessOperator(OperatorABC):
    """
    数据列处理操作符 - 将数据存储到数据库并生成列名意义单位的JSON格式
    
    功能包括：
    1. 数据清洗（使用MaestroDataflow的方法）
    2. 数据存储到数据库
    3. 生成列名的意义和单位说明
    4. 输出JSON格式的列名意义单位
    """
    
    def __init__(
        self,
        dataset_name: str,
        dataset_description: str,
        db_connection_string: str,
        table_name: Optional[str] = None,
        service: APILLMServing | LocalLLMServing | None = None,
        max_columns_per_batch: int = 10
    ):
        self.dataset_name = dataset_name
        self.dataset_description = dataset_description
        self.db_connection_string = db_connection_string
        self.table_name = table_name or f"maestro_{dataset_name.lower().replace(' ', '_')}"
        self.service = service
        self.max_columns_per_batch = max_columns_per_batch
        
        # 初始化子操作符
        self.column_meaning_generator = ColumnMeaningGeneratorOperator(
            dataset_description=dataset_description,
            max_columns_per_batch=max_columns_per_batch,
            service=service
        )
        self.metadata_extractor = ColumnMetadataExtractorOperator()
    
    def _clean_data(self, df: pd.DataFrame, na_threshold: float = 0.5) -> pd.DataFrame:
        """
        使用MaestroDataflow的方法进行数据清洗
        
        Args:
            df: 原始数据框
            na_threshold: 缺失值阈值，超过此比例的列将被删除
            
        Returns:
            清洗后的数据框
        """
        print(f"开始数据清洗，原始数据形状: {df.shape}")
        
        # 1. 删除缺失值过多的列
        missing_ratio = df.isnull().sum() / len(df)
        columns_to_drop = missing_ratio[missing_ratio > na_threshold].index.tolist()
        
        if columns_to_drop:
            print(f"删除缺失值过多的列 (>{na_threshold*100}%): {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
        
        # 2. 删除完全重复的行
        initial_rows = len(df)
        df = df.drop_duplicates()
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"删除重复行: {dropped_rows} 行")
        
        # 3. 基本的数据类型优化
        for col in df.columns:
            if df[col].dtype == 'object':
                # 尝试转换为数值类型
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        print(f"数据清洗完成，清洗后数据形状: {df.shape}")
        return df
    
    def _fill_missing_values(
        self,
        df: pd.DataFrame,
        method: Union[str, Dict[str, Any], None] = "median",
        **kwargs
    ) -> Dict[str, Any]:
        """
        更完善的缺失值填充支持：
        - 全局方法："median" | "mean" | "mode" | "forward" | "backward" | "interpolate" | "knn" | "auto"
        - 按列配置：dict，如 {"colA": "mean", "colB": ("constant", 0), "colC": {"method": "interpolate", "params": {"method": "linear"}}}
        - 组内填充：传入 group_keys=[...] 在分组内计算统计量填充
        - 缺失指示列：create_missing_indicators=True 时为每列添加 <col>__is_missing 指示列
        - 时间序列插值：time_column=... 配合 interpolate_method="time"/"linear" 等
        - KNN 数值填充：knn_numeric=True 或 method="knn" 时尝试对数值列使用 KNNImputer（可选，若不可用则回退）

        Args:
            df: 数据框
            method: 填充方法或字典配置
            **kwargs: 附加配置（group_keys, create_missing_indicators, time_column, interpolate_method,
                      knn_numeric, knn_n_neighbors, fill_order）

        Returns:
            {"df": 填充后的DataFrame, "stats": 每列填充统计}
        """
        from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

        group_keys: List[str] = kwargs.get("group_keys", [])
        create_ind: bool = kwargs.get("create_missing_indicators", False)
        time_col: Optional[str] = kwargs.get("time_column")
        interp_method: str = kwargs.get("interpolate_method", "linear")
        knn_numeric: bool = kwargs.get("knn_numeric", False) or (isinstance(method, str) and method == "knn")
        knn_n_neighbors: int = int(kwargs.get("knn_n_neighbors", 5))
        fill_order: List[str] = kwargs.get("fill_order", [])  # 例如 ["forward", "backward", "mode"]

        print(f"使用缺失值填充策略: {method}")
        stats: Dict[str, Dict[str, Any]] = {}

        # 可选：添加缺失指示列
        if create_ind:
            for col in df.columns:
                missing_mask = df[col].isna()
                if missing_mask.any():
                    ind_col = f"{col}__is_missing"
                    if ind_col not in df.columns:
                        df[ind_col] = missing_mask.astype(int)

        def resolve_col_method(col: str) -> Union[str, Dict[str, Any], tuple]:
            if isinstance(method, dict):
                cfg = method.get(col)
                if cfg is not None:
                    return cfg
            if isinstance(method, str):
                if method == "auto":
                    if is_numeric_dtype(df[col]):
                        return "median"
                    elif is_datetime64_any_dtype(df[col]):
                        return "forward"
                    else:
                        return "mode"
                return method
            return "mode"

        # 组内统计填充辅助
        def group_stat_fill(series: pd.Series, how: str) -> pd.Series:
            if not group_keys:
                return series
            try:
                if how in ("median", "mean"):
                    agg = series.groupby(df[group_keys].apply(tuple, axis=1)).transform(how)
                    return series.fillna(agg)
                elif how == "mode":
                    # mode较复杂，采用分组内众数近似：使用分组内出现频次最高的值
                    grp = df[group_keys].apply(tuple, axis=1)
                    filled = series.copy()
                    for g in grp.unique():
                        idx = grp == g
                        sub = series[idx]
                        if sub.isna().any():
                            mode_val = sub.mode()
                            if not mode_val.empty:
                                filled.loc[idx] = sub.fillna(mode_val.iloc[0])
                    return filled
            except Exception:
                return series
            return series

        # 主填充逻辑（逐列）
        for col in df.columns:
            na_count = int(df[col].isna().sum())
            if na_count == 0:
                continue
            used_method = resolve_col_method(col)
            filled = df[col].copy()
            before_na = int(filled.isna().sum())

            def apply_simple_fill(series: pd.Series, how: str) -> pd.Series:
                if how == "median" and is_numeric_dtype(series):
                    return series.fillna(series.median())
                if how == "mean" and is_numeric_dtype(series):
                    return series.fillna(series.mean())
                if how == "mode":
                    mode_value = series.mode()
                    if not mode_value.empty:
                        return series.fillna(mode_value.iloc[0])
                    return series
                if how == "forward":
                    return series.fillna(method="ffill")
                if how == "backward":
                    return series.fillna(method="bfill")
                if how == "interpolate" and is_numeric_dtype(series):
                    try:
                        if time_col and time_col in df.columns:
                            # 基于时间列排序临时插值，不改变原顺序
                            order = df[time_col].argsort(kind="mergesort")
                            tmp = series.iloc[order]
                            tmp = tmp.interpolate(method=interp_method, limit_direction="both")
                            inv = pd.Series(index=order, data=tmp.values)
                            inv = inv.sort_index()
                            return series.fillna(inv)
                        else:
                            return series.interpolate(method=interp_method, limit_direction="both")
                    except Exception:
                        return series
                return series

            # 处理列级配置
            if isinstance(used_method, tuple) and len(used_method) == 2 and str(used_method[0]).lower() == "constant":
                filled = filled.fillna(used_method[1])
            elif isinstance(used_method, dict):
                how = str(used_method.get("method", "mode")).lower()
                params = used_method.get("params", {})
                if how == "interpolate":
                    m = params.get("method", interp_method)
                    filled = apply_simple_fill(filled, "interpolate")
                else:
                    filled = apply_simple_fill(filled, how)
            elif isinstance(used_method, str):
                how = used_method.lower()
                # 优先组内填充
                if group_keys:
                    filled = group_stat_fill(filled, how)
                # 常规填充
                filled = apply_simple_fill(filled, how)

            # 按顺序回退填充剩余缺失
            if fill_order:
                for fb in fill_order:
                    if int(filled.isna().sum()) == 0:
                        break
                    filled = apply_simple_fill(filled, fb)

            # 写回列
            df[col] = filled
            after_na = int(df[col].isna().sum())
            stats[col] = {
                "method": used_method if not isinstance(used_method, dict) else used_method.get("method", "custom"),
                "before_missing": before_na,
                "after_missing": after_na,
                "filled_count": before_na - after_na,
                "group_keys_used": bool(group_keys),
            }

        # 可选：对数值列使用KNN补充剩余缺失
        if knn_numeric:
            try:
                from sklearn.impute import KNNImputer  # 可选依赖
                num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
                if num_cols:
                    imputer = KNNImputer(n_neighbors=knn_n_neighbors)
                    num_array = imputer.fit_transform(df[num_cols])
                    # 统计填充增量
                    for i, c in enumerate(num_cols):
                        before_na = int(df[c].isna().sum())
                        df[c] = num_array[:, i]
                        after_na = int(df[c].isna().sum())
                        if c in stats:
                            stats[c]["method"] = f"{stats[c]['method']}+knn"
                            stats[c]["filled_count"] += before_na - after_na
                        else:
                            stats[c] = {"method": "knn", "before_missing": before_na, "after_missing": after_na, "filled_count": before_na - after_na, "group_keys_used": False}
            except Exception:
                print("⚠️ KNNImputer 不可用或执行失败，已跳过 KNN 数值填充")

        return {"df": df, "stats": stats}
    
    def _store_to_database(self, df: pd.DataFrame) -> DBStorage:
        """
        将数据存储到数据库
        
        Args:
            df: 要存储的数据框
            
        Returns:
            DBStorage实例
        """
        print(f"将数据存储到数据库表: {self.table_name}")
        
        # 创建DBStorage实例
        db_storage = DBStorage(
            connection_string=self.db_connection_string,
            table_name=self.table_name
        )
        
        # 存储数据
        db_storage.write(df, key="cleaned_data")
        
        # 存储数据集元信息
        dataset_info = {
            "dataset_name": self.dataset_name,
            "dataset_description": self.dataset_description,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": df.columns.tolist()
        }
        db_storage.write(dataset_info, key="dataset_info")
        
        print(f"数据存储完成，共 {len(df)} 行 {len(df.columns)} 列")
        return db_storage
    
    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """
        执行完整的数据控制流程
        
        Args:
            storage: 输入存储
            **kwargs: 其他参数，包括：
                - na_threshold: 缺失值阈值 (默认0.5)
                - fill_method: 填充方法 (默认"median")
                - llm_service: LLM服务实例
                
        Returns:
            包含处理结果的字典
        """
        # 获取参数
        na_threshold = kwargs.get("na_threshold", 0.5)
        fill_method = kwargs.get("fill_method", "median")
        llm_service = self.service or kwargs.get("llm_service")
        
        if llm_service is None:
            raise ValueError("DataColumnProcessOperator 需要传入 llm_service 或在初始化时提供 service")
        
        # 对于FileStorage，需要先调用step()来初始化处理步骤
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        print(f"=== 开始数据控制流程: {self.dataset_name} ===")
        
        # 1. 读取原始数据
        df = storage.read(output_type="dataframe")
        print(f"读取原始数据: {df.shape}")
        
        # 2. 数据清洗
        cleaned_df = self._clean_data(df, na_threshold)
        
        # 3. 填充缺失值（支持高级策略）
        fill_result = self._fill_missing_values(cleaned_df, fill_method, **kwargs)
        final_df = fill_result["df"]
        fill_stats = fill_result["stats"]
        
        # 4. 存储到数据库
        db_storage = self._store_to_database(final_df)
        
        # 5. 生成列名意义和单位
        print("=== 生成列名意义和单位 ===")
        
        # 创建临时存储来传递数据给子操作符
        temp_storage = storage.step()
        temp_storage.write(final_df)
        
        # 生成列名意义
        # 直接传入清洗后的DataFrame以绕过缓存读取
        meaning_result = self.column_meaning_generator.run(
            temp_storage,
            llm_service=llm_service,
            df=final_df
        )
        column_meanings = meaning_result["column_meanings"]
        
        # 生成元数据
        # 直接传入清洗后的DataFrame以绕过缓存读取
        metadata_result = self.metadata_extractor.run(temp_storage, df=final_df)
        column_metadata = metadata_result["metadata"]
        
        # 6. 合并结果并输出JSON格式
        final_result = {
            "dataset_info": {
                "name": self.dataset_name,
                "description": self.dataset_description,
                "total_rows": len(final_df),
                "total_columns": len(final_df.columns),
                "processing_date": pd.Timestamp.now().isoformat(),
                "database_table": self.table_name
            },
            "column_meanings": column_meanings["columns"],
            "column_metadata": column_metadata["columns"],
            "processing_summary": {
                "original_shape": df.shape,
                "final_shape": final_df.shape,
                "na_threshold_used": na_threshold,
                "fill_method_used": fill_method,
                "columns_dropped": df.shape[1] - final_df.shape[1],
                "rows_dropped": df.shape[0] - final_df.shape[0],
                "missing_fill_stats": {
                    "total_columns_with_missing": int(sum(1 for c in fill_stats if fill_stats[c]["before_missing"] > 0)),
                    "total_filled": int(sum(s["filled_count"] for s in fill_stats.values())),
                    "per_column": fill_stats
                }
            }
        }
        
        # 7. 保存最终结果
        output_path = storage.write(final_result)
        
        # 8. 同时存储到数据库（确保JSON可序列化）
        def _to_serializable(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_to_serializable(v) for v in obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj

        try:
            db_storage.write(_to_serializable(final_result), key="column_meanings_output")
        except Exception as e:
            print(f"警告：写入数据库失败（跳过，不影响流程）：{e}")
        
        print("=== 数据控制流程完成 ===")
        
        return {
            "output_path": output_path,
            "database_table": self.table_name,
            "final_data_shape": final_df.shape,
            "column_meanings_count": len(column_meanings["columns"]),
            "result": final_result
        }


class QuickDataColumnProcessOperator(OperatorABC):
    """
    快速数据列处理操作符 - 简化版本，主要用于快速生成列名意义单位的JSON
    """
    
    def __init__(
        self,
        dataset_description: str,
        service: APILLMServing | LocalLLMServing | None = None
    ):
        self.dataset_description = dataset_description
        self.service = service
        
        self.column_meaning_generator = ColumnMeaningGeneratorOperator(
            dataset_description=dataset_description,
            service=service
        )
    
    def run(self, storage: MaestroStorage, **kwargs) -> Dict[str, Any]:
        """
        快速生成列名意义单位的JSON格式
        """
        llm_service = self.service or kwargs.get("llm_service")
        if llm_service is None:
            raise ValueError("QuickDataColumnProcessOperator 需要传入 llm_service")
        
        if hasattr(storage, 'operator_step') and storage.operator_step == -1:
            storage = storage.step()
        
        # 生成列名意义
        result = self.column_meaning_generator.run(storage, llm_service=llm_service)
        
        # 简化输出格式
        simplified_result = {
            "dataset_description": self.dataset_description,
            "column_meanings": result["column_meanings"]["columns"],
            "generated_date": pd.Timestamp.now().isoformat()
        }
        
        # 保存结果
        output_path = storage.write(simplified_result)
        
        return {
            "output_path": output_path,
            "column_meanings": simplified_result
        }