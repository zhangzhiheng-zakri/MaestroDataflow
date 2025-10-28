"""
数据列处理操作符 - 整合数据存储到数据库和列名意义单位生成功能

"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import json
from pathlib import Path

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
    
    def _fill_missing_values(self, df: pd.DataFrame, method: str = "median") -> pd.DataFrame:
        """
        填充缺失值
        
        Args:
            df: 数据框
            method: 填充方法 ("median", "mean", "mode", "forward", "backward")
            
        Returns:
            填充后的数据框
        """
        print(f"使用 {method} 方法填充缺失值")
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if method == "median" and df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                elif method == "mean" and df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == "mode":
                    mode_value = df[col].mode()
                    if not mode_value.empty:
                        df[col].fillna(mode_value.iloc[0], inplace=True)
                elif method == "forward":
                    df[col].fillna(method='ffill', inplace=True)
                elif method == "backward":
                    df[col].fillna(method='bfill', inplace=True)
                else:
                    # 默认使用众数填充
                    mode_value = df[col].mode()
                    if not mode_value.empty:
                        df[col].fillna(mode_value.iloc[0], inplace=True)
        
        return df
    
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
        
        # 3. 填充缺失值
        final_df = self._fill_missing_values(cleaned_df, fill_method)
        
        # 4. 存储到数据库
        db_storage = self._store_to_database(final_df)
        
        # 5. 生成列名意义和单位
        print("=== 生成列名意义和单位 ===")
        
        # 创建临时存储来传递数据给子操作符
        temp_storage = storage.step()
        temp_storage.write(final_df)
        
        # 生成列名意义
        meaning_result = self.column_meaning_generator.run(temp_storage, llm_service=llm_service)
        column_meanings = meaning_result["column_meanings"]
        
        # 生成元数据
        metadata_result = self.metadata_extractor.run(temp_storage)
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
                "rows_dropped": df.shape[0] - final_df.shape[0]
            }
        }
        
        # 7. 保存最终结果
        output_path = storage.write(final_result)
        
        # 8. 同时存储到数据库
        db_storage.write(final_result, key="column_meanings_output")
        
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