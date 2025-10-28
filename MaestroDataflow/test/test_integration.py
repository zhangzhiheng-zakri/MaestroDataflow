#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaestroDataflow 集成测试

测试各个组件之间的协作
"""

import os
import pandas as pd
from maestro.utils.storage import FileStorage
from maestro.utils.db_storage import DBStorage
from maestro.operators.basic_ops import FilterRowsOperator
from maestro.pipeline.pipeline import Pipeline
try:
    import pytest
except ImportError:
    import sys
    import warnings
    warnings.warn("pytest 未安装，跳过测试运行。如需运行测试，请执行: pip install pytest")
    sys.exit(0)

class TestIntegration:
    """集成测试类"""
    
    def test_file_storage_pipeline_integration(self, sample_data, clean_cache):
        """测试FileStorage与Pipeline的集成"""
        # 创建测试CSV文件
        test_file = "test_integration.csv"
        df = pd.DataFrame(sample_data)
        df.to_csv(test_file, index=False)
        
        try:
            # 创建存储和管道
            storage = FileStorage(test_file, cache_path="./test_cache", cache_type="csv")
            pipeline = Pipeline(storage)
            
            # 添加过滤操作
            filter_op = FilterRowsOperator(lambda df: df["salary"] > 55000)
            pipeline.add_operator(filter_op, "filter")
            
            # 运行管道
            result = pipeline.run()
            
            # 验证结果 - 从最新步骤读取
            result_data = storage.step().read(output_type="dict")
            assert len(result_data) == 3  # 应该有3条记录满足条件 (李四: 60000, 王五: 70000, 赵六: 80000)
            assert all(row["salary"] > 55000 for row in result_data)
            
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_db_storage_pipeline_integration(self, sample_data, clean_cache):
        """测试DBStorage与Pipeline的集成"""
        # 创建数据库存储
        db_storage = DBStorage("sqlite:///test/db/test_integration.db")
        db_storage.reset()
        db_storage.write(sample_data)
        
        try:
            # 使用step()方法获取正确的步骤
            db_storage_step = db_storage.step()
            pipeline = Pipeline(db_storage_step)
            
            # 添加过滤操作
            filter_op = FilterRowsOperator(lambda df: df["city"].isin(["上海", "广州"]))
            pipeline.add_operator(filter_op, "filter")
            
            # 运行管道
            result = pipeline.run()
            
            # 验证结果 - 操作符会将结果写入下一步，所以需要再次调用step()
            final_storage = db_storage_step.step()
            result_data = final_storage.read(output_type="dict")
            assert len(result_data) == 2  # 应该有2条记录满足条件
            assert all(row["city"] in ["上海", "广州"] for row in result_data)
            
        finally:
            # 清理数据库文件
            try:
                if os.path.exists("test/db/test_integration.db"):
                    os.remove("test/db/test_integration.db")
            except PermissionError:
                pass  # 忽略权限错误
    
    def test_storage_compatibility(self, sample_data, clean_cache):
        """测试不同存储类型的兼容性"""
        # 创建测试CSV文件
        test_file = "test_compatibility.csv"
        df = pd.DataFrame(sample_data)
        df.to_csv(test_file, index=False)
        
        try:
            # FileStorage处理
            file_storage = FileStorage(test_file, cache_path="./test_cache", cache_type="csv")
            file_pipeline = Pipeline(file_storage)
            filter_op1 = FilterRowsOperator(lambda df: df["salary"] >= 60000)
            file_pipeline.add_operator(filter_op1, "filter")
            file_result_meta = file_pipeline.run()
            file_result = file_storage.step().read(output_type="dict")
            
            # DBStorage处理
            db_storage = DBStorage("sqlite:///test/db/test_compatibility.db")
            db_storage.reset()
            db_storage.write(sample_data)
            db_storage_step = db_storage.step()
            db_pipeline = Pipeline(db_storage_step)
            filter_op2 = FilterRowsOperator(lambda df: df["salary"] >= 60000)
            db_pipeline.add_operator(filter_op2, "filter")
            db_result_meta = db_pipeline.run()
            db_result = db_storage_step.step().read(output_type="dict")
            
            # 验证两种存储方式的结果一致
            assert len(file_result) == len(db_result)
            
            # 按name排序后比较
            file_sorted = sorted(file_result, key=lambda x: x["name"])
            db_sorted = sorted(db_result, key=lambda x: x["name"])
            
            for f_row, d_row in zip(file_sorted, db_sorted):
                assert f_row["name"] == d_row["name"]
                assert f_row["city"] == d_row["city"]
                assert f_row["salary"] == d_row["salary"]
                
        finally:
            # 清理文件
            for file_path in [test_file, "test/db/test_compatibility.db"]:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except PermissionError:
                    pass  # 忽略权限错误


if __name__ == "__main__":
    pytest.main([__file__, "-v"])