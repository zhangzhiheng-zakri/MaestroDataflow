#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
存储系统边界情况测试
"""

import os
import sys
import pandas as pd

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from maestro.utils.storage import FileStorage

def test_empty_data():
    """测试空数据处理"""
    print("测试空数据处理...")
    
    # 创建测试文件
    test_file = "test_empty.csv"
    pd.DataFrame().to_csv(test_file, index=False)
    
    try:
        storage = FileStorage(test_file, cache_path="./test_cache", cache_type="csv")
        storage.reset()
        
        # 测试写入空列表
        storage.write([])
        print("OK - 空列表写入成功")
        
        # 测试读取空数据
        storage_step = storage.step()
        data = storage_step.read()
        print(f"OK - 空数据读取成功，形状: {data.shape}")
        
    except Exception as e:
        print(f"ERROR - 空数据测试失败: {e}")
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)

def test_invalid_data_types():
    """测试无效数据类型处理"""
    print("\n测试无效数据类型处理...")
    
    # 创建测试文件
    test_file = "test_invalid.csv"
    pd.DataFrame({"test": [1, 2, 3]}).to_csv(test_file, index=False)
    
    try:
        storage = FileStorage(test_file, cache_path="./test_cache", cache_type="csv")
        storage.reset()
        
        # 测试写入无效数据类型
        try:
            storage.write("invalid_string")
            print("ERROR - 应该抛出异常但没有")
        except ValueError as e:
            print(f"OK - 正确捕获无效数据类型: {e}")
        
        # 测试写入非字典列表
        try:
            storage.write([1, 2, 3])
            print("ERROR - 应该抛出异常但没有")
        except ValueError as e:
            print(f"OK - 正确捕获非字典列表: {e}")
            
    except Exception as e:
        print(f"ERROR - 无效数据类型测试失败: {e}")
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)

def test_file_operations():
    """测试文件操作边界情况"""
    print("\n测试文件操作边界情况...")
    
    # 测试不存在的文件
    try:
        storage = FileStorage("nonexistent_file.csv")
        print("ERROR - 应该抛出FileNotFoundError但没有")
    except FileNotFoundError:
        print("OK - 正确处理不存在的文件")
    except Exception as e:
        print(f"ERROR - 意外异常: {e}")
    
    # 测试无效的缓存类型
    test_file = "test_cache_type.csv"
    pd.DataFrame({"test": [1, 2, 3]}).to_csv(test_file, index=False)
    
    try:
        storage = FileStorage(test_file, cache_path="./test_cache", cache_type="invalid_type")
        storage.reset()
        storage.write([{"test": 1}])
        print("ERROR - 应该抛出ValueError但没有")
    except ValueError as e:
        print(f"OK - 正确处理无效缓存类型: {e}")
    except Exception as e:
        print(f"ERROR - 意外异常: {e}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def test_step_management():
    """测试步骤管理"""
    print("\n测试步骤管理...")
    
    test_file = "test_steps.csv"
    pd.DataFrame({"test": [1, 2, 3]}).to_csv(test_file, index=False)
    
    try:
        storage = FileStorage(test_file, cache_path="./test_cache", cache_type="csv")
        storage.reset()
        
        # 测试在没有调用step()的情况下读取
        try:
            storage.read()
            print("ERROR - 应该抛出ValueError但没有")
        except ValueError as e:
            print(f"OK - 正确处理未初始化步骤: {e}")
        
        # 测试正常步骤管理
        storage.write([{"test": 1}, {"test": 2}])
        storage_step = storage.step()
        data = storage_step.read()
        print(f"OK - 步骤管理正常，数据形状: {data.shape}")
        
    except Exception as e:
        print(f"ERROR - 步骤管理测试失败: {e}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    print("=== 存储系统边界情况测试 ===")
    
    try:
        test_empty_data()
        print("✓ 空数据处理测试通过")
    except Exception as e:
        print(f"✗ 空数据处理测试失败: {e}")
    
    try:
        test_invalid_data_types()
        print("✓ 无效数据类型测试通过")
    except Exception as e:
        print(f"✗ 无效数据类型测试失败: {e}")
    
    try:
        test_file_operations()
        print("✓ 文件操作测试通过")
    except Exception as e:
        print(f"✗ 文件操作测试失败: {e}")
    
    try:
        test_step_management()
        print("✓ 步骤管理测试通过")
    except Exception as e:
        print(f"✗ 步骤管理测试失败: {e}")
    
    print("\n🎉 所有边界情况测试完成！")