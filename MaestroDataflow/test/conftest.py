#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest 配置文件

提供测试的通用配置和fixture
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
try:
    import pytest
except ImportError:
    import sys
    sys.exit("pytest 未安装，请先运行 pip install pytest")

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def temp_dir():
    """创建临时目录用于测试"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture(scope="function")
def clean_cache():
    """清理缓存目录"""
    cache_dirs = ["./cache", "./test_cache"]
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
    yield
    # 测试后清理
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def sample_data():
    """提供测试用的样本数据"""
    return [
        {"name": "张三", "city": "北京", "salary": 50000},
        {"name": "李四", "city": "上海", "salary": 60000},
        {"name": "王五", "city": "广州", "salary": 70000},
        {"name": "赵六", "city": "深圳", "salary": 80000}
    ]