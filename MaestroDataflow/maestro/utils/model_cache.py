"""
Model caching system for MaestroDataflow.
Provides caching capabilities for AI models and their outputs.
"""

import os
import json
import pickle
import hashlib
import time
from typing import Dict, Any, Optional, Union, List, Callable
from datetime import datetime, timedelta
import logging
from pathlib import Path
import threading
from collections import OrderedDict


class CacheEntry:
    """缓存条目类。"""

    def __init__(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化缓存条目。

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒）
            metadata: 元数据
        """
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
        self.access_count = 0
        self.last_accessed = self.created_at
        self.metadata = metadata or {}

    def is_expired(self) -> bool:
        """检查是否过期。"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def access(self):
        """记录访问。"""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "metadata": self.metadata
        }


class InMemoryCache:
    """内存缓存实现。"""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = None,
        cleanup_interval: int = 300  # 5分钟
    ):
        """
        初始化内存缓存。

        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认TTL（秒）
            cleanup_interval: 清理间隔（秒）
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

        self.logger = logging.getLogger(__name__)

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值。"""
        with self._lock:
            self._cleanup_if_needed()

            if key not in self._cache:
                return None

            entry = self._cache[key]

            if entry.is_expired():
                del self._cache[key]
                return None

            # 更新访问信息并移到末尾（LRU）
            entry.access()
            self._cache.move_to_end(key)

            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """设置缓存值。"""
        with self._lock:
            self._cleanup_if_needed()

            # 使用默认TTL
            if ttl is None:
                ttl = self.default_ttl

            # 创建缓存条目
            entry = CacheEntry(key, value, ttl)

            # 如果已存在，更新
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # 检查容量限制
                if len(self._cache) >= self.max_size:
                    # 移除最旧的条目
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

                self._cache[key] = entry

            return True

    def delete(self, key: str) -> bool:
        """删除缓存条目。"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """清空缓存。"""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息。"""
        with self._lock:
            total_entries = len(self._cache)
            expired_count = sum(1 for entry in self._cache.values() if entry.is_expired())

            return {
                "total_entries": total_entries,
                "expired_entries": expired_count,
                "max_size": self.max_size,
                "memory_usage_mb": self._estimate_memory_usage() / (1024 * 1024)
            }

    def _cleanup_if_needed(self):
        """如果需要则清理过期条目。"""
        current_time = time.time()
        if current_time - self._last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = current_time

    def _cleanup_expired(self):
        """清理过期条目。"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _estimate_memory_usage(self) -> int:
        """估算内存使用量（字节）。"""
        # 简单估算，实际使用量可能不同
        total_size = 0
        for entry in self._cache.values():
            try:
                total_size += len(pickle.dumps(entry.value))
            except:
                total_size += 1024  # 默认估算
        return total_size


class DiskCache:
    """磁盘缓存实现。"""

    def __init__(
        self,
        cache_dir: str,
        max_size_mb: int = 1000,
        default_ttl: Optional[int] = None
    ):
        """
        初始化磁盘缓存。

        Args:
            cache_dir: 缓存目录
            max_size_mb: 最大缓存大小（MB）
            default_ttl: 默认TTL（秒）
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl

        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 元数据文件
        self.metadata_file = self.cache_dir / "cache_metadata.json"

        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

        # 加载元数据
        self._load_metadata()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值。"""
        with self._lock:
            cache_file = self._get_cache_file_path(key)

            if not cache_file.exists():
                return None

            # 检查是否过期
            if self._is_expired(key):
                self._delete_cache_file(key)
                return None

            try:
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)

                # 更新访问信息
                self._update_access_info(key)

                return value

            except Exception as e:
                self.logger.error(f"Failed to load cache file {cache_file}: {e}")
                self._delete_cache_file(key)
                return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """设置缓存值。"""
        with self._lock:
            try:
                # 使用默认TTL
                if ttl is None:
                    ttl = self.default_ttl

                # 保存到文件
                cache_file = self._get_cache_file_path(key)
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)

                # 更新元数据
                self._update_metadata(key, ttl, cache_file.stat().st_size)

                # 检查磁盘空间限制
                self._cleanup_if_needed()

                return True

            except Exception as e:
                self.logger.error(f"Failed to save cache file for key {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """删除缓存条目。"""
        with self._lock:
            return self._delete_cache_file(key)

    def clear(self):
        """清空缓存。"""
        with self._lock:
            # 删除所有缓存文件
            for cache_file in self.cache_dir.glob("cache_*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    self.logger.error(f"Failed to delete cache file {cache_file}: {e}")

            # 清空元数据
            self.metadata = {}
            self._save_metadata()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息。"""
        with self._lock:
            total_size = sum(
                entry.get("size", 0) for entry in self.metadata.values()
            )

            expired_count = sum(
                1 for key in self.metadata.keys() if self._is_expired(key)
            )

            return {
                "total_entries": len(self.metadata),
                "expired_entries": expired_count,
                "total_size_mb": total_size / (1024 * 1024),
                "max_size_mb": self.max_size_mb,
                "cache_dir": str(self.cache_dir)
            }

    def _get_cache_file_path(self, key: str) -> Path:
        """获取缓存文件路径。"""
        # 使用MD5哈希作为文件名
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"cache_{key_hash}.pkl"

    def _load_metadata(self):
        """加载元数据。"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load cache metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """保存元数据。"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")

    def _update_metadata(self, key: str, ttl: Optional[int], file_size: int):
        """更新元数据。"""
        now = datetime.now()

        self.metadata[key] = {
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=ttl)).isoformat() if ttl else None,
            "access_count": 1,
            "last_accessed": now.isoformat(),
            "size": file_size
        }

        self._save_metadata()

    def _update_access_info(self, key: str):
        """更新访问信息。"""
        if key in self.metadata:
            self.metadata[key]["access_count"] = self.metadata[key].get("access_count", 0) + 1
            self.metadata[key]["last_accessed"] = datetime.now().isoformat()
            self._save_metadata()

    def _is_expired(self, key: str) -> bool:
        """检查是否过期。"""
        if key not in self.metadata:
            return True

        expires_at = self.metadata[key].get("expires_at")
        if expires_at is None:
            return False

        return datetime.now() > datetime.fromisoformat(expires_at)

    def _delete_cache_file(self, key: str) -> bool:
        """删除缓存文件。"""
        try:
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                cache_file.unlink()

            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()

            return True

        except Exception as e:
            self.logger.error(f"Failed to delete cache file for key {key}: {e}")
            return False

    def _cleanup_if_needed(self):
        """如果需要则清理缓存。"""
        # 清理过期条目
        expired_keys = [key for key in self.metadata.keys() if self._is_expired(key)]
        for key in expired_keys:
            self._delete_cache_file(key)

        # 检查磁盘空间
        total_size = sum(entry.get("size", 0) for entry in self.metadata.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024

        if total_size > max_size_bytes:
            # 按最后访问时间排序，删除最旧的条目
            sorted_keys = sorted(
                self.metadata.keys(),
                key=lambda k: self.metadata[k].get("last_accessed", "")
            )

            for key in sorted_keys:
                if total_size <= max_size_bytes:
                    break

                file_size = self.metadata[key].get("size", 0)
                if self._delete_cache_file(key):
                    total_size -= file_size


class ModelCache:
    """
    AI模型缓存系统。
    支持内存和磁盘缓存。
    """

    def __init__(
        self,
        cache_type: str = "memory",
        cache_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化模型缓存系统。

        Args:
            cache_type: 缓存类型（"memory", "disk", "hybrid"）
            cache_config: 缓存配置
        """
        self.cache_type = cache_type
        self.cache_config = cache_config or {}

        self.logger = logging.getLogger(__name__)

        # 初始化缓存实例
        if cache_type == "memory":
            self.cache = InMemoryCache(**self.cache_config)
        elif cache_type == "disk":
            self.cache = DiskCache(**self.cache_config)
        elif cache_type == "hybrid":
            # 混合缓存：内存 + 磁盘
            memory_config = self.cache_config.get("memory", {})
            disk_config = self.cache_config.get("disk", {})

            self.memory_cache = InMemoryCache(**memory_config)
            self.disk_cache = DiskCache(**disk_config)
            self.cache = None  # 使用自定义逻辑
        else:
            raise ValueError(f"Unsupported cache type: {cache_type}")

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值。"""
        if self.cache_type == "hybrid":
            # 先查内存缓存
            value = self.memory_cache.get(key)
            if value is not None:
                return value

            # 再查磁盘缓存
            value = self.disk_cache.get(key)
            if value is not None:
                # 将热点数据加载到内存
                self.memory_cache.set(key, value)
                return value

            return None
        else:
            return self.cache.get(key)

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_level: str = "both"  # "memory", "disk", "both"
    ) -> bool:
        """设置缓存值。"""
        if self.cache_type == "hybrid":
            success = True

            if cache_level in ["memory", "both"]:
                success &= self.memory_cache.set(key, value, ttl)

            if cache_level in ["disk", "both"]:
                success &= self.disk_cache.set(key, value, ttl)

            return success
        else:
            return self.cache.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """删除缓存条目。"""
        if self.cache_type == "hybrid":
            memory_result = self.memory_cache.delete(key)
            disk_result = self.disk_cache.delete(key)
            return memory_result or disk_result
        else:
            return self.cache.delete(key)

    def clear(self):
        """清空缓存。"""
        if self.cache_type == "hybrid":
            self.memory_cache.clear()
            self.disk_cache.clear()
        else:
            self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息。"""
        if self.cache_type == "hybrid":
            return {
                "cache_type": self.cache_type,
                "memory_cache": self.memory_cache.get_stats(),
                "disk_cache": self.disk_cache.get_stats()
            }
        else:
            stats = self.cache.get_stats()
            stats["cache_type"] = self.cache_type
            return stats

    def cache_model_output(
        self,
        model_name: str,
        input_data: Any,
        output_data: Any,
        ttl: Optional[int] = None
    ) -> str:
        """
        缓存模型输出。

        Args:
            model_name: 模型名称
            input_data: 输入数据
            output_data: 输出数据
            ttl: 生存时间

        Returns:
            str: 缓存键
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(model_name, input_data)

        # 缓存输出
        self.set(cache_key, output_data, ttl)

        return cache_key

    def get_model_output(
        self,
        model_name: str,
        input_data: Any
    ) -> Optional[Any]:
        """
        获取缓存的模型输出。

        Args:
            model_name: 模型名称
            input_data: 输入数据

        Returns:
            Optional[Any]: 缓存的输出数据
        """
        cache_key = self._generate_cache_key(model_name, input_data)
        return self.get(cache_key)

    def _generate_cache_key(self, model_name: str, input_data: Any) -> str:
        """
        生成缓存键。

        Args:
            model_name: 模型名称
            input_data: 输入数据

        Returns:
            str: 缓存键
        """
        # 将输入数据序列化并计算哈希
        try:
            input_str = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            # 如果无法JSON序列化，使用字符串表示
            input_str = str(input_data)

        input_hash = hashlib.md5(input_str.encode()).hexdigest()

        return f"{model_name}:{input_hash}"


# 全局模型缓存实例
_global_model_cache = None


def get_global_model_cache() -> ModelCache:
    """获取全局模型缓存实例。"""
    global _global_model_cache

    if _global_model_cache is None:
        _global_model_cache = ModelCache(
            cache_type="hybrid",
            cache_config={
                "memory": {"max_size": 100, "default_ttl": 3600},
                "disk": {"cache_dir": "./cache/models", "max_size_mb": 500, "default_ttl": 86400}
            }
        )

    return _global_model_cache


def cache_model_output(
    model_name: str,
    input_data: Any,
    output_data: Any,
    ttl: Optional[int] = None
) -> str:
    """缓存模型输出的便捷函数。"""
    cache = get_global_model_cache()
    return cache.cache_model_output(model_name, input_data, output_data, ttl)


def get_cached_model_output(
    model_name: str,
    input_data: Any
) -> Optional[Any]:
    """获取缓存模型输出的便捷函数。"""
    cache = get_global_model_cache()
    return cache.get_model_output(model_name, input_data)