"""本地数据缓存层:把 baostock 的 K 线拉到本地,避免重复请求。

设计目标:
- 按 (code, frequency, adjustflag) 一文件,文件名为 hash;命中时按 start/end 切片返回
- 写入失败时回退到内存结果,不让缓存层抛错
- 提供 clear() 用于 CLI `cache clear` 强制刷新
- _CACHE_VERSION 在 schema 升级时手动 +1,旧文件自动失效
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1


def _cache_key(code: str, frequency: str, adjustflag: str) -> str:
    raw = f"v{_CACHE_VERSION}|{code}|{frequency}|{adjustflag}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _cache_path(code: str, frequency: str, adjustflag: str, cache_dir: str) -> str:
    safe_code = code.replace(".", "_").replace("/", "_")
    fname = f"{safe_code}_{frequency}_{adjustflag}_{_cache_key(code, frequency, adjustflag)}.csv"
    return os.path.join(ensure_dir(cache_dir), fname)


def read_cached(
    code: str,
    start: str,
    end: str,
    frequency: str,
    adjustflag: str,
    cache_dir: str = ".cache/kline",
) -> Optional[pd.DataFrame]:
    """从本地缓存读,按 start/end 切片;无缓存或不可读时返回 None。"""
    if not cache_dir:
        return None
    path = _cache_path(code, frequency, adjustflag, cache_dir)
    if not os.path.exists(path):
        return None
    try:
        # 第一列不一定是 "date" —— 兼容各种索引名
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        df.index.name = "date"
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        sliced = df.loc[(df.index >= s) & (df.index <= e)]
        return sliced
    except Exception as e:
        logger.debug("read_cached %s 失败: %s", code, e)
        return None


def write_cached(
    code: str,
    df: pd.DataFrame,
    frequency: str,
    adjustflag: str,
    cache_dir: str = ".cache/kline",
) -> None:
    """把整段 K 线写入本地,失败仅记日志不抛错。"""
    if not cache_dir or df.empty:
        return
    path = _cache_path(code, frequency, adjustflag, cache_dir)
    try:
        ensure_dir(cache_dir)
        # 写时统一把索引命名为 "date",便于 read 时识别
        out = df.copy()
        out.index.name = "date"
        # 合并写入:已存在则扩展日期区间,避免覆盖更早数据
        if os.path.exists(path):
            old = pd.read_csv(path, index_col=0, parse_dates=True)
            old.index.name = "date"
            merged = pd.concat([old, out]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
        else:
            merged = out.sort_index()
        merged.to_csv(path, index_label="date")
    except Exception as e:
        logger.debug("write_cached %s 失败: %s", code, e)


def clear(cache_dir: str = ".cache/kline") -> int:
    """清空缓存目录,返回删除的文件数。"""
    if not cache_dir or not os.path.isdir(cache_dir):
        return 0
    n = 0
    for fname in os.listdir(cache_dir):
        full = os.path.join(cache_dir, fname)
        if os.path.isfile(full):
            try:
                os.remove(full)
                n += 1
            except Exception:
                pass
    return n


def cache_info(cache_dir: str = ".cache/kline") -> dict:
    """查看缓存目录统计。"""
    info = {"dir": cache_dir, "exists": False, "files": 0, "size_mb": 0.0}
    if not os.path.isdir(cache_dir):
        return info
    info["exists"] = True
    total = 0
    n = 0
    for root, _, files in os.walk(cache_dir):
        for f in files:
            full = os.path.join(root, f)
            total += os.path.getsize(full)
            n += 1
    info["files"] = n
    info["size_mb"] = round(total / 1024 / 1024, 3)
    return info
