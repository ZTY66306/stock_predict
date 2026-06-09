"""通用工具:日期处理、字段类型转换、文件 IO。"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Iterable, Optional, Union

import pandas as pd


def to_date(s: Union[str, datetime, pd.Timestamp]) -> str:
    """统一把日期转成 baostock 要求的 'YYYY-MM-DD' 字符串。"""
    if isinstance(s, str):
        return s
    return pd.Timestamp(s).strftime("%Y-%m-%d")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def yesterday_str() -> str:
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def default_start(days_back: int = 365) -> str:
    return (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")


def ensure_dir(path: str) -> str:
    """确保目录存在,返回绝对路径。"""
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def safe_float(x, default: Optional[float] = None) -> Optional[float]:
    """安全地把 baostock 返回的字符串转 float;空串/非数字返回 default。"""
    if x is None or x == "" or (isinstance(x, str) and x.strip() in {"", "None"}):
        return default
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def safe_int(x, default: Optional[int] = None) -> Optional[int]:
    f = safe_float(x, default=None)
    if f is None:
        return default
    return int(f)


# baostock K 线常用字段的强制类型转换
KLINE_NUMERIC_FIELDS = (
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "turn", "tradestatus", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
    "isST", "adjustflag",
)


def normalize_kline_df(df: pd.DataFrame) -> pd.DataFrame:
    """把 baostock 字符串 K 线 DataFrame 转成数值类型,date 设为索引。"""
    if df.empty:
        return df
    for col in KLINE_NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df.set_index("date")
    return df


def to_records(df: pd.DataFrame, limit: Optional[int] = None) -> list[dict]:
    """把 DataFrame 转成 list[dict] 便于打印或 JSON 化。"""
    if limit is not None:
        df = df.head(limit)
    return df.reset_index().to_dict(orient="records")


def chunked(seq: Iterable, size: int):
    """把可迭代对象切分成 size 大小的块,用于批量查询时控制并发。"""
    chunk = []
    for item in seq:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
