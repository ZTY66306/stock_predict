"""数据缓存层离线测试。"""
import sys
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baostock_tool import data_cache


@pytest.fixture
def tmp_cache():
    d = tempfile.mkdtemp(prefix="bstock_cache_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_write_and_read(tmp_cache):
    idx = pd.bdate_range("2024-01-01", periods=30)
    df = pd.DataFrame({
        "open": np.arange(30, dtype=float),
        "close": np.arange(30, dtype=float) + 1,
        "high": np.arange(30, dtype=float) + 2,
        "low": np.arange(30, dtype=float) - 1,
        "volume": np.arange(30, dtype=float) * 100,
    }, index=idx)
    data_cache.write_cached("sh.600000", df, "d", "1", tmp_cache)
    out = data_cache.read_cached("sh.600000", "2024-01-01", "2024-02-29", "d", "1", tmp_cache)
    assert out is not None
    assert len(out) == 30
    assert out.iloc[0]["close"] == 1.0


def test_read_sliced(tmp_cache):
    idx = pd.bdate_range("2024-01-01", periods=100)
    df = pd.DataFrame({"close": np.arange(100, dtype=float)}, index=idx)
    data_cache.write_cached("sh.600000", df, "d", "1", tmp_cache)
    # 只读 30-60
    out = data_cache.read_cached("sh.600000", "2024-02-15", "2024-03-15", "d", "1", tmp_cache)
    assert out is not None
    # 至少包含 1.15-3.15 的数据
    assert len(out) >= 1
    assert out.index[0] >= pd.Timestamp("2024-02-15")


def test_clear(tmp_cache):
    idx = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=idx)
    data_cache.write_cached("sh.600000", df, "d", "1", tmp_cache)
    data_cache.write_cached("sh.600001", df, "d", "1", tmp_cache)
    n = data_cache.clear(tmp_cache)
    assert n == 2
    assert data_cache.read_cached("sh.600000", "2024-01-01", "2024-01-05", "d", "1", tmp_cache) is None


def test_cache_info(tmp_cache):
    info = data_cache.cache_info(tmp_cache)
    assert info["exists"] is True
    assert info["files"] == 0


def test_read_nonexistent_returns_none(tmp_cache):
    out = data_cache.read_cached("sh.999999", "2024-01-01", "2024-01-31", "d", "1", tmp_cache)
    assert out is None
