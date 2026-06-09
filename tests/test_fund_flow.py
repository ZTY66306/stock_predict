"""fund_flow 离线测试(优先 mock akshare,不依赖外网)。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from baostock_tool import fund_flow as ff


# ============ 工具 ============

@pytest.fixture(autouse=True)
def _restore_akshare():
    """每个测试结束后还原 akshare 模块状态。"""
    yield


# ============ _normalize_code ============

def test_normalize_code_with_prefix():
    assert ff._normalize_code("sh.600000") == ("600000", "sh")
    assert ff._normalize_code("sz.000001") == ("000001", "sz")
    assert ff._normalize_code("SH.688981") == ("688981", "sh")


def test_normalize_code_no_prefix():
    assert ff._normalize_code("600000") == ("600000", "")
    assert ff._normalize_code("000001") == ("000001", "")


# ============ akshare 缺失时行为 ============

def test_require_akshare_raises_when_missing(monkeypatch):
    """akshare 未装时,所有 fetcher 应抛 ImportError。"""
    monkeypatch.setattr(ff, "_HAS_AKSHARE", False)
    with pytest.raises(ImportError):
        ff.get_fund_flow("sh.600000")
    with pytest.raises(ImportError):
        ff.get_northbound()
    with pytest.raises(ImportError):
        ff.get_longhubang("20240101", "20240115")
    with pytest.raises(ImportError):
        ff.get_sector_fund_flow()


# ============ mock akshare 的端到端测试 ============

def _patch_ak(monkeypatch, fn_name, return_value):
    """把 akshare 的某个函数 monkey-patch 成返回固定 DataFrame。"""
    import akshare as ak
    monkeypatch.setattr(ak, fn_name, lambda *a, **k: return_value)


def test_get_fund_flow_with_mock(monkeypatch):
    df_mock = pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "主力净流入": [1e8, -5e7, 2e8],
        "超大单净流入": [5e7, -3e7, 1e8],
    })
    _patch_ak(monkeypatch, "stock_individual_fund_flow", df_mock)
    out = ff.get_fund_flow("sh.600000")
    assert len(out) == 3
    assert "主力净流入" in out.columns


def test_get_fund_flow_filters_by_date(monkeypatch):
    df_mock = pd.DataFrame({
        "日期": ["20240102", "20240115", "20240120", "20240201"],
        "主力净流入": [1, 2, 3, 4],
    })
    _patch_ak(monkeypatch, "stock_individual_fund_flow", df_mock)
    out = ff.get_fund_flow("sh.600000", start_date="2024-01-10", end_date="2024-01-31")
    assert len(out) == 2  # 1/15 + 1/20


def test_get_northbound_with_mock(monkeypatch):
    df_mock = pd.DataFrame({
        "类型": ["沪港通", "深港通"],
        "板块": ["沪股通", "深股通"],
        "成交净买额": [1.2e9, 8e8],
    })
    _patch_ak(monkeypatch, "stock_hsgt_fund_flow_summary_em", df_mock)
    out = ff.get_northbound()
    assert len(out) == 2
    assert "板块" in out.columns


def test_get_longhubang_with_mock(monkeypatch):
    df_mock = pd.DataFrame({
        "代码": ["000001", "600000"],
        "名称": ["平安银行", "浦发银行"],
        "上榜日": ["2024-01-15", "2024-01-15"],
        "解读": ["日涨幅偏离值达7%", "日涨幅偏离值达7%"],
    })
    _patch_ak(monkeypatch, "stock_lhb_detail_em", df_mock)
    out = ff.get_longhubang("20240115", "20240115")
    assert len(out) == 2
    assert "代码" in out.columns


def test_get_sector_fund_flow_with_mock(monkeypatch):
    df_mock = pd.DataFrame({
        "名称": ["电子", "医药", "银行"],
        "主力净流入": [5e8, -2e8, 1e9],
    })
    _patch_ak(monkeypatch, "stock_sector_fund_flow_rank", df_mock)
    out = ff.get_sector_fund_flow(indicator="今日", sector_type="行业资金流")
    assert len(out) == 3
    assert "主力净流入" in out.columns


def test_strong_capital_inflow_filters(monkeypatch):
    df_mock = pd.DataFrame({
        "名称": ["电子", "医药", "银行"],
        "主力净流入": [5e8, -2e8, 1e9],
    })
    _patch_ak(monkeypatch, "stock_sector_fund_flow_rank", df_mock)
    out = ff.strong_capital_inflow(threshold=0)
    # 电子(5e8)和银行(1e9)应保留,医药(-2e8)被过滤
    names = out["名称"].tolist()
    assert "医药" not in names
    assert "电子" in names
    assert "银行" in names


# ============ 缓存 ============

def test_clear_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ff, "_CACHE_DIR", str(tmp_path))
    # 创建一些假文件
    (tmp_path / "a.csv").write_text("x")
    (tmp_path / "b.csv").write_text("y")
    n = ff.clear_cache()
    assert n == 2
    assert list(tmp_path.iterdir()) == []  # 目录为空


def test_clear_cache_no_dir(monkeypatch, tmp_path):
    """缓存目录不存在时返回 0。"""
    monkeypatch.setattr(ff, "_CACHE_DIR", str(tmp_path / "no_such_dir"))
    assert ff.clear_cache() == 0
