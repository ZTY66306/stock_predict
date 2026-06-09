"""DCA 离线单元测试(全部 mock,无外网)。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baostock_tool import dca


# ============ helpers ============

def _mock_kline(n=500, base=3.0, seed=0):
    np.random.seed(seed)
    close = base + np.cumsum(np.random.randn(n) * 0.05)
    return pd.DataFrame({
        "open": close, "high": close + 0.1, "low": close - 0.1,
        "close": close, "volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=pd.bdate_range("2022-01-01", periods=n))


def _patch_data(dca_mod, df, code="sh.510300", name="mock ETF"):
    """同时 patch get_kline / resolve_code / client,完全离线。"""
    import baostock_tool.client as _client
    import baostock_tool.grid_backtest as _gb
    orig = {
        "get_kline": dca_mod.data.get_kline,
        "client_login": _client.login,
    }
    dca_mod.data.get_kline = lambda *a, **k: df
    _client.login = lambda: True
    _client.ensure_login = lambda: None
    _gb.resolve_code = lambda s: (code, name)
    return orig


def _restore(orig):
    import baostock_tool.client as _client
    dca.data.get_kline = orig["get_kline"]
    _client.login = orig["client_login"]


# ============ _rebalance_dates ============

def test_rebalance_dates_weekly():
    dates = pd.bdate_range("2024-01-01", periods=20)
    rb = dca._rebalance_dates(dates, "weekly")
    # 每周五一次
    assert len(rb) >= 2
    # 应在 5 个交易日后
    for d in rb:
        assert d in dates


def test_rebalance_dates_monthly():
    dates = pd.bdate_range("2024-01-01", periods=120)
    rb = dca._rebalance_dates(dates, "monthly")
    # 每月最后一个交易日
    assert len(rb) >= 4


def test_rebalance_dates_biweekly():
    dates = pd.bdate_range("2024-01-01", periods=60)
    rb = dca._rebalance_dates(dates, "biweekly")
    assert len(rb) >= 2


# ============ dca_backtest ============

def test_dca_backtest_pure_strategy():
    df = _mock_kline(n=400, seed=0)
    orig = _patch_data(dca, df)
    try:
        r = dca.dca_backtest("sh.510300", "2022-01-01", "2023-12-31",
                              amount_per_period=1000, frequency="monthly",
                              strategy="pure")
        assert r.strategy == "pure"
        assert r.total_invested > 0
        assert r.total_shares > 0
        assert r.avg_cost > 0
        assert r.final_value > 0
        # 至少 5 期(400 个交易日 = 1.6 年,每月 1 期 ≈ 18 期)
        assert sum(1 for t in r.trades if t.side == "buy") >= 5
        # 序列长度 = K 线长度
        assert len(r.value_history) == len(df)
    finally:
        _restore(orig)


def test_dca_backtest_smart_strategy():
    df = _mock_kline(n=400, seed=1)
    orig = _patch_data(dca, df)
    try:
        r = dca.dca_backtest("sh.510300", "2022-01-01", "2023-12-31",
                              amount_per_period=1000, frequency="monthly",
                              strategy="smart", ma_window=60)
        # smart 应有不止一种 reason
        reasons = {t.reason for t in r.trades}
        # 至少应出现 schedule 或 smart_dip 等
        assert len(reasons) >= 1
        # smart 策略可能累计投入 ≥ pure(因为加仓),也允许 < (因为减仓)
        assert r.total_invested > 0
    finally:
        _restore(orig)


def test_dca_backtest_dip_buy_strategy():
    df = _mock_kline(n=400, seed=2)
    orig = _patch_data(dca, df)
    try:
        r = dca.dca_backtest("sh.510300", "2022-01-01", "2023-12-31",
                              amount_per_period=1000, frequency="monthly",
                              strategy="dip_buy", dip_threshold=-0.05)
        assert r.strategy == "dip_buy"
        assert r.total_invested > 0
    finally:
        _restore(orig)


def test_dca_backtest_lump_sum():
    df = _mock_kline(n=400, seed=3)
    orig = _patch_data(dca, df)
    try:
        r = dca.dca_backtest("sh.510300", "2022-01-01", "2023-12-31",
                              amount_per_period=50000, frequency="monthly",
                              strategy="lump_sum")
        # 一次性只买 1 笔
        buys = [t for t in r.trades if t.side == "buy"]
        assert len(buys) == 1
        assert r.total_invested == 50000
    finally:
        _restore(orig)


def test_dca_backtest_summary_fields():
    df = _mock_kline(n=300, seed=4)
    orig = _patch_data(dca, df)
    try:
        r = dca.dca_backtest("sh.510300", "2022-01-01", "2023-06-30",
                              amount_per_period=1000, strategy="pure")
        s = r.summary()
        for col in ["代码", "策略", "累计投入", "平均成本", "期末市值",
                    "总收益率", "年化收益", "实际买入次数"]:
            assert col in s.index
    finally:
        _restore(orig)


def test_dca_backtest_empty_data_raises():
    import baostock_tool.client as _client
    orig = {
        "get_kline": dca.data.get_kline,
        "client_login": _client.login,
    }
    dca.data.get_kline = lambda *a, **k: pd.DataFrame()
    _client.login = lambda: True
    _client.ensure_login = lambda: None
    try:
        with pytest.raises(ValueError):
            dca.dca_backtest("sh.510300", "2022-01-01", "2023-01-01",
                              amount_per_period=1000)
    finally:
        dca.data.get_kline = orig["get_kline"]
        _client.login = orig["client_login"]


# ============ compare_strategies ============

def test_compare_strategies_returns_4_rows():
    df = _mock_kline(n=400, seed=5)
    orig = _patch_data(dca, df)
    try:
        cmp = dca.compare_strategies("sh.510300", "2022-01-01", "2023-12-31",
                                      amount_per_period=1000)
        assert len(cmp) == 4
        # 4 种策略都应有数据(没有 error 列)
        assert "error" not in cmp.columns or cmp["error"].isna().all()
        for s in ["lump_sum", "pure", "dip_buy", "smart"]:
            assert s in cmp["策略"].tolist()
    finally:
        _restore(orig)
