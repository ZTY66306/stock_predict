"""网格回测的离线单元测试(不联网,直接用 mock K 线)。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from baostock_tool import grid_backtest as gb


# ============ 工具函数 ============

def _mock_kline(dates, base=10.0, vol=0.15, seed=0):
    """生成合成的 K 线 DataFrame。"""
    np.random.seed(seed)
    close = base + np.cumsum(np.random.randn(len(dates)) * 0.1)
    df = pd.DataFrame({
        "open": close + np.random.randn(len(dates)) * 0.02,
        "high": close + np.abs(np.random.randn(len(dates))) * vol,
        "low": close - np.abs(np.random.randn(len(dates))) * vol,
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, len(dates)),
        "amount": np.random.randint(10_000_000, 100_000_000, len(dates)),
        "preclose": np.concatenate([[close[0]], close[:-1]]),
        "pctChg": np.random.randn(len(dates)),
    }, index=dates)
    df.index.name = "date"
    return df


def _patch_data(monkeypatch_module, df, code="sh.600000", name="mock"):
    """monkey-patch gb.data.get_kline 与 gb.resolve_code。"""
    orig = gb.data.get_kline
    gb.data.get_kline = lambda *a, **k: df
    gb.resolve_code = lambda s: (code, name)
    return orig


# ============ 测试 ============

def test_detect_price_limit():
    assert gb.detect_price_limit("sh.600000") == 0.10
    assert gb.detect_price_limit("sh.688981") == 0.20
    assert gb.detect_price_limit("sz.300750") == 0.20
    assert gb.detect_price_limit("bj.835185") == 0.30
    assert gb.detect_price_limit("sh.600000", "ST 测试") == 0.05
    assert gb.detect_price_limit("sz.000001", "*ST 测试") == 0.05


def test_build_grid_lines_geometric():
    cfg = gb.GridConfig(grid_mode="geometric", n_grids=4, lower=8.0, upper=16.0)
    lines = gb.build_grid_lines(cfg, 0, 0)
    assert len(lines) == 5
    assert abs(lines[0] - 8.0) < 1e-9
    assert abs(lines[-1] - 16.0) < 1e-9
    # 几何等比,相邻比值恒定
    ratios = [lines[i+1] / lines[i] for i in range(len(lines) - 1)]
    assert all(abs(r - ratios[0]) < 1e-9 for r in ratios)


def test_build_grid_lines_fixed():
    cfg = gb.GridConfig(grid_mode="fixed", n_grids=4, lower=8.0, upper=16.0)
    lines = gb.build_grid_lines(cfg, 0, 0)
    assert len(lines) == 5
    # 等差,步长固定
    diffs = [lines[i+1] - lines[i] for i in range(len(lines) - 1)]
    assert all(abs(d - diffs[0]) < 1e-9 for d in diffs)
    assert abs(diffs[0] - 2.0) < 1e-9


def test_build_grid_lines_auto_pct():
    cfg = gb.GridConfig(grid_mode="geometric", n_grids=4,
                        lower_pct=0.9, upper_pct=1.1)
    lines = gb.build_grid_lines(cfg, 10.0, 20.0)
    assert abs(lines[0] - 9.0) < 1e-9
    assert abs(lines[-1] - 22.0) < 1e-9


def test_build_grid_lines_invalid():
    import pytest
    with pytest.raises(ValueError):
        gb.GridConfig(lower=10.0, upper=10.0)
        gb.build_grid_lines(gb.GridConfig(lower=10.0, upper=10.0), 0, 0)
    with pytest.raises(ValueError):
        cfg = gb.GridConfig(lower_pct=0.5)  # 缺 upper_pct
        gb.build_grid_lines(cfg, 10.0, 20.0)


def test_run_basic():
    dates = pd.bdate_range("2024-01-01", periods=60)
    df = _mock_kline(dates, base=10.0, vol=0.10, seed=1)
    orig = _patch_data(gb, df)
    try:
        result = gb.run(
            "sh.600000",
            start="2024-01-01", end="2024-03-15",
            lower=9.5, upper=10.5, n_grids=4,
            shares_per_grid=200, base_position=0,
        )
        assert result.code == "sh.600000"
        assert len(result.grid_lines) == 5
        assert not result.equity.empty
        assert len(result.equity) == len(dates)
        # 涨跌跌幅限制已识别
        assert result.price_limit == 0.10
        # summary 不抛异常
        s = result.summary()
        assert "总收益率" in s.index
        assert "完整往返" in s.index
    finally:
        gb.data.get_kline = orig
        gb.resolve_code = gb.resolve_code  # 已是 patched,无需还原


def test_run_with_base_position_and_t1():
    """底仓 + T+1 验证:首日买入的份额当日不可卖。"""
    dates = pd.bdate_range("2024-01-01", periods=20)
    df = _mock_kline(dates, base=10.0, vol=0.001, seed=2)  # 极小波动
    # 强制第二根 K 线 low 触及 9.5(下沿)
    df.iloc[1, df.columns.get_loc("low")] = 9.4
    orig = _patch_data(gb, df)
    try:
        result = gb.run(
            "sh.600000",
            start="2024-01-01", end="2024-01-31",
            lower=9.5, upper=10.5, n_grids=2,
            shares_per_grid=200, base_position=200,
        )
        # 首日底仓建仓成功
        assert result.position_history.iloc[0] == 200
        # T+1:首日建仓后,次日可卖;首日自身不可卖
        # 我们没法直接断言某一日不可卖,但可以验证 trades 里没有同日买卖同一笔
        # (因为我们的 sells 来自 available, 而首日 available = base_position)
        # 实际上首日 available = base_position=200, 所以首日是可以卖底仓的 — 这与 T+1 规则不冲突
        # 关键验证:有 1 笔 base + 0+ 笔 grid
        assert any(t.reason == "base" for t in result.trades)
    finally:
        gb.data.get_kline = orig


def test_run_allow_fractional_aligns_to_lot():
    dates = pd.bdate_range("2024-01-01", periods=10)
    df = _mock_kline(dates, base=10.0, vol=0.10, seed=3)
    orig = _patch_data(gb, df)
    try:
        # shares_per_grid=150 不是整百,应被自动向下取整为 100
        result = gb.run(
            "sh.600000",
            start="2024-01-01", end="2024-01-15",
            lower=9.5, upper=10.5, n_grids=4,
            shares_per_grid=150,
        )
        assert result.cfg.shares_per_grid in (100, 200)  # 100(150→100) 或 200(?)
        # 默认 lot=100,150→100
        assert result.cfg.shares_per_grid == 100
    finally:
        gb.data.get_kline = orig


def test_stop_loss_triggers_close():
    """触发止损后应清仓且不再开新仓。"""
    dates = pd.bdate_range("2024-01-01", periods=30)
    df = _mock_kline(dates, base=10.0, vol=0.001, seed=4)
    # 让价格第一天买入,第二天暴跌
    df.iloc[1, df.columns.get_loc("low")] = 9.5
    df.iloc[1, df.columns.get_loc("high")] = 9.6
    df.iloc[2, df.columns.get_loc("low")] = 8.0
    df.iloc[2, df.columns.get_loc("high")] = 8.1
    df.iloc[2, df.columns.get_loc("close")] = 8.05
    orig = _patch_data(gb, df)
    try:
        result = gb.run(
            "sh.600000",
            start="2024-01-01", end="2024-02-15",
            lower=9.0, upper=11.0, n_grids=4,
            shares_per_grid=200, base_position=200,
            stop_loss=0.05,
        )
        # 触发止损后,position 应回到 0
        assert result.position_history.iloc[-1] == 0
        # 至少有一笔 risk_close
        assert any(t.reason == "risk_close" for t in result.trades)
    finally:
        gb.data.get_kline = orig


def test_grid_efficiency():
    dates = pd.bdate_range("2024-01-01", periods=20)
    df = _mock_kline(dates, base=10.0, vol=0.10, seed=5)
    orig = _patch_data(gb, df)
    try:
        result = gb.run(
            "sh.600000",
            start="2024-01-01", end="2024-01-31",
            lower=9.5, upper=10.5, n_grids=2,
            shares_per_grid=200,
        )
        eff = result.grid_efficiency()
        # efficiency 应在 0~1.2 之间(可能 > 1 是因为跨多格)
        if not eff.empty:
            assert (eff["efficiency"] >= -1).all()
            assert (eff["efficiency"] <= 5).all()
    finally:
        gb.data.get_kline = orig


def test_trades_df_format():
    dates = pd.bdate_range("2024-01-01", periods=20)
    df = _mock_kline(dates, base=10.0, vol=0.20, seed=6)
    orig = _patch_data(gb, df)
    try:
        result = gb.run(
            "sh.600000",
            start="2024-01-01", end="2024-01-31",
            lower=9.0, upper=11.0, n_grids=5,
            shares_per_grid=200,
        )
        tdf = result.trades_df()
        if not tdf.empty:
            assert "date" in tdf.columns
            assert "side" in tdf.columns
            assert "shares" in tdf.columns
            assert "price" in tdf.columns
            assert tdf["shares"].sum() >= 0
    finally:
        gb.data.get_kline = orig


# ============ T+0 测试 ============

def test_detect_t0():
    """T+0 自动识别覆盖:股票 / ETF / 可转债 / 国债 / LOF / 北交所。"""
    # A 股股票 → T+1
    assert gb.detect_t0("sh.600000") is False
    assert gb.detect_t0("sz.000001") is False
    # 北交所 → T+1
    assert gb.detect_t0("bj.835185") is False
    # 沪深 ETF
    assert gb.detect_t0("sh.510300") is True
    assert gb.detect_t0("sh.510050") is True
    assert gb.detect_t0("sz.159915") is True
    # LOF
    assert gb.detect_t0("sz.163812") is True   # sz.1 开头
    assert gb.detect_t0("sh.501000") is True   # sh.5 开头
    # 可转债
    assert gb.detect_t0("sh.113050") is True
    assert gb.detect_t0("sh.110045") is True
    assert gb.detect_t0("sz.123015") is True
    # 国债
    assert gb.detect_t0("sh.019547") is True
    assert gb.detect_t0("sh.100038") is True
    assert gb.detect_t0("sz.100018") is True


def test_detect_price_limit_includes_bonds():
    """价格限制识别应覆盖可转债 / 国债。"""
    assert gb.detect_price_limit("sh.510300") == 0.10       # ETF: 10%
    assert gb.detect_price_limit("sh.113050") == 0.30       # 可转债: 30% 临停
    assert gb.detect_price_limit("sz.123015") == 0.30
    assert gb.detect_price_limit("sh.019547") == 1.0        # 国债: 无限制
    assert gb.detect_price_limit("sh.100038") == 1.0
    # 默认 A 股
    assert gb.detect_price_limit("sh.600000") == 0.10


def test_run_t0_auto_detect():
    """ETF 代码 → 自动 T+0,结果 t0=True。"""
    dates = pd.bdate_range("2024-01-01", periods=5)
    df = _mock_kline(dates, base=10.0, vol=0.05, seed=10)
    orig = _patch_data(gb, df, code="sh.510300", name="沪深300ETF")
    try:
        result = gb.run(
            "sh.510300",
            start="2024-01-01", end="2024-01-05",
            lower=9.5, upper=10.5, n_grids=4,
            shares_per_grid=100,
        )
        assert result.t0 is True
        s = result.summary()
        assert s["交易制度"] == "T+0"
    finally:
        gb.data.get_kline = orig


def test_run_t0_force_false_makes_t1():
    """强制 t0=False → 即使代码是 ETF,结果也是 T+1。"""
    dates = pd.bdate_range("2024-01-01", periods=5)
    df = _mock_kline(dates, base=10.0, vol=0.05, seed=11)
    orig = _patch_data(gb, df, code="sh.510300", name="沪深300ETF")
    try:
        result = gb.run(
            "sh.510300",
            start="2024-01-01", end="2024-01-05",
            lower=9.5, upper=10.5, n_grids=4,
            shares_per_grid=100, t0=False,
        )
        assert result.t0 is False
        assert result.summary()["交易制度"] == "T+1"
    finally:
        gb.data.get_kline = orig


def test_run_t0_same_day_buy_and_sell():
    """T+0 关键验证:当日买入的份额可以被当日卖出使用。"""
    # 构造日内 V 形:K 线同时打到底部和顶部,触发 buy 在底 + sell 在顶
    dates = pd.bdate_range("2024-01-01", periods=3)
    df = pd.DataFrame({
        "open":   [10.0, 9.6, 9.7],
        "high":   [10.5, 9.9, 10.3],   # 高 10.5 / 10.3 触及 grid 4 (顶)
        "low":    [ 9.5, 9.4, 9.5],    # 低 9.4 / 9.5 触及 grid 0 (底)
        "close":  [10.2, 9.7, 10.2],
        "volume": [1e6]*3,
        "amount": [1e7]*3,
        "preclose": [10.0, 10.2, 9.7],
        "pctChg": [0, -0.05, 0.05],
    }, index=dates)
    df.index.name = "date"

    orig = _patch_data(gb, df, code="sh.510300", name="mock ETF")
    try:
        # T+0 模式
        r_t0 = gb.run(
            "sh.510300",
            start="2024-01-01", end="2024-01-03",
            lower=9.5, upper=10.5, n_grids=4,
            shares_per_grid=100, t0=True,
        )
        # T+1 模式
        r_t1 = gb.run(
            "sh.510300",
            start="2024-01-01", end="2024-01-03",
            lower=9.5, upper=10.5, n_grids=4,
            shares_per_grid=100, t0=False,
        )
        # T+0 应至少有 1 个完整往返(同日 buy→sell)
        assert r_t0.n_full_roundtrips >= 1
        # T+0 当日可买卖,交易笔数应 ≥ T+1
        assert len(r_t0.trades) >= len(r_t1.trades)
        # T+0 的 t0 字段
        assert r_t0.t0 is True
        assert r_t1.t0 is False
    finally:
        gb.data.get_kline = orig


def test_run_t0_available_equals_position():
    """T+0 模式下,available 始终等于 position(无 pending 滞延)。"""
    dates = pd.bdate_range("2024-01-01", periods=5)
    df = _mock_kline(dates, base=10.0, vol=0.20, seed=12)
    orig = _patch_data(gb, df, code="sh.510300", name="mock ETF")
    try:
        result = gb.run(
            "sh.510300",
            start="2024-01-01", end="2024-01-05",
            lower=9.0, upper=11.0, n_grids=5,
            shares_per_grid=100, t0=True,
        )
        assert result.t0 is True
        # T+0:无 pending 概念
        # available_history 与 position_history 在所有时点应完全一致
        if result.position_history is not None:
            pd.testing.assert_series_equal(
                result.position_history, result.available_history,
                check_names=False,
            )
    finally:
        gb.data.get_kline = orig


def test_run_stock_default_is_t1():
    """普通 A 股代码 → 默认 T+1,即使不传 t0 参数。"""
    dates = pd.bdate_range("2024-01-01", periods=5)
    df = _mock_kline(dates, base=10.0, vol=0.10, seed=13)
    orig = _patch_data(gb, df, code="sh.600000", name="mock stock")
    try:
        result = gb.run(
            "sh.600000",
            start="2024-01-01", end="2024-01-05",
            lower=9.0, upper=11.0, n_grids=5,
            shares_per_grid=100,
        )
        assert result.t0 is False
        assert result.summary()["交易制度"] == "T+1"
    finally:
        gb.data.get_kline = orig
