"""单元测试:技术指标与回测引擎的纯算法部分(不联网)。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from baostock_tool import indicators as ind
from baostock_tool import strategy, backtest


@pytest.fixture
def sample_df():
    """构造一只模拟股票:60 个交易日,收盘价在 10~20 之间"""
    np.random.seed(42)
    n = 60
    close = 10 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    open_ = close + np.random.randn(n) * 0.05
    volume = np.random.randint(1_000_000, 10_000_000, n)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=idx)


def test_ma(sample_df):
    ma5 = ind.MA(sample_df["close"], 5)
    assert len(ma5) == len(sample_df)
    assert ma5.iloc[-1] == pytest.approx(sample_df["close"].iloc[-5:].mean(), rel=1e-6)


def test_macd(sample_df):
    dif, dea, hist = ind.MACD(sample_df["close"])
    assert (dif - dea).iloc[-1] == pytest.approx(hist.iloc[-1] / 2)


def test_rsi_range(sample_df):
    rsi = ind.RSI(sample_df["close"], 14)
    assert rsi.dropna().between(0, 100).all()


def test_boll_order(sample_df):
    mid, up, lo = ind.BOLL(sample_df["close"])
    assert (up >= mid).all()
    assert (mid >= lo).all()


def test_atr_positive(sample_df):
    atr = ind.ATR(sample_df)
    assert (atr.dropna() >= 0).all()


def test_add_all_columns(sample_df):
    out = ind.add_all(sample_df)
    for col in ["MA5", "MA20", "MACD_DIF", "KDJ_K", "RSI14", "BOLL_MID", "ATR14", "CCI", "OBV", "ret_1"]:
        assert col in out.columns


def test_golden_cross(sample_df):
    # 短均线 [1, 2, 3, 4, 5], 长均线 [5, 4, 3, 2, 1]: 在 iloc=3 出现金叉
    s = pd.Series([1, 2, 3, 4, 5])
    l = pd.Series([5, 4, 3, 2, 1])
    gc = ind.golden_cross(s, l)
    assert gc.iloc[3]
    assert gc.sum() == 1
    # 反过来测死叉
    dc = ind.death_cross(s, l)
    assert dc.sum() == 0  # 此处短均线从未下穿长均线


def test_strategy_signal_format(sample_df):
    for name in strategy.STRATEGIES:
        sig = strategy.run_strategy(name, sample_df)
        assert "signal" in sig.columns
        assert "position" in sig.columns
        assert sig["signal"].isin([-1, 0, 1]).all()


def test_backtest_runs(sample_df):
    sig = strategy.run_strategy("ma_cross", sample_df, {"short": 5, "long": 20})
    engine = backtest.BacktestEngine(backtest.BacktestConfig(initial_cash=100000))
    result = engine.run(sample_df, sig)
    assert len(result.equity) == len(sample_df)
    assert result.equity.iloc[0] == pytest.approx(100000, rel=1e-6)
    s = result.summary()
    assert "总收益率" in s.index
    assert "最大回撤" in s.index
    assert "夏普比率" in s.index
