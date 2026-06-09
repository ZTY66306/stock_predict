"""回测风险指标离线测试。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from baostock_tool import backtest, strategy


@pytest.fixture
def bt_result(sample_df):
    sig = strategy.run_strategy("ma_cross", sample_df, {"short": 5, "long": 20})
    return backtest.BacktestEngine().run(sample_df, sig)


@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 200
    close = 10 + np.cumsum(np.random.randn(n) * 0.2)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1_000_000, 5_000_000, n)
    idx = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=idx)


def test_sortino_finite(bt_result):
    assert isinstance(bt_result.sortino, float)
    # 不应 NaN
    assert not np.isnan(bt_result.sortino)


def test_calmar_finite(bt_result):
    assert isinstance(bt_result.calmar, float)


def test_var_cvar(bt_result):
    var = bt_result.var(0.05)
    cvar = bt_result.cvar(0.05)
    assert 0 <= var <= 1.0
    assert 0 <= cvar <= 1.0
    # CVaR >= VaR
    assert cvar >= var


def test_alpha_beta_with_self(bt_result):
    # 跟自己比 alpha/beta 应接近 0 / 1
    bench_eq = bt_result.equity
    alpha, beta = bt_result.alpha_beta(bench_eq)
    assert abs(beta - 1.0) < 0.1
    # alpha 不应太大
    assert abs(alpha) < 0.5


def test_risk_metrics_keys(bt_result):
    metrics = bt_result.risk_metrics()
    expected = ["总收益率", "年化收益", "最大回撤", "夏普比率", "Sortino",
                "Calmar", "VaR(5%)", "CVaR(5%)", "胜率", "盈亏比", "交易次数"]
    for k in expected:
        assert k in metrics.index


def test_monthly_returns_shape(bt_result):
    m = bt_result.monthly_returns()
    assert isinstance(m, pd.DataFrame)
    # 行=年,列=月(出现的月份数 ≤ 12,数据范围决定)
    assert m.shape[0] >= 1
    assert 1 <= m.shape[1] <= 12


def test_yearly_returns(bt_result):
    y = bt_result.yearly_returns()
    assert isinstance(y, pd.Series)
    assert len(y) >= 1
