"""组合回测离线测试。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from baostock_tool import portfolio, backtest


@pytest.fixture
def price_df():
    np.random.seed(0)
    dates = pd.bdate_range("2024-01-01", periods=120)
    return pd.DataFrame({
        "A": 10 + np.cumsum(np.random.randn(120) * 0.1),
        "B": 20 + np.cumsum(np.random.randn(120) * 0.1),
        "C": 15 + np.cumsum(np.random.randn(120) * 0.1),
    }, index=dates)


def test_equal_weight():
    w = portfolio.equal_weight(["a", "b", "c"], pd.Timestamp.now())
    assert (w == 1 / 3).all()


def test_risk_parity_returns_weights():
    np.random.seed(0)
    n = 60
    rets = pd.DataFrame({
        "A": np.random.randn(n) * 0.01,
        "B": np.random.randn(n) * 0.03,
        "C": np.random.randn(n) * 0.02,
    })
    cov = rets.cov() * 252
    w = portfolio.risk_parity_weights(cov)
    assert len(w) == 3
    assert abs(w.sum() - 1.0) < 1e-6
    # 高波动 B 的权重应小于低波动 A
    assert w["A"] > w["B"]


def test_portfolio_run(price_df):
    p = portfolio.Portfolio(price_df, rebalance_freq="W-MON")
    res = p.run()
    assert isinstance(res, portfolio.PortfolioResult)
    assert not res.equity.empty
    assert res.equity.iloc[0] == pytest.approx(100000, rel=1e-3)


def test_long_short_backtest(price_df):
    sig = price_df.pct_change(5).shift(-5)  # 5 日后的收益作为信号
    res = portfolio.long_short_backtest(price_df, sig, top_k=3, rebalance_freq="W-MON")
    assert not res.equity.empty
    assert res.long_short is True


def test_rebalance_dates():
    idx = pd.bdate_range("2024-01-01", periods=30)
    d = portfolio.rebalance_dates(idx, "W-MON")
    assert len(d) > 0
    # 周一
    assert all(d.weekday == 0)
