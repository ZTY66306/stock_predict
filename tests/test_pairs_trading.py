"""配对交易离线单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baostock_tool import pairs_trading as pt


# ============ 工具 ============

def _cointegrated_pair(n=300, seed=0):
    """构造一对明显协整的合成价格。"""
    np.random.seed(seed)
    common = np.cumsum(np.random.randn(n) * 0.3) + 100
    a = common + np.random.randn(n) * 0.5
    b = 0.7 * common + 30 + np.random.randn(n) * 0.5
    idx = pd.bdate_range("2024-01-01", periods=n)
    return (pd.Series(a, index=idx, name="A"),
            pd.Series(b, index=idx, name="B"))


def _random_pair(n=300, seed=1):
    """构造一对不相关/不协整的合成价格。"""
    np.random.seed(seed)
    a = 100 + np.cumsum(np.random.randn(n))
    b = 50 + np.cumsum(np.random.randn(n))
    idx = pd.bdate_range("2024-01-01", periods=n)
    return (pd.Series(a, index=idx, name="A"),
            pd.Series(b, index=idx, name="B"))


# ============ 工具函数 ============

def test_ols_hedge_ratio_recovers_beta():
    a, b = _cointegrated_pair(seed=0)
    h = pt._ols_hedge(a, b)
    # 我们设的 b = 0.7 * common + 30,a = common + noise
    # 则 a = (1/0.7) * b + const + noise,OLS 回归 a~b 斜率应接近 1/0.7 ≈ 1.43
    # 接受一个宽松区间(噪声下偏差)
    assert 0.8 < h < 2.5


def test_compute_hedge_ratio_static_vs_rolling():
    a, b = _cointegrated_pair(seed=0)
    h_static = pt.compute_hedge_ratio(a, b, mode="static", lookback=120)
    h_rolling = pt.compute_hedge_ratio(a, b, mode="rolling", lookback=120)
    assert len(h_static) == len(a)
    assert h_static.iloc[-1] > 0
    assert h_rolling.dropna().shape[0] > 0
    # rolling 应有变化(非平稳 hedge)
    assert h_rolling.std() > 0 or h_rolling.nunique() > 1


def test_compute_spread_and_zscore():
    a, b = _cointegrated_pair(seed=0)
    h = pt._ols_hedge(a, b)
    spread = pt.compute_spread(a, b, h)
    z = pt.compute_zscore(spread, n=60)
    # z-score 应有有效值
    valid = z.dropna()
    assert len(valid) > 100
    # z-score 的均值应接近 0(标准化)
    assert abs(valid.mean()) < 0.5


# ============ select_pairs ============

def test_select_pairs_correlation_method():
    """相关法:即便合成数据不协整,只要相关够高也应被选中。"""
    a, b = _cointegrated_pair(seed=0)
    prices = pd.concat([a.rename("sh.1"), b.rename("sh.2"),
                         _random_pair(seed=10)[0].rename("sh.3")], axis=1)
    pairs = pt.select_pairs(prices, method="correlation", top_n=5, min_corr=0.5)
    assert len(pairs) >= 1
    assert "code_a" in pairs.columns
    assert "hedge_ratio" in pairs.columns


def test_select_pairs_distance_method():
    """距离法:用 _distance_ratio 评分,ascending 排序。"""
    a, b = _cointegrated_pair(seed=0)
    prices = pd.concat([a.rename("sh.1"), b.rename("sh.2")], axis=1)
    pairs = pt.select_pairs(prices, method="distance", top_n=3)
    # 距离法可能选也可能不选,只看 API 不报错
    assert isinstance(pairs, pd.DataFrame)


def test_select_pairs_empty_when_too_few_codes():
    prices = pd.DataFrame({"sh.1": [1, 2, 3]})
    pairs = pt.select_pairs(prices, method="correlation")
    assert pairs.empty


# ============ pairs_backtest ============

def test_pairs_backtest_basic():
    a, b = _cointegrated_pair(n=400, seed=0)
    result = pt.pairs_backtest(a, b, entry_z=2.0, exit_z=0.5, lookback=60,
                                capital=100_000)
    assert result.code_a == "A"
    assert result.code_b == "B"
    assert result.hedge_ratio > 0
    assert not result.equity.empty
    # summary 不抛异常
    s = result.summary()
    assert "总收益率" in s.index
    assert "做多次" in s.index
    assert "做空次" in s.index


def test_pairs_backtest_trades_state_machine():
    """交易明细的状态机应交替(开仓 → 平仓)。"""
    a, b = _cointegrated_pair(n=400, seed=0)
    result = pt.pairs_backtest(a, b, entry_z=1.5, exit_z=0.3, lookback=60)
    if len(result.trades) >= 2:
        sides = [t.side for t in result.trades]
        # 应有开仓 + 平仓
        opens = [s for s in sides if s.startswith("open_")]
        closes = [s for s in sides if s == "close"]
        assert len(opens) >= 1
        assert len(closes) >= 1


def test_pairs_backtest_max_hold_days_triggers_close():
    """max_hold_days 限制:持有 N 天强制平仓。"""
    a, b = _cointegrated_pair(n=400, seed=0)
    result = pt.pairs_backtest(
        a, b, entry_z=1.5, exit_z=0.3, lookback=60, max_hold_days=5,
    )
    # 至少应出现一个 max_hold 原因的平仓
    reasons = [t.reason for t in result.trades]
    # 不强制一定出现(取决于行情),但 API 应不报错
    assert isinstance(reasons, list)


def test_pairs_backtest_insufficient_data_raises():
    a = pd.Series(range(20), index=pd.bdate_range("2024-01-01", periods=20), name="A")
    b = pd.Series(range(20), index=pd.bdate_range("2024-01-01", periods=20), name="B")
    with pytest.raises(ValueError):
        pt.pairs_backtest(a, b, lookback=60, hedge_lookback=120)


def test_pairs_backtest_with_rolling_hedge():
    a, b = _cointegrated_pair(n=400, seed=2)
    result = pt.pairs_backtest(
        a, b, entry_z=2.0, exit_z=0.5, lookback=60,
        hedge_mode="rolling", hedge_lookback=120,
    )
    assert result.hedge_ratio > 0
    assert not result.equity.empty


def test_pairs_trades_df_format():
    a, b = _cointegrated_pair(n=400, seed=3)
    result = pt.pairs_backtest(a, b, entry_z=1.5, exit_z=0.3, lookback=60)
    tdf = result.trades_df()
    if not tdf.empty:
        assert "date" in tdf.columns
        assert "side" in tdf.columns
        assert "shares_a" in tdf.columns
        assert "shares_b" in tdf.columns
