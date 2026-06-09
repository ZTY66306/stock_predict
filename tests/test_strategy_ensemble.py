"""EnsembleStrategy 离线单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baostock_tool import strategy as st


def _mock_kline(n=200, base=10.0, seed=0):
    np.random.seed(seed)
    close = base + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({
        "open": close,
        "high": close + np.abs(np.random.randn(n)) * 0.2,
        "low":  close - np.abs(np.random.randn(n)) * 0.2,
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=pd.bdate_range("2024-01-01", periods=n))


# ============ EnsembleStrategy ============

def test_ensemble_weighted_voting():
    df = _mock_kline(seed=0)
    ens = st.EnsembleStrategy(["ma_cross", "macd", "kdj"],
                               voting="weighted", entry_threshold=0.3)
    sig = ens.run(df)
    assert "score" in sig.columns
    assert "position" in sig.columns
    assert "signal" in sig.columns
    assert sig["position"].isin([0, 1]).all()


def test_ensemble_majority_voting():
    df = _mock_kline(seed=1)
    ens = st.EnsembleStrategy(["ma_cross", "macd", "kdj", "rsi_oversold"],
                               voting="majority", majority_min=2)
    sig = ens.run(df)
    assert sig["position"].isin([-1, 0, 1]).all()
    # majority 比 weighted 更保守,position=1 的天数应更少
    ens_w = st.EnsembleStrategy(["ma_cross", "macd", "kdj", "rsi_oversold"],
                                 voting="weighted", entry_threshold=0.3)
    sig_w = ens_w.run(df)
    assert sig["position"].sum() <= sig_w["position"].sum()


def test_ensemble_veto_voting():
    df = _mock_kline(seed=2)
    ens = st.EnsembleStrategy(["ma_cross", "macd", "kdj"],
                               voting="veto", majority_min=2)
    sig = ens.run(df)
    # veto 必然 ≤ majority
    ens_m = st.EnsembleStrategy(["ma_cross", "macd", "kdj"],
                                 voting="majority", majority_min=2)
    sig_m = ens_m.run(df)
    assert sig["position"].abs().sum() <= sig_m["position"].abs().sum() + 1


def test_ensemble_custom_weights():
    df = _mock_kline(seed=3)
    ens = st.EnsembleStrategy(["ma_cross", "macd", "kdj"],
                               weights=[0.5, 0.3, 0.2],
                               voting="weighted", entry_threshold=0.3)
    # 权重被归一化,sum 应 = 1
    assert abs(sum(ens.weights) - 1.0) < 1e-6
    sig = ens.run(df)
    assert not sig.empty


def test_ensemble_with_strategy_objects():
    """支持直接传 Strategy 对象,不只名字。"""
    df = _mock_kline(seed=4)
    s1 = st.get_strategy("ma_cross")
    s2 = st.get_strategy("macd")
    ens = st.EnsembleStrategy([s1, s2], voting="weighted")
    sig = ens.run(df, params_list=[{"short": 5, "long": 20}, None])
    assert "position" in sig.columns


def test_ensemble_with_per_strategy_params():
    """每套策略可以传不同的参数。"""
    df = _mock_kline(seed=5)
    ens = st.EnsembleStrategy(["ma_cross", "macd", "kdj"])
    sig = ens.run(df, params_list=[
        {"short": 3, "long": 10},
        None,
        {"n": 9, "k_period": 3, "d_period": 3},
    ])
    assert "position" in sig.columns


def test_ensemble_empty_strategies_raises():
    with pytest.raises(ValueError):
        st.EnsembleStrategy([])


def test_ensemble_weight_length_mismatch_raises():
    with pytest.raises(ValueError):
        st.EnsembleStrategy(["ma_cross", "macd"], weights=[0.5, 0.3, 0.2])


def test_ensemble_signal_changes_match_position_diff():
    """signal 字段应与 position.diff 一致(只在 0→1 / 1→0 时分别触发 +1/-1)。"""
    df = _mock_kline(seed=6)
    ens = st.EnsembleStrategy(["ma_cross", "macd", "kdj", "boll"],
                               voting="majority", majority_min=2)
    sig = ens.run(df)
    pos_diff = sig["position"].diff()
    # position 从 0 → 1 时 signal 应 = 1
    on_enter = (pos_diff == 1)
    on_exit = (pos_diff == -1)
    assert (sig["signal"][on_enter] == 1).all()
    assert (sig["signal"][on_exit] == -1).all()
