"""单元测试:quant 与 screener 的纯逻辑部分(不联网)。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from baostock_tool import quant, screener, predict


def test_factor_ic_basic():
    np.random.seed(0)
    dates = pd.bdate_range("2024-01-01", periods=20)
    codes = [f"s{i}" for i in range(30)]
    factor = pd.DataFrame(np.random.randn(20, 30), index=dates, columns=codes)
    # 让 forward_ret 与 factor 完全正相关
    forward = factor * 0.01 + np.random.randn(20, 30) * 0.0001
    forward = pd.DataFrame(forward, index=dates, columns=codes)
    ic = quant.factor_ic(factor, forward, method="spearman")
    assert (ic["IC"] > 0.5).mean() > 0.5  # 大部分期数 IC 应为正


def test_ic_summary_keys():
    dates = pd.bdate_range("2024-01-01", periods=10)
    ic = pd.DataFrame({"IC": np.random.randn(10) * 0.05, "abs_IC": np.abs(np.random.randn(10) * 0.05),
                       "pvalue": np.random.rand(10)}, index=dates)
    summary = quant.ic_summary(ic)
    for k in ["IC_mean", "IC_std", "IC_IR", "abs_IC_mean", "期数"]:
        assert k in summary.index


def test_layered_backtest_returns_columns():
    np.random.seed(1)
    dates = pd.bdate_range("2024-01-01", periods=30)
    codes = [f"s{i}" for i in range(40)]
    factor = pd.DataFrame(np.random.randn(30, 40), index=dates, columns=codes)
    forward = pd.DataFrame(factor.values * 0.01 + np.random.randn(30, 40) * 0.001,
                           index=dates, columns=codes)
    layered = quant.layered_backtest(factor, forward, q=5)
    assert "期均收益" in layered.columns
    assert "累计收益" in layered.columns
    # 多空差应大于 0(因子与收益同向)
    if hasattr(layered, "attrs"):
        assert layered.attrs.get("多空差", 0) > 0


def test_build_features_shape():
    n = 200
    close = 10 + np.cumsum(np.random.randn(n) * 0.1)
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.05,
        "high": close + np.abs(np.random.randn(n) * 0.2),
        "low": close - np.abs(np.random.randn(n) * 0.2),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=pd.bdate_range("2023-01-01", periods=n))
    feat = predict.build_features(df)
    assert feat.shape[0] == n
    assert feat.shape[1] > 20
    # 特征不能全 NaN
    assert feat.dropna(how="all").shape[0] > 100


def test_pe_between_condition():
    df = pd.DataFrame({"peTTM": [10, 25, 50, 5]}, index=range(4))
    cond = screener.pe_between(0, 30)
    mask = cond(df)
    assert mask.tolist() == [True, True, False, True]
