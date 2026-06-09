"""走步优化 + 网格搜索离线测试。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from baostock_tool import optimizer, backtest, strategy


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


def test_grid_search_returns_dataframe(sample_df):
    grid = {"short": [3, 5], "long": [10, 20]}
    df = optimizer.grid_search(sample_df, "ma_cross", grid, metric="sharpe")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # 2x2
    assert "score" in df.columns


def test_grid_search_sorted(sample_df):
    grid = {"short": [3, 5, 8], "long": [10, 20, 30]}
    df = optimizer.grid_search(sample_df, "ma_cross", grid, metric="total_return")
    # 应按 score 降序
    assert (df["score"].diff().dropna() <= 0).all()


def test_walk_forward_runs(sample_df):
    grid = {"short": [3, 5], "long": [10, 20]}
    opt = optimizer.WalkForwardOptimizer("ma_cross", grid, n_splits=3, metric="total_return")
    res = opt.run(sample_df)
    assert isinstance(res, optimizer.WalkForwardResult)
    assert len(res.folds) >= 2
    # OOS 权益应非空
    assert res.oos_equity is not None and not res.oos_equity.empty


def test_walk_forward_params_per_fold(sample_df):
    grid = {"short": [3, 5, 8], "long": [10, 20, 30]}
    opt = optimizer.WalkForwardOptimizer("ma_cross", grid, n_splits=3, metric="calmar_ratio")
    res = opt.run(sample_df)
    assert res.best_params_per_fold is not None
    assert "short" in res.best_params_per_fold.columns
    assert "long" in res.best_params_per_fold.columns
