"""RollingRobustness 离线单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baostock_tool import optimizer as opt


def _mock_kline(n=600, base=10.0, seed=0):
    np.random.seed(seed)
    close = base + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({
        "open": close, "high": close + 0.2, "low": close - 0.2,
        "close": close, "volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=pd.bdate_range("2022-01-01", periods=n))


# ============ RollingRobustness ============

def test_rolling_robustness_basic():
    df = _mock_kline(n=600, seed=0)
    rr = opt.RollingRobustness("ma_cross", params={"short": 5, "long": 20},
                                window=252, step=63)
    result = rr.run(df)
    assert len(result.folds) >= 5
    assert result.strategy_name == "ma_cross"
    assert result.params == {"short": 5, "long": 20}


def test_rolling_robustness_summary():
    df = _mock_kline(n=600, seed=1)
    rr = opt.RollingRobustness("ma_cross", params={"short": 5, "long": 20},
                                window=252, step=63)
    result = rr.run(df)
    s = result.summary()
    assert "中位夏普" in s.index
    assert "fold 数" in s.index
    assert "盈利 fold 数" in s.index
    assert "稳健性评分" in s.index or "中位最大回撤" in s.index


def test_rolling_robustness_score_components():
    """robustness_score 应返回 0~1 之间的 overall,以及三个分量。"""
    df = _mock_kline(n=800, seed=2)
    rr = opt.RollingRobustness("ma_cross", params={"short": 5, "long": 20},
                                window=252, step=63)
    result = rr.run(df)
    score = result.robustness_score()
    assert 0.0 <= score["overall"] <= 1.0
    assert 0.0 <= score["profit_rate"] <= 1.0
    assert 0.0 <= score["stability"] <= 1.0
    assert 0.0 <= score["dd_score"] <= 1.0


def test_rolling_robustness_folds_df():
    df = _mock_kline(n=600, seed=3)
    rr = opt.RollingRobustness("ma_cross", params={"short": 5, "long": 20},
                                window=252, step=63)
    result = rr.run(df)
    fdf = result.folds_df()
    assert not fdf.empty
    assert "sharpe" in fdf.columns
    assert "total_return" in fdf.columns
    assert "max_drawdown" in fdf.columns


def test_rolling_robustness_insufficient_data_raises():
    df = _mock_kline(n=100, seed=4)
    rr = opt.RollingRobustness("ma_cross", params={"short": 5, "long": 20},
                                window=252, step=63)
    with pytest.raises(ValueError):
        rr.run(df)


def test_rolling_robustness_step_larger_than_window_gives_one_fold():
    """step 极大时仍会生成 1 个 fold(rang(0, X, big) 包含 0)。"""
    df = _mock_kline(n=400, seed=5)
    rr = opt.RollingRobustness("ma_cross", params={"short": 5, "long": 20},
                                window=252, step=1000)
    result = rr.run(df)
    # 只跑了 1 个 fold,不报错
    assert len(result.folds) == 1


def test_rolling_robustness_empty_result_summary():
    """空 folds 时 summary 不报错。"""
    result = opt.RollingRobustnessResult()
    s = result.summary()
    assert s.empty
    assert result.robustness_score() == {}


def test_rolling_robustness_compare_params():
    """同一组数据、不同参数应给出不同的稳健性评分。"""
    df = _mock_kline(n=600, seed=6)
    r1 = opt.RollingRobustness("ma_cross", params={"short": 5, "long": 20},
                                window=252, step=63).run(df)
    r2 = opt.RollingRobustness("ma_cross", params={"short": 3, "long": 60},
                                window=252, step=63).run(df)
    # 不同参数,至少 robustness_score 之一分量应不同(可能 overall 相同)
    s1 = r1.robustness_score()
    s2 = r2.robustness_score()
    # 至少 profit_rate 或 dd_score 中之一不同
    differs = (s1["profit_rate"] != s2["profit_rate"] or
               s1["dd_score"] != s2["dd_score"] or
               s1["stability"] != s2["stability"])
    assert differs
