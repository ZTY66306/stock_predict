"""K 线形态识别离线单元测试。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from baostock_tool import patterns as ptn


def _make_df(rows):
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    return pd.DataFrame(rows, index=idx)


def test_doji_detect():
    df = _make_df([
        # 开收几乎相同
        {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.001},
        # 正常阳线
        {"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.4},
    ])
    mask = ptn.doji(df, tol=0.01)
    assert mask.iloc[0]  # 十字星
    assert not mask.iloc[1]


def test_hammer_basic():
    # 锤头:小实体 + 长下影 + 极短上影
    # body=0.1, ls=0.9, us=0.0(实际 0.001),rng=1.0
    # 0.1 <= 0.3 ✓, 0.9 >= 0.2 ✓, 0.001 <= 0.05 ✓
    df = _make_df([
        {"open": 10.0, "high": 10.001, "low": 9.0, "close": 10.1},
    ])
    mask = ptn.hammer(df, body_ratio=0.3, shadow_ratio=2.0)
    assert mask.iloc[0]


def test_engulfing_bullish():
    # 前阴后阳,阳线实体包住阴线
    df = _make_df([
        {"open": 10.0, "high": 10.1, "low": 9.5, "close": 9.6},  # 阴
        {"open": 9.4, "high": 10.2, "low": 9.3, "close": 10.1},  # 阳
    ])
    mask = ptn.engulfing_bullish(df)
    assert mask.iloc[1]
    assert not mask.iloc[0]


def test_three_white_soldiers():
    rows = [
        {"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.4},
        {"open": 10.2, "high": 10.6, "low": 10.1, "close": 10.5},
        {"open": 10.4, "high": 10.9, "low": 10.3, "close": 10.8},
    ]
    df = _make_df(rows)
    mask = ptn.three_white_soldiers(df)
    assert mask.iloc[2]


def test_pattern_score_bullish():
    rows = [
        # 阴
        {"open": 10.0, "high": 10.1, "low": 9.5, "close": 9.6},
        # 阳,完全吞没前阴
        {"open": 9.4, "high": 10.2, "low": 9.3, "close": 10.1},
        # 穿头破脚阳线
        {"open": 10.0, "high": 10.5, "low": 9.95, "close": 10.45},
    ]
    df = _make_df(rows)
    score = ptn.pattern_score(df, bullish=True)
    # 第 1 行:engulfing_bullish → ≥1
    assert score.iloc[1] >= 1
    # 总分应 > 0
    assert (score > 0).sum() >= 1


def test_list_patterns_complete():
    names = ptn.list_patterns()
    assert "doji" in names
    assert "morning_star" in names
    assert "engulfing_bullish" in names
    assert "three_black_crows" in names
