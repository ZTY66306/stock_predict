"""仓位管理离线测试。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from baostock_tool import position


def test_kelly_basic():
    # 胜率 60%,盈亏比 2: f* = (0.6*2 - 0.4)/2 = 0.4,默认 cap=0.25 → 0.25
    f = position.kelly_fraction(0.6, 2.0)
    assert abs(f - 0.25) < 1e-6
    # 不 cap 时应得到 0.4
    f_raw = position.kelly_fraction(0.6, 2.0, kelly_cap=1.0)
    assert abs(f_raw - 0.4) < 1e-6


def test_kelly_cap():
    # 极端胜率 → cap
    f = position.kelly_fraction(0.99, 10.0, kelly_cap=0.25)
    assert f == 0.25


def test_kelly_negative():
    # 期望为负时返回 0
    f = position.kelly_fraction(0.2, 1.0)
    assert f == 0.0


def test_volatility_target_low_vol():
    np.random.seed(42)
    # 低波动率 -> 高权重
    low_vol = pd.Series(np.random.randn(200) * 0.005)
    weights = position.volatility_target(low_vol, target_vol=0.15, lookback=60)
    assert weights.max() > 1.0


def test_volatility_target_high_vol():
    np.random.seed(1)
    # 高波动率 -> 低权重
    high_vol = pd.Series(np.random.randn(200) * 0.05)
    weights = position.volatility_target(high_vol, target_vol=0.15, lookback=60)
    assert weights.max() < 1.0


def test_fixed_fractional():
    cap = 100000
    # 单笔风险 2%,止损 5% -> 仓位 = 100000*0.02/0.05 = 40000
    pos = position.fixed_fractional(cap, 0.02, 0.05)
    assert pos == pytest.approx(40000, rel=1e-6)


def test_position_sizer_all_in():
    sizer = position.PositionSizer(method="all-in")
    assert sizer.size({"capital": 100000}) == 1.0


def test_position_sizer_kelly():
    # 默认 cap=0.25,(0.55*1.5-0.45)/1.5 = 0.25 → 不 cap
    sizer = position.PositionSizer(method="kelly", win_rate=0.6, win_loss_ratio=2.0)
    # 0.4 被默认 cap 0.25 截断
    assert abs(sizer.size({}) - 0.25) < 1e-6
    sizer2 = position.PositionSizer(method="kelly", win_rate=0.6, win_loss_ratio=2.0,
                                     max_leverage=1.0)  # 占位避免影响
    # 改用更大 cap
    sizer2.kelly_fraction = None  # 让 sizer 改用 _kelly
    # 直接验证 raw 凯利 = 0.4
    f = position.kelly_fraction(0.6, 2.0, kelly_cap=1.0)
    assert abs(f - 0.4) < 1e-6


def test_max_drawdown_target():
    eq = pd.Series([100, 110, 105, 100, 90, 95, 110])
    w = position.max_drawdown_target(eq, max_dd=0.20, recovery_factor=0.5)
    # 最大回撤深度 20/110 ≈ 0.18,比 0.20 小
    # 后期恢复到 0.0~0.5 之间
    assert (w >= 0).all() and (w <= 1.0).all()
