"""仓位管理:凯利公式 / 波动率目标 / 固定风险法。

输出永远在 [0, max_pos] 之间,支持回测中按日调整仓位。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def kelly_fraction(win_rate: float, win_loss_ratio: float,
                   kelly_cap: float = 0.25) -> float:
    """凯利公式:f* = (p*b - q) / b, 其中 p=胜率, q=1-p, b=盈亏比。
    实际使用建议取 1/4 Kelly,默认 cap=0.25 防止极端值。

    参数:
        win_rate: 胜率 0~1
        win_loss_ratio: 平均盈利 / 平均亏损 (>=0)
        kelly_cap: 最终仓位上限
    """
    if win_loss_ratio <= 0 or not (0 <= win_rate <= 1):
        return 0.0
    p = win_rate
    q = 1 - p
    b = win_loss_ratio
    raw = (p * b - q) / b
    return max(0.0, min(raw, kelly_cap))


def volatility_target(returns: pd.Series, target_vol: float = 0.15,
                      lookback: int = 60, max_leverage: float = 2.0) -> pd.Series:
    """波动率目标仓位:weight = target_vol / realized_vol,夹到 [0, max_leverage]。

    returns: 日度简单收益序列
    target_vol: 年化目标波动率(0.15 = 15%)
    lookback: 计算已实现波动率的窗口
    """
    if returns.empty:
        return pd.Series(dtype=float)
    vol = returns.rolling(lookback, min_periods=lookback // 2).std() * math.sqrt(252)
    weight = (target_vol / vol.replace(0, np.nan)).clip(upper=max_leverage).fillna(0)
    return weight.shift(1).fillna(0)  # 滞后一日避免未来函数


def fixed_fractional(capital: float, risk_per_trade: float,
                     stop_loss_pct: float) -> float:
    """固定风险法:仓位 = 资金 × 单笔风险比例 / 止损幅度。
    stop_loss_pct: 0.05 表示 5% 止损。
    """
    if stop_loss_pct <= 0:
        return 0.0
    return max(0.0, capital * risk_per_trade / stop_loss_pct)


def max_drawdown_target(equity: pd.Series, max_dd: float = 0.10,
                        recovery_factor: float = 0.5) -> pd.Series:
    """回撤风控仓位:接近最大回撤上限时减仓。
    简单实现:当前回撤 / max_dd;若超过阈值则按 recovery_factor 削减。
    """
    if equity.empty:
        return pd.Series(dtype=float)
    peak = equity.cummax()
    dd = (equity / peak - 1).abs()
    ratio = (dd / max_dd).clip(0, 1)
    return (1 - ratio * recovery_factor).clip(0, 1)


@dataclass
class PositionSizer:
    """统一仓位接口。method ∈ {'all-in','kelly','vol-target','fixed-frac','dd-guard'}"""
    method: str = "all-in"
    target_vol: float = 0.15
    lookback: int = 60
    risk_per_trade: float = 0.02
    stop_loss: float = 0.05
    max_dd: float = 0.10
    recovery_factor: float = 0.5
    win_rate: float = 0.55
    win_loss_ratio: float = 1.5
    max_leverage: float = 2.0

    def size(self, ctx: dict) -> float:
        """ctx: 必含 'capital',按 method 可能需要 'returns','equity','stop_loss'。"""
        capital = ctx.get("capital", 0)
        if self.method == "all-in":
            return min(1.0, capital / max(capital, 1e-9))
        if self.method == "kelly":
            return kelly_fraction(self.win_rate, self.win_loss_ratio)
        if self.method == "vol-target":
            rets = ctx.get("returns")
            if rets is None or rets.empty:
                return 0.0
            weight = volatility_target(rets, self.target_vol, self.lookback, self.max_leverage)
            return float(weight.iloc[-1]) if not weight.empty else 0.0
        if self.method == "fixed-frac":
            sl = ctx.get("stop_loss", self.stop_loss)
            return fixed_fractional(capital, self.risk_per_trade, sl) / max(capital, 1e-9)
        if self.method == "dd-guard":
            eq = ctx.get("equity")
            if eq is None or eq.empty:
                return 1.0
            w = max_drawdown_target(eq, self.max_dd, self.recovery_factor)
            return float(w.iloc[-1])
        raise ValueError(f"未知 method: {self.method}")
