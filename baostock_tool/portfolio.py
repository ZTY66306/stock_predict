"""组合回测与权重构造:多空组合 / 风险平价 / 信号驱动再平衡。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

from baostock_tool import backtest


# ============ 权重构造 ============

def equal_weight(codes: list[str], date: pd.Timestamp) -> pd.Series:
    return pd.Series(1.0 / len(codes), index=codes)


def risk_parity_weights(cov: pd.DataFrame, max_iter: int = 100, tol: float = 1e-6) -> pd.Series:
    """风险平价:每只资产对组合风险的贡献相等。迭代法实现(基于对角协方差近似)。

    简化:从等权开始,反复调整 w_i ∝ 1/Σ_ii(代表波动率倒数加权),直到稳定。
    这种方法对非对角协方差矩阵较稳健,适合实际组合。
    """
    if cov.empty:
        return pd.Series(dtype=float)
    cov = cov.dropna(how="any").dropna(axis=1, how="any")
    if cov.empty:
        return pd.Series(dtype=float)
    if len(cov) == 1:
        return pd.Series([1.0], index=cov.columns)
    # 用对角线(即单资产波动率)做近似风险平价
    diag = pd.Series(np.diag(cov.values), index=cov.columns).clip(lower=1e-12)
    w = pd.Series(1.0 / np.sqrt(diag.values), index=cov.columns)
    w = w / w.sum()
    return w


def signal_weighted(signals: pd.Series, top_k: Optional[int] = None,
                    long_only: bool = True) -> pd.Series:
    """按 signal 大小加权(做多为正权重),可选 top_k 截断。"""
    if long_only:
        s = signals.clip(lower=0)
    else:
        s = signals.copy()
    if top_k is not None and len(s) > top_k:
        if long_only:
            keep = s.nlargest(top_k).index
            s = s.loc[keep]
        else:
            top = s.nlargest(top_k // 2).index
            bot = s.nsmallest(top_k - top_k // 2).index
            keep = top.union(bot)
            s = s.loc[keep]
    total = s.abs().sum()
    if total == 0:
        return pd.Series(0.0, index=signals.index)
    return s / total


# ============ 调仓日历 ============

def rebalance_dates(index: pd.DatetimeIndex, freq: str = "W-MON") -> pd.DatetimeIndex:
    """从交易日序列中提取指定频率的调仓日。"""
    if index.empty:
        return index
    # 用 pandas 的 resample 找锚点
    s = pd.Series(1, index=index)
    grouped = s.resample(freq).first()
    grouped = grouped.dropna()
    return grouped.index


# ============ 组合回测引擎 ============

@dataclass
class PortfolioResult:
    equity: pd.Series
    weights_history: pd.DataFrame          # 行=调仓日,列=code
    turnover: pd.Series                    # 每日换手率
    returns: pd.Series                     # 组合日收益
    cfg: backtest.BacktestConfig
    long_short: bool = False

    @property
    def total_return(self) -> float:
        if self.equity.empty:
            return 0.0
        return float(self.equity.iloc[-1] / self.cfg.initial_cash - 1)

    @property
    def annual_return(self) -> float:
        if self.equity.empty:
            return 0.0
        days = len(self.equity)
        years = days / self.cfg.annual_trading_days
        if years <= 0:
            return 0.0
        return (self.equity.iloc[-1] / self.cfg.initial_cash) ** (1 / years) - 1

    @property
    def sharpe(self) -> float:
        if self.returns.empty or self.returns.std() == 0:
            return 0.0
        return float(self.returns.mean() / self.returns.std() * math.sqrt(self.cfg.annual_trading_days))

    @property
    def max_drawdown(self) -> float:
        if self.equity.empty:
            return 0.0
        peak = self.equity.cummax()
        return float((self.equity / peak - 1).min())

    @property
    def avg_turnover(self) -> float:
        if self.turnover.empty:
            return 0.0
        return float(self.turnover[self.turnover > 0].mean()) if (self.turnover > 0).any() else 0.0

    def summary(self) -> pd.Series:
        return pd.Series({
            "总收益率": f"{self.total_return * 100:.2f}%",
            "年化收益": f"{self.annual_return * 100:.2f}%",
            "最大回撤": f"{self.max_drawdown * 100:.2f}%",
            "夏普比率": f"{self.sharpe:.2f}",
            "平均换手": f"{self.avg_turnover * 100:.2f}%",
            "调仓次数": int((self.turnover > 0).sum()),
        })


class Portfolio:
    """多标的组合回测:支持自定义权重函数 / 调仓频率。"""

    def __init__(self, prices: pd.DataFrame, cfg: Optional[backtest.BacktestConfig] = None,
                 rebalance_freq: str = "W-MON", long_only: bool = True,
                 weight_fn: Optional[Callable[[pd.Timestamp, pd.DataFrame], pd.Series]] = None):
        self.prices = prices.sort_index().ffill()
        self.cfg = cfg or backtest.BacktestConfig()
        self.rebalance_freq = rebalance_freq
        self.long_only = long_only
        self.weight_fn = weight_fn or (
            lambda dt, pr: equal_weight(list(pr.columns), dt)
        )

    def run(self) -> PortfolioResult:
        rebal_dates = set(rebalance_dates(self.prices.index, self.rebalance_freq))
        n = len(self.prices)
        weights = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        turnover = pd.Series(0.0, index=self.prices.index)
        rets = self.prices.pct_change().fillna(0)
        for i, dt in enumerate(self.prices.index):
            if dt in rebal_dates or i == 0:
                w = self.weight_fn(dt, self.prices)
                w = w.reindex(self.prices.columns).fillna(0)
                if self.long_only:
                    w = w.clip(lower=0)
                total = w.sum()
                if total > 0:
                    w = w / total
                # 计算换手
                turnover.iloc[i] = (w - weights.iloc[i - 1] if i > 0 else w).abs().sum()
                weights.iloc[i] = w
            else:
                # 沿用上一期权重
                if i > 0:
                    weights.iloc[i] = weights.iloc[i - 1]
                    turnover.iloc[i] = 0.0
        # 收益:用前一期权重 × 当期收益
        port_ret = (weights.shift(1).fillna(0) * rets).sum(axis=1)
        # 扣手续费
        port_ret = port_ret - turnover * self.cfg.commission
        equity = (1 + port_ret).cumprod() * self.cfg.initial_cash
        return PortfolioResult(
            equity=equity, weights_history=weights, turnover=turnover,
            returns=port_ret, cfg=self.cfg, long_short=not self.long_only,
        )


def long_short_backtest(prices: pd.DataFrame, signal: pd.DataFrame,
                        top_k: int = 10, rebalance_freq: str = "W-MON",
                        cfg: Optional[backtest.BacktestConfig] = None) -> PortfolioResult:
    """多空组合回测:每个调仓日,signal 最高的 top_k/2 做多,最低的 top_k/2 做空,等权。"""
    if signal.empty or signal.shape != prices.shape:
        raise ValueError("signal 与 prices 必须同 shape")

    def _w(dt: pd.Timestamp, _pr) -> pd.Series:
        if dt not in signal.index:
            return pd.Series(0.0, index=signal.columns)
        row = signal.loc[dt].dropna()
        if row.empty:
            return pd.Series(0.0, index=signal.columns)
        n = min(top_k, len(row))
        long = row.nlargest(n // 2).index
        short = row.nsmallest(n - n // 2).index
        w = pd.Series(0.0, index=signal.columns)
        if len(long) > 0:
            w.loc[long] = 1.0 / n
        if len(short) > 0:
            w.loc[short] = -1.0 / n
        return w

    p = Portfolio(prices, cfg=cfg, rebalance_freq=rebalance_freq, long_only=False, weight_fn=_w)
    result = p.run()
    result.long_short = True
    return result
