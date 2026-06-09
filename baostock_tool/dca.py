"""智能定投 (Dollar-Cost Averaging) 回测。

提供:
    - dca_backtest(...)    跑一次定投回测(pure / smart / dip_buy / lump_sum)
    - compare_strategies(...)  一次性对比 4 种策略(pure / smart / dip / lump)

典型用法:
    from baostock_tool import dca
    r = dca.dca_backtest("sh.510300", "2018-01-01", "2024-12-31",
                         amount_per_period=2000, frequency="monthly",
                         strategy="smart")
    print(r.summary())
    cmp = dca.compare_strategies("sh.510300", "2018-01-01", "2024-12-31",
                                  amount_per_period=2000)
    print(cmp)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import numpy as np
import pandas as pd

from . import data, indicators as ind


Frequency = Literal["weekly", "biweekly", "monthly"]
DCAStrategy = Literal["pure", "smart", "dip_buy", "lump_sum"]


# ============ 调仓日历 ============

def _rebalance_dates(index: pd.DatetimeIndex,
                      freq: Frequency) -> list[pd.Timestamp]:
    """从 K 线索引中提取定投调仓日。"""
    if freq == "weekly":
        rule = "W-FRI"
    elif freq == "biweekly":
        rule = "2W-FRI"
    else:
        rule = "ME"
    grouped = pd.Series(1, index=index).resample(rule).first()
    grouped = grouped.dropna()
    # 只保留实际交易日内的
    return [d for d in grouped.index if d in index]


# ============ 主回测 ============

@dataclass
class DCATrade:
    """单笔定投买入/卖出。"""
    date: pd.Timestamp
    side: str                # 'buy' / 'sell'
    shares: float
    price: float
    amount: float            # 投入金额(buy 为正,sell 为回收)
    cum_shares: float
    cum_invested: float
    reason: str = "schedule"  # 'schedule' / 'dip_buy' / 'overbought' / 'lump_sum'


@dataclass
class DCAResult:
    code: str
    name: str
    start: str
    end: str
    strategy: str
    frequency: str
    amount_per_period: float
    total_invested: float
    total_shares: float
    avg_cost: float
    final_value: float
    total_return: float
    annualized_return: float
    cost_basis_history: pd.Series
    shares_history: pd.Series
    value_history: pd.Series
    trades: list[DCATrade] = field(default_factory=list)
    cfg: dict = field(default_factory=dict)

    def summary(self) -> pd.Series:
        cfg = self.cfg
        return pd.Series({
            "代码": self.code,
            "名称": self.name,
            "回测区间": f"{cfg['start']} ~ {self.end}",
            "策略": self.strategy,
            "频率": self.frequency,
            "每期投入": f"{self.amount_per_period:,.2f}",
            "实际买入次数": sum(1 for t in self.trades if t.side == "buy"),
            "实际卖出次数": sum(1 for t in self.trades if t.side == "sell"),
            "累计投入": f"{self.total_invested:,.2f}",
            "累计股数": f"{self.total_shares:.0f}",
            "平均成本": f"{self.avg_cost:.4f}",
            "期末市值": f"{self.final_value:,.2f}",
            "总收益率": f"{self.total_return*100:.2f}%",
            "年化收益": f"{self.annualized_return*100:.2f}%",
        })


def dca_backtest(code_or_name: str,
                 start: str,
                 end: str,
                 amount_per_period: float = 2000.0,
                 frequency: Frequency = "monthly",
                 strategy: DCAStrategy = "pure",
                 ma_window: int = 120,
                 dip_threshold: float = -0.05,
                 overbought_threshold: float = 0.10,
                 dip_multiplier: float = 2.0,
                 overbought_multiplier: float = 0.5,
                 max_multiplier: float = 3.0,
                 commission: float = 0.0003,
                 min_commission: float = 5.0,
                 lot_size: int = 100) -> DCAResult:
    """跑一次定投回测。

    strategy:
        pure      — 固定每期 amount_per_period 元
        smart     — 均线偏离加仓:< 0.95×MA → 1.5x; < 0.90×MA → 2x; > 1.10×MA → 0.5x
        dip_buy   — 跌幅加仓:当周/当月跌幅 < dip_threshold → dip_multiplier 倍;否则正常
        lump_sum  — 一次性投入(在首日)
    """
    from . import client
    client.ensure_login()
    # 自动解析名称
    from .grid_backtest import resolve_code
    code, name = resolve_code(code_or_name)
    df = data.get_kline(code, start, end, use_cache=True)
    if df.empty:
        raise ValueError(f"无 K 线数据: {code}")
    df = df.sort_index()
    end_str = df.index[-1].strftime("%Y-%m-%d")

    # 均线
    ma = ind.MA(df["close"], ma_window) if ma_window > 0 else pd.Series(np.nan, index=df.index)

    # 调仓日
    if strategy == "lump_sum":
        rebal_dates = [df.index[0]]
    else:
        rebal_dates = _rebalance_dates(df.index, frequency)

    # 状态
    cum_invested = 0.0
    cum_shares = 0.0
    trades: list[DCATrade] = []
    invested_list = []
    shares_list = []
    value_list = []

    for dt, row in df.iterrows():
        price = float(row["close"])
        if dt in rebal_dates:
            # 计算本期投入
            if strategy == "lump_sum" and len(trades) == 0:
                amount = amount_per_period
                reason = "lump_sum"
            else:
                amount = amount_per_period
                reason = "schedule"
                if strategy == "smart":
                    if not pd.isna(ma.loc[dt]):
                        ma_v = float(ma.loc[dt])
                        ratio = price / ma_v
                        if ratio < 0.90:
                            amount *= min(dip_multiplier * 1.5, max_multiplier)
                            reason = "smart_dip_strong"
                        elif ratio < 0.95:
                            amount *= dip_multiplier
                            reason = "smart_dip"
                        elif ratio > 1.0 + overbought_threshold:
                            amount *= overbought_multiplier
                            reason = "smart_overbought"
                elif strategy == "dip_buy":
                    # 看上一调仓日到今天的跌幅
                    if trades:
                        last_dt = trades[-1].date
                        last_price = float(df.loc[last_dt, "close"])
                        change = (price - last_price) / last_price
                        if change <= dip_threshold:
                            amount *= dip_multiplier
                            reason = "dip_buy"
            if amount <= 0:
                invested_list.append(cum_invested)
                shares_list.append(cum_shares)
                value_list.append(cum_shares * price)
                continue
            # 扣手续费
            shares_bought = amount / price
            fee = max(min_commission, amount * commission)
            if fee >= amount:
                # 投入太少不够手续费,跳过
                invested_list.append(cum_invested)
                shares_list.append(cum_shares)
                value_list.append(cum_shares * price)
                continue
            net_amount = amount - fee
            actual_shares = net_amount / price
            cum_invested += amount
            cum_shares += actual_shares
            trades.append(DCATrade(
                date=dt, side="buy", shares=actual_shares, price=price,
                amount=amount, cum_shares=cum_shares, cum_invested=cum_invested,
                reason=reason,
            ))

        invested_list.append(cum_invested)
        shares_list.append(cum_shares)
        value_list.append(cum_shares * price)

    # 收尾
    last_price = float(df["close"].iloc[-1])
    final_value = cum_shares * last_price
    total_return = (final_value - cum_invested) / cum_invested if cum_invested > 0 else 0.0
    days = len(df)
    years = days / 252 if days > 0 else 0
    if years > 0 and final_value > 0 and cum_invested > 0:
        # 几何年化:用累计投入 / 期末市值的比值
        annualized = (final_value / cum_invested) ** (1 / years) - 1
    else:
        annualized = 0.0
    avg_cost = cum_invested / cum_shares if cum_shares > 0 else 0.0

    return DCAResult(
        code=code, name=name, start=start, end=end_str,
        strategy=strategy, frequency=frequency,
        amount_per_period=amount_per_period,
        total_invested=cum_invested, total_shares=cum_shares,
        avg_cost=avg_cost, final_value=final_value,
        total_return=total_return, annualized_return=annualized,
        cost_basis_history=pd.Series(invested_list, index=df.index, name="invested"),
        shares_history=pd.Series(shares_list, index=df.index, name="shares"),
        value_history=pd.Series(value_list, index=df.index, name="value"),
        trades=trades,
        cfg=dict(start=start, end=end_str, frequency=frequency,
                 strategy=strategy, amount_per_period=amount_per_period,
                 ma_window=ma_window, dip_threshold=dip_threshold),
    )


def compare_strategies(code_or_name: str,
                        start: str, end: str,
                        amount_per_period: float = 2000.0,
                        frequency: Frequency = "monthly",
                        **kwargs) -> pd.DataFrame:
    """一次跑 4 种策略,横向对比。"""
    rows = []
    results = {}
    for s in ["lump_sum", "pure", "dip_buy", "smart"]:
        try:
            r = dca_backtest(
                code_or_name, start, end, amount_per_period=amount_per_period,
                frequency=frequency, strategy=s, **kwargs,
            )
            results[s] = r
            rows.append({
                "策略": s,
                "累计投入": r.total_invested,
                "期末市值": r.final_value,
                "总收益率": r.total_return,
                "年化收益": r.annualized_return,
                "平均成本": r.avg_cost,
                "买入次数": sum(1 for t in r.trades if t.side == "buy"),
            })
        except Exception as e:
            rows.append({"策略": s, "累计投入": np.nan, "期末市值": np.nan,
                          "总收益率": np.nan, "年化收益": np.nan,
                          "平均成本": np.nan, "买入次数": 0,
                          "error": str(e)})
    return pd.DataFrame(rows)


# ============ 可视化 ============

def plot(result: DCAResult, save_path: Optional[str] = None):
    """累计投入 vs 市值 vs 价格 三联图。"""
    import os
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(result.value_history.index, result.value_history.values,
             color="navy", linewidth=1.2, label="市值")
    ax1.plot(result.cost_basis_history.index, result.cost_basis_history.values,
             color="grey", linestyle="--", linewidth=0.8, label="累计投入")
    ax1.fill_between(result.cost_basis_history.index,
                     result.cost_basis_history.values,
                     result.value_history.values,
                     where=result.value_history.values >= result.cost_basis_history.values,
                     color="red", alpha=0.15, label="浮盈")
    ax1.fill_between(result.cost_basis_history.index,
                     result.cost_basis_history.values,
                     result.value_history.values,
                     where=result.value_history.values < result.cost_basis_history.values,
                     color="green", alpha=0.15, label="浮亏")
    ax1.set_title(
        f"DCA  {result.code} {result.name}  strategy={result.strategy}  "
        f"freq={result.frequency}  return={result.total_return*100:.2f}%"
    )
    ax1.set_ylabel("金额")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # 平均成本 vs 收盘价
    from . import data as _data
    df = _data.get_kline(result.code, result.start, result.end, use_cache=True)
    avg_cost_series = result.cost_basis_history / result.shares_history.replace(0, np.nan)
    ax2.plot(df.index, df["close"].values, color="black", linewidth=0.7,
             label="Close")
    ax2.plot(avg_cost_series.index, avg_cost_series.values,
             color="orange", linewidth=1.0, label="Avg Cost")
    ax2.set_ylabel("价格")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, (ax1, ax2)


def plot_compare(results: dict, save_path: Optional[str] = None):
    """多策略市值曲线对比。"""
    import os
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, r in results.items():
        if not isinstance(r, DCAResult):
            continue
        ax.plot(r.value_history.index, r.value_history.values,
                linewidth=1.2, label=f"{name} ({r.total_return*100:.1f}%)")
    ax.set_title("DCA 策略对比(期末市值)")
    ax.set_ylabel("市值")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, ax


def write_text_report(result: DCAResult, path: str):
    """写文本报告。"""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 60)
    lines.append(f"DCA 报告  {result.code} {result.name}")
    lines.append("=" * 60)
    for k, v in result.summary().items():
        lines.append(f"  {k:>14s}: {v}")
    if result.trades:
        lines.append("")
        lines.append(f"交易明细(共 {len(result.trades)} 笔,展示前 20 / 后 5):")
        for t in result.trades[:20]:
            lines.append(
                f"  {t.date.date()}  {t.side:>4s}  "
                f"amount={t.amount:>10.2f}  price={t.price:.4f}  "
                f"reason={t.reason}"
            )
        if len(result.trades) > 25:
            lines.append("  ...")
            for t in result.trades[-5:]:
                lines.append(
                    f"  {t.date.date()}  {t.side:>4s}  "
                    f"amount={t.amount:>10.2f}  price={t.price:.4f}  "
                    f"reason={t.reason}"
                )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
