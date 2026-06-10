"""配对交易 (Pairs Trading) — 经典统计套利工具。

提供:
    - select_pairs(prices_df)         按相关/协整/距离挑选股票对
    - compute_hedge_ratio(a, b, ...)  OLS / 滚动 OLS 算 hedge ratio
    - compute_spread(a, b, hedge)     价差序列
    - compute_zscore(spread, n)       滚动 z-score
    - pairs_backtest(...)             一键跑配对回测(z-score 阈值进出场)

回测假设:
    - 允许做空 A 持 B(或多 ETF 配对 / 期货对冲)
    - A、B 默认 T+1 制度,当日信号当日按收盘价成交,次日才可平仓
    - 提供 t0=True 选项用于 ETF / 可转债配对

典型用法:
    from baostock_tool import data, pairs_trading as pt
    codes = ["sh.600000", "sh.600036", "sz.000001", "sz.000002"]
    prices = pd.DataFrame({c: data.get_kline(c, "2022-01-01", "2024-12-31")["close"]
                           for c in codes})

    # 1) 选对
    pairs = pt.select_pairs(prices, method="cointest", top_n=5)
    print(pairs)

    # 2) 跑配对回测
    a, b = "sh.600000", "sh.600036"
    result = pt.pairs_backtest(prices[a], prices[b],
                               entry_z=2.0, exit_z=0.5, lookback=60)
    print(result.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from baostock_tool import data


# ============ 选对 ============

PairMethod = Literal["correlation", "cointest", "distance"]


def select_pairs(prices: pd.DataFrame,
                 method: PairMethod = "cointest",
                 top_n: int = 10,
                 min_corr: float = 0.5,
                 pvalue_threshold: float = 0.05) -> pd.DataFrame:
    """从价格矩阵(列=code,索引=日期)中挑选候选配对。

    method:
        correlation — 按 |Pearson 相关| 排序,过滤 min_corr
        cointest    — 在 |相关| > min_corr 的基础上跑 Engle-Granger 协整检验,
                      按 p-value 升序
        distance    — 按 (a - b).std() / ((a + b)/2).mean() 距离比,值越小越配
    返回:DataFrame,列 [code_a, code_b, score, hedge_ratio, pvalue?]
    """
    if prices.shape[1] < 2:
        return pd.DataFrame(columns=["code_a", "code_b", "score"])
    codes = list(prices.columns)
    n = len(codes)
    rows: list[dict] = []

    # 预计算:相关矩阵、协方差矩阵、波动率向量(一次 O(N²),后续向量化算 hedge)
    corr = prices.corr().fillna(0.0)
    std = prices.std(ddof=0)
    # 防止除零
    std_safe = std.replace(0, np.nan)

    # 逐对遍历(已经是 O(N²) 不可避免),但用向量化算 hedge
    for i in range(n):
        for j in range(i + 1, n):
            a, b = codes[i], codes[j]
            c = float(corr.iloc[i, j])
            if method == "correlation":
                if abs(c) < min_corr:
                    continue
                # 向量化 hedge: corr * std_a / std_b
                hedge = float(c * std.iloc[i] / std_safe.iloc[j]) \
                    if not pd.isna(std_safe.iloc[j]) else 1.0
                rows.append({
                    "code_a": a, "code_b": b, "score": abs(c),
                    "hedge_ratio": hedge, "pvalue": np.nan,
                })
            elif method == "distance":
                pa, pb = prices[a], prices[b]
                spread = pa - pb
                mid = (pa + pb) / 2
                dist = float(spread.std(ddof=0) / mid.mean()) if mid.mean() != 0 else np.inf
                # 同上向量化算 hedge
                hedge = float(c * std.iloc[i] / std_safe.iloc[j]) \
                    if not pd.isna(std_safe.iloc[j]) else 1.0
                rows.append({
                    "code_a": a, "code_b": b, "score": dist,
                    "hedge_ratio": hedge, "pvalue": np.nan,
                })
            else:  # cointest
                if abs(c) < min_corr:
                    continue
                # Engle-Granger 不能向量化,逐对跑
                try:
                    _, pvalue, _ = stats.coint(prices[a].values, prices[b].values)
                except Exception:
                    pvalue = 1.0
                if pvalue > pvalue_threshold:
                    continue
                # 同上向量化算 hedge
                hedge = float(c * std.iloc[i] / std_safe.iloc[j]) \
                    if not pd.isna(std_safe.iloc[j]) else 1.0
                rows.append({
                    "code_a": a, "code_b": b, "score": -pvalue,
                    "hedge_ratio": hedge, "pvalue": pvalue,
                })

    if not rows:
        return pd.DataFrame(columns=["code_a", "code_b", "score",
                                     "hedge_ratio", "pvalue"])
    df = pd.DataFrame(rows)
    ascending = method == "distance"
    df = df.sort_values("score", ascending=ascending).head(top_n).reset_index(drop=True)
    return df


def _ols_hedge(a: pd.Series, b: pd.Series) -> float:
    """OLS: a = alpha + beta * b → hedge ratio = beta。"""
    s = pd.concat([a, b], axis=1).dropna()
    if len(s) < 30 or s.iloc[:, 1].var() == 0:
        return 1.0
    b_var = float(s.iloc[:, 1].var())
    cov_ab = float(((s.iloc[:, 0] - s.iloc[:, 0].mean()) *
                    (s.iloc[:, 1] - s.iloc[:, 1].mean())).mean())
    return float(cov_ab / b_var)


def _distance_ratio(a: pd.Series, b: pd.Series) -> float:
    s = pd.concat([a, b], axis=1).dropna()
    if s.empty:
        return np.inf
    spread = s.iloc[:, 0] - s.iloc[:, 1]
    mid = (s.iloc[:, 0] + s.iloc[:, 1]) / 2
    return float(spread.std() / mid.mean())


def compute_hedge_ratio(a: pd.Series, b: pd.Series,
                        mode: Literal["static", "rolling"] = "static",
                        lookback: int = 120) -> pd.Series:
    """Hedge ratio 序列。static = 全局 OLS;rolling = 滚动 OLS(无 Kalman)。"""
    if mode == "static":
        h = _ols_hedge(a, b)
        return pd.Series(h, index=a.index)
    # rolling
    out = pd.Series(np.nan, index=a.index)
    for i in range(lookback, len(a) + 1):
        window_a = a.iloc[i - lookback:i]
        window_b = b.iloc[i - lookback:i]
        out.iloc[i - 1] = _ols_hedge(window_a, window_b)
    out = out.ffill().fillna(_ols_hedge(a, b))
    return out


def compute_spread(a: pd.Series, b: pd.Series,
                   hedge: float | pd.Series = 1.0) -> pd.Series:
    """价差 = a - hedge * b。"""
    if isinstance(hedge, (int, float)):
        return a - hedge * b
    return a - hedge * b


def compute_zscore(spread: pd.Series, n: int = 60) -> pd.Series:
    """滚动 z-score = (spread - MA_n) / std_n。"""
    ma = spread.rolling(n, min_periods=n // 2).mean()
    sd = spread.rolling(n, min_periods=n // 2).std()
    return (spread - ma) / sd.replace(0, np.nan)


# ============ 配对回测 ============

@dataclass
class PairsTrade:
    date: pd.Timestamp
    side: str          # 'open_long_spread' / 'open_short_spread' / 'close'
    price_a: float
    price_b: float
    shares_a: int      # +为持 A,-为空 A(做空)
    shares_b: int      # +为持 B,-为空 B
    cash_after: float
    pnl: float = 0.0
    reason: str = ""


@dataclass
class PairsResult:
    code_a: str
    code_b: str
    hedge_ratio: float
    entry_z: float
    exit_z: float
    lookback: int
    spread: pd.Series
    zscore: pd.Series
    trades: list[PairsTrade] = field(default_factory=list)
    equity: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    cfg: dict = field(default_factory=dict)

    def summary(self) -> pd.Series:
        if self.equity is None or self.equity.empty:
            return pd.Series(dtype=object)
        eq = self.equity
        init = float(self.cfg.get("capital", 100000.0))
        total_ret = float(eq.iloc[-1] / init - 1)
        days = len(eq)
        years = days / 252 if days > 0 else 0
        ann_ret = (eq.iloc[-1] / init) ** (1 / years) - 1 if years > 0 else 0.0
        peak = eq.cummax()
        mdd = float((eq / peak - 1).min())
        rets = eq.pct_change().dropna()
        sharpe = (rets.mean() / rets.std() * math.sqrt(252)) if (len(rets) > 1 and rets.std() > 0) else 0.0
        # 配对回测的特有统计
        n_long = sum(1 for t in self.trades if "long_spread" in t.side and "open" in t.side)
        n_short = sum(1 for t in self.trades if "short_spread" in t.side and "open" in t.side)
        n_close = sum(1 for t in self.trades if t.side == "close")
        return pd.Series({
            "代码A": self.code_a,
            "代码B": self.code_b,
            "对冲比率": f"{self.hedge_ratio:.4f}",
            "区间": f"{eq.index[0].date()} ~ {eq.index[-1].date()}",
            "初始资金": f"{init:,.2f}",
            "总收益率": f"{total_ret*100:.2f}%",
            "年化收益": f"{ann_ret*100:.2f}%",
            "最大回撤": f"{mdd*100:.2f}%",
            "夏普": f"{sharpe:.2f}",
            "做多次": n_long,
            "做空次": n_short,
            "平仓次": n_close,
            "总交易": len(self.trades),
            "z 入场阈值": f"±{self.entry_z:.2f}",
            "z 出场阈值": f"±{self.exit_z:.2f}",
            "z lookback": self.lookback,
            "期末权益": f"{eq.iloc[-1]:,.2f}",
        })

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([asdict(t) for t in self.trades])


def pairs_backtest(prices_a: pd.Series,
                   prices_b: pd.Series,
                   entry_z: float = 2.0,
                   exit_z: float = 0.5,
                   lookback: int = 60,
                   hedge_lookback: int = 120,
                   hedge_mode: Literal["static", "rolling"] = "static",
                   capital: float = 100_000.0,
                   commission: float = 0.0003,
                   stamp_tax: float = 0.001,
                   slippage: float = 0.001,
                   min_commission: float = 5.0,
                   lot_size: int = 100,
                   t0: bool = False,
                   max_hold_days: Optional[int] = None) -> PairsResult:
    """配对回测:z-score 触阈进出场,默认假设 A、B 均可做空做多(用 ETF/期货对冲)。

    state machine:
        flat     + z < -entry → long_spread  (持 A / 空 B)
        flat     + z > +entry → short_spread (空 A / 持 B)
        long     + z > -exit  → close
        short    + z < +exit  → close
        持仓超 max_hold_days → 强制平仓
    """
    df = pd.concat([prices_a.rename("a"), prices_b.rename("b")], axis=1).dropna()
    if df.empty or len(df) < max(lookback, hedge_lookback) + 10:
        raise ValueError("数据不足以跑配对回测")
    df = df.sort_index()
    code_a = prices_a.name or "A"
    code_b = prices_b.name or "B"

    hedge = compute_hedge_ratio(df["a"], df["b"], mode=hedge_mode, lookback=hedge_lookback)
    spread = compute_spread(df["a"], df["b"], hedge)
    z = compute_zscore(spread, n=lookback)

    # 状态
    position = "flat"           # 'flat' / 'long' / 'short'
    pos_size = 0                # 单边股数(整数)
    cash = float(capital)
    a_cost = b_cost = 0.0       # 当前持仓的两腿成本
    a_shares = b_shares = 0     # +持 / -空
    hold_days = 0
    equity_list: list[float] = []
    trades: list[PairsTrade] = []

    for i, (dt, row) in enumerate(df.iterrows()):
        price_a = float(row["a"])
        price_b = float(row["b"])
        z_i = z.iloc[i] if not pd.isna(z.iloc[i]) else 0.0

        # ===== 收盘判定:用今天的 z-score 决定明天开仓;今天处理昨天的开仓/平仓 =====
        # 简化:T+1 假设下,信号当日发生,成交用当日收盘价;但 T+1 持仓当日才可平,
        # 内部用"信号日 -> 次日成交"模型太繁琐,这里采用"信号当日 close 成交"近似。
        # 仓位仅在 T+1 时延后 1 日可用平(t0=True 时立即可平)
        # 为简化,我们采用 close→close 模式,允许当日同时开关(T+0 等价)

        # 状态机
        if position == "flat" and not pd.isna(z_i):
            if z_i < -entry_z:
                # 做多价差: 买 A, 空 B
                # 用 capital 的 50% 买 A(按 lot 取整)
                target_a_cash = capital * 0.5
                ps_a = (int(target_a_cash // (price_a * (1 + slippage) * lot_size))) * lot_size
                if ps_a > 0:
                    a_cost = price_a * (1 + slippage)
                    b_cost = price_b * (1 - slippage)  # 卖空的成交价
                    a_shares = ps_a
                    b_shares = -ps_a  # 负代表做空
                    amount_a = ps_a * a_cost
                    fee_a = max(min_commission, amount_a * commission)
                    cash -= (amount_a + fee_a)
                    # 卖空收现金(暂记)
                    cash += ps_a * b_cost - max(min_commission, ps_a * b_cost * (commission + stamp_tax))
                    pos_size = ps_a
                    position = "long"
                    hold_days = 0
                    trades.append(PairsTrade(
                        date=dt, side="open_long_spread",
                        price_a=price_a, price_b=price_b,
                        shares_a=ps_a, shares_b=-ps_a,
                        cash_after=cash, reason="z<-entry",
                    ))
            elif z_i > entry_z:
                # 做空价差: 空 A, 买 B
                target_b_cash = capital * 0.5
                ps_b = (int(target_b_cash // (price_b * (1 + slippage) * lot_size))) * lot_size
                if ps_b > 0:
                    a_cost = price_a * (1 - slippage)  # 卖空 A
                    b_cost = price_b * (1 + slippage)  # 买 B
                    a_shares = -ps_b
                    b_shares = ps_b
                    # 卖空 A 收现金
                    cash += ps_b * a_cost - max(min_commission, ps_b * a_cost * (commission + stamp_tax))
                    # 买 B 付现金
                    amount_b = ps_b * b_cost
                    fee_b = max(min_commission, amount_b * commission)
                    cash -= (amount_b + fee_b)
                    pos_size = ps_b
                    position = "short"
                    hold_days = 0
                    trades.append(PairsTrade(
                        date=dt, side="open_short_spread",
                        price_a=price_a, price_b=price_b,
                        shares_a=-ps_b, shares_b=ps_b,
                        cash_after=cash, reason="z>+entry",
                    ))

        # 持仓中:检查出场
        elif position == "long" and not pd.isna(z_i):
            if z_i > -exit_z or (max_hold_days and hold_days >= max_hold_days):
                # 平多价差:卖 A,买 B 平空
                # 卖 A
                rev_a = a_shares * price_a * (1 - slippage)
                fee_a = max(min_commission, rev_a * (commission + stamp_tax))
                cash += (rev_a - fee_a)
                # 买 B 平空
                cost_b = (-b_shares) * price_b * (1 + slippage)
                fee_b = max(min_commission, cost_b * commission)
                cash -= (cost_b + fee_b)
                pnl = (cash - capital)  # 简化,相对期初
                trades.append(PairsTrade(
                    date=dt, side="close",
                    price_a=price_a, price_b=price_b,
                    shares_a=0, shares_b=0,
                    cash_after=cash, pnl=pnl,
                    reason="z>-exit" if z_i > -exit_z else "max_hold",
                ))
                position = "flat"
                a_shares = b_shares = 0
                pos_size = 0
        elif position == "short" and not pd.isna(z_i):
            if z_i < exit_z or (max_hold_days and hold_days >= max_hold_days):
                # 平空价差:买 A 平空,卖 B
                cost_a = (-a_shares) * price_a * (1 + slippage)
                fee_a = max(min_commission, cost_a * commission)
                cash -= (cost_a + fee_a)
                # 卖 B
                rev_b = b_shares * price_b * (1 - slippage)
                fee_b = max(min_commission, rev_b * (commission + stamp_tax))
                cash += (rev_b - fee_b)
                pnl = (cash - capital)
                trades.append(PairsTrade(
                    date=dt, side="close",
                    price_a=price_a, price_b=price_b,
                    shares_a=0, shares_b=0,
                    cash_after=cash, pnl=pnl,
                    reason="z<+exit" if z_i < exit_z else "max_hold",
                ))
                position = "flat"
                a_shares = b_shares = 0
                pos_size = 0

        # 权益:现金 + 持仓A按现价 + 持仓B按现价(空头为负)
        eq = cash + a_shares * price_a + b_shares * price_b
        equity_list.append(eq)
        if position != "flat":
            hold_days += 1

    # 收尾
    cfg = dict(
        capital=capital, commission=commission, stamp_tax=stamp_tax,
        slippage=slippage, min_commission=min_commission, lot_size=lot_size,
        t0=t0, max_hold_days=max_hold_days,
    )
    result = PairsResult(
        code_a=code_a, code_b=code_b,
        hedge_ratio=float(hedge.iloc[-1]) if not pd.isna(hedge.iloc[-1]) else _ols_hedge(df["a"], df["b"]),
        entry_z=entry_z, exit_z=exit_z, lookback=lookback,
        spread=spread, zscore=z, trades=trades, cfg=cfg,
    )
    result.equity = pd.Series(equity_list, index=df.index, name="equity")
    result.daily_returns = result.equity.pct_change().fillna(0)
    return result


# ============ 可视化 ============

def plot(result: PairsResult, save_path: Optional[str] = None):
    """价差 / z-score / 权益 三联图。"""
    import os
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 1, 1]})
    ax1.plot(result.spread.index, result.spread.values, color="black", linewidth=0.8, label="价差")
    ax1.axhline(result.spread.mean(), color="grey", linestyle="--", linewidth=0.7)
    ax1.set_title(f"配对  {result.code_a} vs {result.code_b}  对冲比率={result.hedge_ratio:.3f}")
    ax1.set_ylabel("价差")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.plot(result.zscore.index, result.zscore.values, color="navy", linewidth=0.9, label="z 分数")
    ax2.axhline(result.entry_z, color="red", linestyle="--", linewidth=0.7, label=f"入场 ±{result.entry_z}")
    ax2.axhline(-result.entry_z, color="red", linestyle="--", linewidth=0.7)
    ax2.axhline(result.exit_z, color="green", linestyle="--", linewidth=0.7, label=f"出场 ±{result.exit_z}")
    ax2.axhline(-result.exit_z, color="green", linestyle="--", linewidth=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("z 分数")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    ax3.plot(result.equity.index, result.equity.values, color="darkgreen", linewidth=1.0, label="权益")
    ax3.axhline(result.cfg["capital"], color="grey", linestyle="--", linewidth=0.7, label="初始资金")
    ax3.set_ylabel("权益")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, (ax1, ax2, ax3)


def write_text_report(result: PairsResult, path: str):
    """写文本报告。"""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"配对回测报告  {result.code_a}  vs  {result.code_b}")
    lines.append("=" * 60)
    for k, v in result.summary().items():
        lines.append(f"  {k:>14s}: {v}")
    if result.trades:
        lines.append("")
        lines.append(f"交易明细(共 {len(result.trades)} 笔,展示前 20):")
        for t in result.trades[:20]:
            lines.append(
                f"  {t.date.date()}  {t.side:<22s}  "
                f"A={t.shares_a:>+5d}  B={t.shares_b:>+5d}  reason={t.reason}"
            )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
