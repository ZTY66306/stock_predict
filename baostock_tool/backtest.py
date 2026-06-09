"""事件驱动回测引擎。

用法:
    from baostock_tool.backtest import BacktestEngine, BacktestConfig
    from baostock_tool.strategy import run_strategy

    sig = run_strategy("ma_cross", df, {"short": 5, "long": 20})
    engine = BacktestEngine(BacktestConfig(initial_cash=100000, commission=0.0003, slippage=0.001))
    result = engine.run(df, sig)
    print(result.summary())

输入:
    df:  K 线 DataFrame(必须有 close / open 列)
    sig: 包含 signal(+1/-1/0) 和 position(0/1) 列
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_cash: float = 100000.0
    commission: float = 0.0003     # 单边手续费(万三)
    stamp_tax: float = 0.001       # 印花税(卖出)
    slippage: float = 0.001        # 滑点(成交价 * (1 + slippage) 买入,卖出减)
    min_commission: float = 5.0    # 最低手续费
    lot_size: int = 100            # A 股最小买入 100 股
    allow_fractional: bool = False # 是否允许碎股
    stop_loss: Optional[float] = None   # 止损比例(0.05=5%)
    take_profit: Optional[float] = None # 止盈比例
    annual_trading_days: int = 252
    risk_free_rate: float = 0.0     # 年化无风险利率
    position_sizer: str = "all-in"  # 仓位策略名(预留,见 position.py)


@dataclass
class Trade:
    date: pd.Timestamp
    side: str       # 'buy' / 'sell'
    price: float
    shares: int
    cash_after: float
    pnl: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade] = field(default_factory=list)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    cfg: BacktestConfig = field(default_factory=BacktestConfig)

    # ---- 指标 ----
    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.cfg.initial_cash - 1)

    @property
    def annual_return(self) -> float:
        days = len(self.equity)
        if days == 0:
            return 0.0
        years = days / self.cfg.annual_trading_days
        if years <= 0:
            return 0.0
        return (self.equity.iloc[-1] / self.cfg.initial_cash) ** (1 / years) - 1

    @property
    def max_drawdown(self) -> float:
        peak = self.equity.cummax()
        dd = self.equity / peak - 1
        return float(dd.min())

    @property
    def sharpe(self) -> float:
        rets = self.equity.pct_change().dropna()
        if rets.std() == 0 or len(rets) < 2:
            return 0.0
        return float(rets.mean() / rets.std() * math.sqrt(self.cfg.annual_trading_days))

    @property
    def win_rate(self) -> float:
        # 配对买卖:奇数次买入 -> 偶数次卖出
        pairs = self._round_trips()
        if not pairs:
            return 0.0
        wins = sum(1 for p in pairs if p > 0)
        return wins / len(pairs)

    @property
    def profit_factor(self) -> float:
        pairs = self._round_trips()
        if not pairs:
            return 0.0
        gross_profit = sum(p for p in pairs if p > 0)
        gross_loss = -sum(p for p in pairs if p < 0)
        if gross_loss == 0:
            return float("inf")
        return gross_profit / gross_loss

    def _round_trips(self) -> list[float]:
        """根据买卖记录配对计算每笔收益"""
        out: list[float] = []
        cost = None
        shares = 0
        for t in self.trades:
            if t.side == "buy" and cost is None:
                cost = t.price
                shares = t.shares
            elif t.side == "sell" and cost is not None:
                out.append((t.price - cost) * shares - t.price * shares * (self.cfg.commission + self.cfg.stamp_tax) - cost * shares * self.cfg.commission)
                cost = None
                shares = 0
        return out

    def summary(self) -> pd.Series:
        return pd.Series({
            "总收益率": f"{self.total_return * 100:.2f}%",
            "年化收益": f"{self.annual_return * 100:.2f}%",
            "最大回撤": f"{self.max_drawdown * 100:.2f}%",
            "夏普比率": f"{self.sharpe:.2f}",
            "胜率": f"{self.win_rate * 100:.2f}%",
            "盈亏比": f"{self.profit_factor:.2f}" if math.isfinite(self.profit_factor) else "∞",
            "交易次数": len(self.trades),
        })

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([asdict(t) for t in self.trades])

    # ---- 收益序列 ----
    @property
    def daily_returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0)

    @property
    def downside_returns(self) -> pd.Series:
        return self.daily_returns.clip(upper=0)

    # ---- 风险指标 ----
    @property
    def sortino(self) -> float:
        """Sortino 比率:超额收益 / 下行波动率。"""
        rets = self.daily_returns
        if rets.std() == 0 or len(rets) < 2:
            return 0.0
        downside = rets.clip(upper=0)
        dd_std = downside.std()
        if dd_std == 0:
            return 0.0
        ann = self.cfg.annual_trading_days
        excess = rets.mean() - self.cfg.risk_free_rate / ann
        return float(excess / dd_std * math.sqrt(ann))

    @property
    def calmar(self) -> float:
        """Calmar 比率:年化收益 / 最大回撤绝对值。"""
        mdd = abs(self.max_drawdown)
        if mdd == 0:
            return 0.0
        return float(self.annual_return / mdd)

    def var(self, p: float = 0.05) -> float:
        """历史 VaR(损失为正数)。"""
        rets = self.daily_returns.dropna()
        if rets.empty:
            return 0.0
        return float(-rets.quantile(p))

    def cvar(self, p: float = 0.05) -> float:
        """历史 CVaR / Expected Shortfall(尾部损失均值)。"""
        rets = self.daily_returns.dropna()
        if rets.empty:
            return 0.0
        cutoff = rets.quantile(p)
        return float(-rets[rets <= cutoff].mean())

    def alpha_beta(self, bench: pd.Series) -> tuple[float, float]:
        """对基准的 alpha(年化) + beta。"""
        rets = self.daily_returns
        bench = bench.reindex(rets.index).ffill().pct_change().fillna(0)
        common = rets.index.intersection(bench.index)
        if len(common) < 30:
            return 0.0, 0.0
        y = rets.loc[common].values
        x = bench.loc[common].values
        if x.var() == 0:
            return 0.0, 0.0
        cov = np.cov(y, x, ddof=1)
        beta = float(cov[0, 1] / cov[1, 1])
        alpha_daily = float(y.mean() - beta * x.mean())
        alpha_ann = alpha_daily * self.cfg.annual_trading_days
        return alpha_ann, beta

    def information_ratio(self, bench: pd.Series) -> float:
        """信息比率:主动收益 / 跟踪误差。"""
        rets = self.daily_returns
        bench = bench.reindex(rets.index).ffill().pct_change().fillna(0)
        active = (rets - bench).dropna()
        if active.std() == 0:
            return 0.0
        return float(active.mean() / active.std() * math.sqrt(self.cfg.annual_trading_days))

    def risk_metrics(self, benchmark: Optional[pd.Series] = None) -> pd.Series:
        """一次性返回全部风险与绩效指标。"""
        out = {
            "总收益率": self.total_return,
            "年化收益": self.annual_return,
            "最大回撤": self.max_drawdown,
            "夏普比率": self.sharpe,
            "Sortino": self.sortino,
            "Calmar": self.calmar,
            "VaR(5%)": self.var(0.05),
            "CVaR(5%)": self.cvar(0.05),
            "胜率": self.win_rate,
            "盈亏比": self.profit_factor,
            "交易次数": len(self.trades),
        }
        if benchmark is not None and not benchmark.empty:
            alpha, beta = self.alpha_beta(benchmark)
            out["Alpha(年化)"] = alpha
            out["Beta"] = beta
            out["IR"] = self.information_ratio(benchmark)
        return pd.Series(out)

    def monthly_returns(self) -> pd.DataFrame:
        """月度收益透视表: 行=年,列=月。"""
        rets = self.daily_returns
        df = rets.to_frame("ret")
        df["year"] = df.index.year
        df["month"] = df.index.month
        pivot = df.pivot_table(index="year", columns="month", values="ret", aggfunc=lambda x: (1 + x).prod() - 1)
        return pivot

    def yearly_returns(self) -> pd.Series:
        rets = self.daily_returns
        return rets.groupby(rets.index.year).apply(lambda x: (1 + x).prod() - 1)


class BacktestEngine:
    def __init__(self, cfg: Optional[BacktestConfig] = None):
        self.cfg = cfg or BacktestConfig()

    def _commission(self, amount: float) -> float:
        return max(self.cfg.min_commission, amount * self.cfg.commission)

    def run(self, df: pd.DataFrame, sig: pd.DataFrame) -> BacktestResult:
        """执行回测。sig 必须与 df 同索引,含 'signal' 和 'position' 列。"""
        if not df.index.equals(sig.index):
            raise ValueError("df 与 sig 索引必须一致")

        cfg = self.cfg
        cash = cfg.initial_cash
        shares = 0
        cost = 0.0          # 当前持仓成本
        equity = []
        trades: list[Trade] = []
        sigs = sig.copy()

        for i, (dt, row) in enumerate(df.iterrows()):
            price = float(row["close"])
            open_p = float(row["open"])
            sig_val = int(sigs.iloc[i]["signal"])
            pos = int(sigs.iloc[i]["position"])

            # 止损止盈(收盘时检查)
            note_col = sigs.columns.get_loc("note") if "note" in sigs.columns else None
            if shares > 0 and cost > 0:
                ret = (price - cost) / cost
                if cfg.stop_loss and ret <= -cfg.stop_loss:
                    sig_val = -1
                    sigs.iloc[i, sigs.columns.get_loc("signal")] = -1
                    if note_col is not None:
                        sigs.iloc[i, note_col] = "stop_loss"
                elif cfg.take_profit and ret >= cfg.take_profit:
                    sig_val = -1
                    sigs.iloc[i, sigs.columns.get_loc("signal")] = -1
                    if note_col is not None:
                        sigs.iloc[i, note_col] = "take_profit"

            # 优先用次日开盘价成交(更贴近实际),这里近似用 open
            exec_price = open_p if i + 1 < len(df) else price

            if sig_val == 1 and shares == 0 and pos != 0:
                # 买入
                if cfg.allow_fractional:
                    can_buy_shares = cash / (exec_price * (1 + cfg.slippage))
                else:
                    can_buy_shares = (cash // (exec_price * (1 + cfg.slippage) * cfg.lot_size)) * cfg.lot_size
                if can_buy_shares > 0:
                    fill_price = exec_price * (1 + cfg.slippage)
                    amount = can_buy_shares * fill_price
                    fee = self._commission(amount)
                    if amount + fee <= cash:
                        cash -= (amount + fee)
                        shares = int(can_buy_shares)
                        cost = fill_price
                        trades.append(Trade(dt, "buy", fill_price, shares, cash, reason="signal"))
            elif sig_val == -1 and shares > 0:
                # 卖出
                fill_price = exec_price * (1 - cfg.slippage)
                amount = shares * fill_price
                fee = self._commission(amount) + amount * cfg.stamp_tax
                cash += (amount - fee)
                pnl = (fill_price - cost) * shares - fee - self._commission(shares * cost)
                trades.append(Trade(dt, "sell", fill_price, shares, cash, pnl=pnl, reason="signal"))
                shares = 0
                cost = 0.0

            # 计算当日权益: 现金 + 持仓 * 收盘价
            equity.append(cash + shares * price)

        equity_series = pd.Series(equity, index=df.index, name="equity")
        return BacktestResult(
            equity=equity_series,
            trades=trades,
            signals=sigs,
            cfg=cfg,
        )


# ============ 多股票组合回测(等权) ============

def backtest_portfolio(
    prices: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    cfg: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """等权组合回测。prices: 列=code, 索引=日期; signals: {code: signal_df}"""
    cfg = cfg or BacktestConfig()
    aligned = prices.sort_index()
    rets = aligned.pct_change().fillna(0)
    # 把每只股票的 position 扩到整个组合
    pos_df = pd.DataFrame(0.0, index=aligned.index, columns=aligned.columns)
    for code, sig in signals.items():
        if code in pos_df.columns and "position" in sig.columns:
            pos_df[code] = sig["position"].reindex(aligned.index).fillna(0)
    n_active = pos_df.sum(axis=1).replace(0, np.nan)
    weight = pos_df.div(n_active, axis=0).fillna(0)
    # 收益: 持仓下一天的收益按仓位加权
    strategy_ret = (weight.shift(1).fillna(0) * rets).sum(axis=1).fillna(0)
    # 考虑手续费
    turnover = weight.diff().abs().sum(axis=1).fillna(0)
    strategy_ret = strategy_ret - turnover * cfg.commission - weight.shift(1).fillna(0).sum(axis=1) * cfg.commission
    equity = (1 + strategy_ret).cumprod() * cfg.initial_cash
    return BacktestResult(equity=equity, cfg=cfg)
