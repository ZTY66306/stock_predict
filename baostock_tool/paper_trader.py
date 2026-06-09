"""实盘模拟器 (Paper Trader) — 把策略接到"准实时"信号上。

提供:
    - PaperTrader              跟踪虚拟持仓,按策略生成每日推荐
    - 状态持久化(JSON 文件)   进程重启不丢
    - 每日报告(文本 / CSV)   持仓 / 盈亏 / 推荐
    - 可选 webhook 通知        触发信号时推送到钉钉 / 飞书 / Slack

典型用法:
    from baostock_tool import paper_trader as pt

    trader = pt.PaperTrader(
        strategies={"sh.600000": "ma_cross", "sh.510300": "smart_dca"},
        params={"sh.600000": {"short": 5, "long": 20}},
        initial_cash=100_000,
        state_path="./paper_state.json",
    )
    report = trader.run_once()       # 跑一次:取最新 K 线,跑策略,生成推荐
    print(report)
    trader.save_state()              # 落盘
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Literal, Optional

import numpy as np
import pandas as pd

from . import backtest, data, strategy

logger = logging.getLogger(__name__)


# ============ 数据结构 ============

@dataclass
class PaperPosition:
    """虚拟持仓(单只股票)。"""
    code: str
    shares: int = 0
    avg_cost: float = 0.0
    cost_basis: float = 0.0        # 含费
    last_price: float = 0.0
    realized_pnl: float = 0.0

    def market_value(self) -> float:
        return self.shares * self.last_price

    def unrealized_pnl(self) -> float:
        return (self.last_price - self.avg_cost) * self.shares if self.shares > 0 else 0.0


@dataclass
class PaperSignal:
    """单只股票的当日推荐。"""
    date: pd.Timestamp
    code: str
    name: str
    strategy: str
    signal: int                 # -1 / 0 / +1
    position_target: int        # 0 / 1
    last_price: float
    reason: str = ""
    shares_to_trade: int = 0    # 建议买卖股数


@dataclass
class PaperReport:
    """单次运行报告。"""
    date: pd.Timestamp
    cash: float
    market_value: float
    total_equity: float
    total_pnl: float
    pnl_ratio: float
    positions: list[PaperPosition]
    signals: list[PaperSignal]
    log: list[str] = field(default_factory=list)

    def summary(self) -> pd.Series:
        return pd.Series({
            "日期": str(self.date.date()),
            "现金": f"{self.cash:,.2f}",
            "市值": f"{self.market_value:,.2f}",
            "总权益": f"{self.total_equity:,.2f}",
            "浮动盈亏": f"{self.total_pnl:,.2f}",
            "盈亏比": f"{self.pnl_ratio*100:.2f}%",
            "持仓数": sum(1 for p in self.positions if p.shares > 0),
            "信号数": sum(1 for s in self.signals if s.signal != 0),
        })

    def signals_df(self) -> pd.DataFrame:
        if not self.signals:
            return pd.DataFrame()
        return pd.DataFrame([asdict(s) for s in self.signals])

    def positions_df(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame()
        return pd.DataFrame([asdict(p) for p in self.positions])


# ============ PaperTrader ============

class PaperTrader:
    """实盘模拟器:跟踪虚拟持仓,按策略生成每日推荐。"""

    def __init__(self,
                 strategies: dict[str, str | strategy.Strategy],
                 params: Optional[dict[str, dict]] = None,
                 initial_cash: float = 100_000.0,
                 commission: float = 0.0003,
                 stamp_tax: float = 0.001,
                 slippage: float = 0.001,
                 min_commission: float = 5.0,
                 lot_size: int = 100,
                 position_size_pct: float = 1.0,
                 lookback_days: int = 365,
                 state_path: Optional[str] = None,
                 webhook_url: Optional[str] = None,
                 log_path: Optional[str] = None):
        """
        strategies: {code: strategy_name_or_Strategy}
        params: {code: {param: value}}  每只股票可独立参数
        initial_cash: 初始虚拟资金
        position_size_pct: 每次信号使用的资金比例(0~1,默认 1 = all-in)
        state_path: 状态持久化文件路径
        webhook_url: 信号触发时 POST 的 URL(可选)
        log_path: 日志文件路径(可选)
        """
        self.strategies = strategies
        self.params = params or {}
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.slippage = slippage
        self.min_commission = min_commission
        self.lot_size = lot_size
        self.position_size_pct = max(0.0, min(1.0, position_size_pct))
        self.lookback_days = lookback_days
        self.state_path = state_path
        self.webhook_url = webhook_url
        self.log_path = log_path
        self.positions: dict[str, PaperPosition] = {c: PaperPosition(code=c) for c in strategies}
        # 状态
        self.last_run: Optional[str] = None
        self.history: list[dict] = []
        if state_path and os.path.exists(state_path):
            self.load_state()

    # ---- 状态持久化 ----
    def save_state(self) -> str:
        if not self.state_path:
            return ""
        data_ = {
            "cash": self.cash,
            "positions": {c: asdict(p) for c, p in self.positions.items()},
            "last_run": self.last_run,
            "history": self.history[-100:],   # 最近 100 条
        }
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data_, f, ensure_ascii=False, indent=2, default=str)
        return self.state_path

    def load_state(self) -> None:
        if not self.state_path or not os.path.exists(self.state_path):
            return
        with open(self.state_path, "r", encoding="utf-8") as f:
            data_ = json.load(f)
        self.cash = data_.get("cash", self.initial_cash)
        for c, p in data_.get("positions", {}).items():
            self.positions[c] = PaperPosition(**p)
        self.last_run = data_.get("last_run")
        self.history = data_.get("history", [])

    # ---- 核心:跑一次 ----
    def run_once(self, date: Optional[str] = None) -> PaperReport:
        """跑一次模拟:取最新 K 线 → 跑策略 → 生成推荐 → 模拟成交。

        date: 'YYYY-MM-DD',None = 用今天
        """
        from . import client
        client.ensure_login()
        from .grid_backtest import resolve_code
        signals: list[PaperSignal] = []
        log: list[str] = []
        # 用最新一天的日期
        run_date = date or datetime.now().strftime("%Y-%m-%d")
        for code, strat in self.strategies.items():
            try:
                # 拉 lookback 天的 K 线
                start_dt = (pd.Timestamp(run_date) - pd.Timedelta(days=self.lookback_days)) \
                           .strftime("%Y-%m-%d")
                df = data.get_kline(code, start_dt, run_date, use_cache=True)
                if df.empty or len(df) < 30:
                    log.append(f"[{code}] 无数据 / 数据不足")
                    continue
                # 解析名称(网络失败时退回用 code 本身,不静默吞)
                name = code
                try:
                    name = resolve_code(code)[1] or code
                except Exception as e:
                    logger.debug("resolve_code(%s) 失败, 用 code 兜底: %s", code, e)
                last_price = float(df["close"].iloc[-1])
                pos = self.positions[code]
                pos.last_price = last_price
                # 跑策略
                strat_name = strat if isinstance(strat, str) else strat.name
                params = self.params.get(code, {})
                sig = strategy.run_strategy(strat_name, df, params)
                signal_val = int(sig["signal"].iloc[-1])
                position_target = int(sig["position"].iloc[-1])
                # 计算建议股数
                shares_to_trade = self._calc_trade_shares(
                    signal_val, position_target, last_price, pos
                )
                # 如果要开仓 / 平仓,模拟成交
                if signal_val != 0:
                    self._execute(code, last_price, signal_val, shares_to_trade)
                signals.append(PaperSignal(
                    date=df.index[-1], code=code, name=name,
                    strategy=strat_name, signal=signal_val,
                    position_target=position_target, last_price=last_price,
                    reason=self._reason(signal_val, position_target, pos),
                    shares_to_trade=shares_to_trade,
                ))
            except Exception as e:
                log.append(f"[{code}] 失败: {type(e).__name__}: {e}")
        # 报告
        market_value = sum(p.market_value() for p in self.positions.values())
        total_equity = self.cash + market_value
        total_pnl = total_equity - self.initial_cash
        pnl_ratio = total_pnl / self.initial_cash if self.initial_cash > 0 else 0
        report = PaperReport(
            date=pd.Timestamp(run_date),
            cash=self.cash, market_value=market_value,
            total_equity=total_equity, total_pnl=total_pnl,
            pnl_ratio=pnl_ratio,
            positions=list(self.positions.values()),
            signals=signals, log=log,
        )
        self.last_run = run_date
        self.history.append({
            "date": run_date,
            "total_equity": total_equity,
            "cash": self.cash,
            "market_value": market_value,
        })
        # 写日志
        if self.log_path:
            self._write_log(report)
        # 推 webhook
        if self.webhook_url and signals:
            self._send_webhook(signals, report)
        return report

    def _calc_trade_shares(self, signal_val: int, position_target: int,
                            last_price: float, pos: PaperPosition) -> int:
        """计算本次建议成交股数。整百。"""
        if signal_val == 0:
            return 0
        if signal_val == 1:    # 买入
            # 用 position_size_pct 的现金买,整百
            budget = self.cash * self.position_size_pct
            if budget < last_price * self.lot_size:
                return 0
            raw = (int(budget // (last_price * self.lot_size))) * self.lot_size
            return max(0, raw)
        if signal_val == -1:   # 卖出
            return min(pos.shares, pos.shares - (pos.shares % self.lot_size)) \
                if pos.shares >= self.lot_size else pos.shares

    def _execute(self, code: str, price: float, signal_val: int, shares: int) -> None:
        """模拟成交:更新 cash / position。"""
        if shares <= 0:
            return
        pos = self.positions[code]
        if signal_val == 1:    # 买入
            fill_price = price * (1 + self.slippage)
            amount = shares * fill_price
            fee = max(self.min_commission, amount * self.commission)
            if amount + fee > self.cash:
                # 资金不足,缩小股数
                shares = int((self.cash * 0.99) // (fill_price * self.lot_size)) * self.lot_size
                if shares < self.lot_size:
                    return
                amount = shares * fill_price
                fee = max(self.min_commission, amount * self.commission)
            self.cash -= (amount + fee)
            new_total_cost = pos.cost_basis + amount + fee
            new_shares = pos.shares + shares
            pos.avg_cost = new_total_cost / new_shares if new_shares > 0 else 0
            pos.cost_basis = new_total_cost
            pos.shares = new_shares
        elif signal_val == -1 and pos.shares > 0:   # 卖出
            sell_shares = min(shares, pos.shares)
            if self.lot_size > 1 and sell_shares % self.lot_size != 0:
                sell_shares -= sell_shares % self.lot_size
            if sell_shares <= 0:
                return
            fill_price = price * (1 - self.slippage)
            revenue = sell_shares * fill_price
            fee = max(self.min_commission, revenue * self.commission) \
                  + revenue * self.stamp_tax
            self.cash += (revenue - fee)
            pnl = (fill_price - pos.avg_cost) * sell_shares - fee
            pos.realized_pnl += pnl
            # 按比例扣减成本
            if pos.shares > 0:
                pos.cost_basis *= (1 - sell_shares / pos.shares)
            pos.shares -= sell_shares
            if pos.shares == 0:
                pos.avg_cost = 0
                pos.cost_basis = 0

    def _reason(self, signal_val: int, position_target: int, pos: PaperPosition) -> str:
        if signal_val == 0:
            return "hold"
        if signal_val == 1:
            return "open" if pos.shares == 0 else "add"
        if signal_val == -1:
            return "close" if pos.shares > 0 else "no_pos"

    def _write_log(self, report: PaperReport) -> None:
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {report.date.date()} ===\n")
            f.write(report.summary().to_string())
            f.write("\n--- 推荐信号 ---\n")
            for s in report.signals:
                if s.signal != 0:
                    f.write(
                        f"  {s.code} {s.name}  {s.strategy}  "
                        f"signal={s.signal:+d}  shares={s.shares_to_trade}  "
                        f"reason={s.reason}\n"
                    )

    def _send_webhook(self, signals: list[PaperSignal], report: PaperReport) -> None:
        """POST 推荐到 webhook(钉钉/飞书/Slack 通用 JSON)。"""
        try:
            import requests
            payload = {
                "date": str(report.date.date()),
                "total_equity": report.total_equity,
                "signals": [
                    {"code": s.code, "name": s.name, "signal": s.signal,
                     "shares": s.shares_to_trade, "price": s.last_price,
                     "reason": s.reason}
                    for s in signals if s.signal != 0
                ],
            }
            requests.post(self.webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.warning("webhook 推送失败: %s", e)


# ============ 文本报告 ============

def write_text_report(report: PaperReport, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 60)
    lines.append(f"Paper Trader 日报  {report.date.date()}")
    lines.append("=" * 60)
    for k, v in report.summary().items():
        lines.append(f"  {k:>14s}: {v}")
    if report.positions:
        lines.append("")
        lines.append("当前持仓:")
        for p in report.positions:
            if p.shares > 0:
                lines.append(
                    f"  {p.code:<12s} shares={p.shares:>5d}  "
                    f"cost={p.avg_cost:.4f}  price={p.last_price:.4f}  "
                    f"unrealized_pnl={p.unrealized_pnl():.2f}  "
                    f"realized_pnl={p.realized_pnl:.2f}"
                )
    if report.signals:
        lines.append("")
        lines.append("今日信号:")
        for s in report.signals:
            if s.signal != 0:
                lines.append(
                    f"  {s.date.date()}  {s.code}  {s.name}  "
                    f"signal={s.signal:+d}  shares={s.shares_to_trade}  "
                    f"reason={s.reason}"
                )
    if report.log:
        lines.append("")
        lines.append("运行日志:")
        for line in report.log:
            lines.append(f"  {line}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
