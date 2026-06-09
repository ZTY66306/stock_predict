"""网格交易回测工具。

为 A 股设计,内置交易规则识别与回测引擎,可输入股票代码或中文名,
拉取历史 K 线后跑一次完整网格交易复盘,输出收益曲线、网格效率、交易明细。

典型用法:
    from baostock_tool import grid_backtest as gb

    # 直接传中文名,自动解析
    result = gb.run("浦发银行", start="2022-01-01", end="2024-12-31",
                    grid_mode="geometric", n_grids=10, shares_per_grid=200)
    print(result.summary())            # 一行核心指标
    print(result.trades_df().head())   # 交易明细
    eff = result.grid_efficiency()     # 每次往返的网格效率
    gb.plot(result, save_dir="./output/gb_sh600000")

    # 手动区间
    result = gb.run("sh.600000", lower=8.0, upper=12.0, n_grids=8,
                    shares_per_grid=300, base_position=1000,
                    stop_loss=0.10, take_profit=0.50)

    # 自动区间(用过去 60 日 low/high 的 0.8~1.2 倍)
    result = gb.run("sz.000001", lower_pct=0.8, upper_pct=1.2,
                    lookback=60, n_grids=15)
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

from . import client, data

# ============ A 股交易规则常量 ============

_PRICE_LIMIT_BSE = 0.30   # 北交所
_PRICE_LIMIT_20 = 0.20    # 科创板(688) / 创业板(30x)
_PRICE_LIMIT_10 = 0.10    # 沪深主板 / 中小板 / ETF / LOF
_PRICE_LIMIT_5 = 0.05     # ST / *ST
_PRICE_LIMIT_BOND = 0.30  # 可转债: 熔断机制(±20% / ±30% 临停)
_PRICE_LIMIT_NONE = 1.0   # 国债等无涨跌幅品种,用 1.0 表示"无限制"

DEFAULT_LOT = 100


# ============ 交易规则识别 ============

def detect_price_limit(code: str, name: str = "") -> float:
    """根据代码/名称推断日涨跌幅限制(返回小数,0.10 = 10%)。

    规则:
        ST/*ST                              → 5%
        科创板 (sh.688) / 创业板 (sz.30x)   → 20%
        北交所 (bj.*)                       → 30%
        可转债 (sh.110/113, sz.123)         → 30%(熔断机制上限)
        国债 / 地方债 (无涨跌幅)             → 100%(实际无限制)
        其它(主板/中小板/ETF/LOF/企业债)   → 10%
    """
    if name and ("ST" in name or "*ST" in name):
        return _PRICE_LIMIT_5
    code = code.lower()
    if code.startswith("sh.688") or code.startswith("sz.30"):
        return _PRICE_LIMIT_20
    if code.startswith("bj."):
        return _PRICE_LIMIT_BSE
    # 可转债
    if code.startswith(("sh.110", "sh.113", "sz.123")):
        return _PRICE_LIMIT_BOND
    # 国债 / 地方债 / 部分企业债(无涨跌停)
    if code.startswith(("sh.019", "sh.020", "sh.100",
                        "sz.100", "sz.101")):
        return _PRICE_LIMIT_NONE
    # 默认(主板/中小板/ETF/LOF)
    return _PRICE_LIMIT_10


def detect_t0(code: str) -> bool:
    """判断是否为 T+0 品种(可当日买卖)。

    覆盖:
        沪市 ETF/LOF (sh.5xxxxx)
        深市 ETF/LOF (sz.1xxxxx)
        沪市 可转债 / 债券 (sh.11x, sh.12x, sh.01x, sh.02x, sh.10x)
        深市 可转债 / 国债 (sz.10x, sz.123)
    """
    code = code.lower()
    # 沪深 ETF / LOF
    if code.startswith("sh.5") or code.startswith("sz.1"):
        return True
    # 沪市 债券 / 可转债 / 国债
    if code.startswith(("sh.110", "sh.113", "sh.12",
                        "sh.019", "sh.020", "sh.100")):
        return True
    # 深市 国债 / 可转债
    if code.startswith(("sz.10", "sz.101", "sz.123")):
        return True
    return False


def resolve_code(code_or_name: str) -> tuple[str, str]:
    """输入股票代码或中文名,返回 (code, name)。失败抛 ValueError。"""
    s = (code_or_name or "").strip()
    if not s:
        raise ValueError("股票代码或名称不能为空")
    if "." in s:                       # 形如 sh.600000 / sz.000001
        info = data.get_securities_info(s) or {}
        return s, info.get("code_name", "")
    df = data.get_stock_basic(code_name=s)
    if df.empty:
        raise ValueError(f"未找到匹配 '{code_or_name}' 的股票")
    row = df.iloc[0]
    return row["code"], row.get("code_name", "")


# ============ 配置 ============

@dataclass
class GridConfig:
    """网格回测配置。所有字段都给默认值,方便一行调用。"""
    code: str = ""
    name: str = ""
    start: str = "2022-01-01"
    end: Optional[str] = None

    # 资金与成本
    capital: float = 100_000.0
    commission: float = 0.0003         # 单边手续费
    stamp_tax: float = 0.001           # 印花税(卖出)
    slippage: float = 0.001            # 滑点
    min_commission: float = 5.0        # 最低手续费
    lot_size: int = DEFAULT_LOT        # 整手
    allow_fractional: bool = False     # A 股默认禁止碎股

    # 网格
    grid_mode: Literal["fixed", "geometric"] = "geometric"
    lower: Optional[float] = None      # 下沿(价);与 lower_pct 二选一
    upper: Optional[float] = None      # 上沿(价);与 upper_pct 二选一
    lower_pct: Optional[float] = None  # 下沿=lookback 期最低 * lower_pct
    upper_pct: Optional[float] = None  # 上沿=lookback 期最高 * upper_pct
    lookback: int = 60                 # 自动区间用的回看窗口(交易日)
    n_grids: int = 10                  # 网格数
    shares_per_grid: int = 100         # 每格成交股数(默认整手)
    base_position: int = 0             # 初始底仓(股)

    # 风控
    stop_loss: Optional[float] = None  # 持仓回撤止损(相对均价)
    take_profit: Optional[float] = None# 持仓浮盈止盈(相对均价)
    price_limit_override: Optional[float] = None  # 手动指定涨跌幅,跳过识别

    # 交易制度
    t0: Optional[bool] = None          # True=T+0, False=T+1, None=按 detect_t0 自动

    # 行为
    skip_base_buy: bool = False        # True=不做首日底仓建仓
    verbose: bool = False


# ============ 网格价格生成 ============

def build_grid_lines(cfg: GridConfig, ref_low: float, ref_high: float) -> list[float]:
    """根据 cfg 计算 (n_grids + 1) 个网格价格(包含上下沿)。

    等差:step = (upper - lower) / n_grids
    等比:ratio = (upper / lower) ^ (1 / n_grids)
    """
    lo, hi = cfg.lower, cfg.upper
    if lo is None:
        if cfg.lower_pct is None:
            raise ValueError("必须指定 lower 或 lower_pct")
        lo = ref_low * cfg.lower_pct
    if hi is None:
        if cfg.upper_pct is None:
            raise ValueError("必须指定 upper 或 upper_pct")
        hi = ref_high * cfg.upper_pct
    if lo >= hi:
        raise ValueError(f"网格下沿 ({lo:.4f}) 必须 < 上沿 ({hi:.4f})")
    if lo <= 0 and cfg.grid_mode == "geometric":
        raise ValueError("等比网格要求下沿 > 0")

    cfg.lower, cfg.upper = lo, hi
    if cfg.grid_mode == "fixed":
        step = (hi - lo) / cfg.n_grids
        return [lo + i * step for i in range(cfg.n_grids + 1)]
    ratio = (hi / lo) ** (1.0 / cfg.n_grids)
    return [lo * (ratio ** i) for i in range(cfg.n_grids + 1)]


# ============ 数据结构 ============

@dataclass
class GridTrade:
    """单笔网格交易。"""
    date: pd.Timestamp
    side: str               # 'buy' / 'sell'
    price: float
    grid_idx: int           # 触发的网格索引;底仓/止损/止盈为 -1
    shares: int
    cash_after: float
    position_after: int
    available_after: int
    reason: str = "grid"    # 'grid' / 'base' / 'risk_close'


@dataclass
class GridResult:
    """网格回测结果。"""
    code: str
    name: str
    cfg: GridConfig
    grid_lines: list[float]
    price_limit: float
    t0: bool = False
    trades: list[GridTrade] = field(default_factory=list)

    equity: Optional[pd.Series] = None
    cash_history: Optional[pd.Series] = None
    position_history: Optional[pd.Series] = None
    available_history: Optional[pd.Series] = None
    avg_cost_history: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None

    n_grid_buys: int = 0
    n_grid_sells: int = 0
    n_full_roundtrips: int = 0
    stock_return: float = 0.0   # 同区间内纯持有股票的收益率(对照)

    # ---- 汇总 ----
    def summary(self) -> pd.Series:
        if self.equity is None or self.equity.empty:
            return pd.Series(dtype=object)
        eq = self.equity
        cfg = self.cfg
        total_ret = float(eq.iloc[-1] / cfg.capital - 1)
        days = len(eq)
        years = days / 252 if days > 0 else 0
        ann_ret = (eq.iloc[-1] / cfg.capital) ** (1 / years) - 1 if years > 0 else 0.0
        peak = eq.cummax()
        mdd = float((eq / peak - 1).min())
        rets = eq.pct_change().dropna()
        sharpe = (rets.mean() / rets.std() * math.sqrt(252)) if (len(rets) > 1 and rets.std() > 0) else 0.0
        return pd.Series({
            "代码": self.code,
            "名称": self.name,
            "回测区间": f"{cfg.start} ~ {eq.index[-1].date()}",
            "初始资金": f"{cfg.capital:,.2f}",
            "总收益率": f"{total_ret*100:.2f}%",
            "年化收益": f"{ann_ret*100:.2f}%",
            "最大回撤": f"{mdd*100:.2f}%",
            "夏普比率": f"{sharpe:.2f}",
            "网格模式": cfg.grid_mode,
            "网格数": cfg.n_grids,
            "网格区间": f"{cfg.lower:.4f} ~ {cfg.upper:.4f}",
            "每格股数": cfg.shares_per_grid,
            "涨跌幅限制": f"{self.price_limit*100:.0f}%",
            "交易制度": "T+0" if self.t0 else "T+1",
            "交易笔数": len(self.trades),
            "  买入": self.n_grid_buys,
            "  卖出": self.n_grid_sells,
            "完整往返": self.n_full_roundtrips,
            "期末持仓": int(self.position_history.iloc[-1]) if self.position_history is not None else 0,
            "期末权益": f"{eq.iloc[-1]:,.2f}",
            "同期股票收益": f"{self.stock_return*100:.2f}%",
            "超额(网格-股票)": f"{(total_ret - self.stock_return)*100:.2f}%",
        })

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = []
        last_buy_price = None
        for t in self.trades:
            row = asdict(t)
            if t.side == "buy":
                last_buy_price = t.price
                row["round_pnl"] = ""
            else:
                if last_buy_price is not None:
                    row["round_pnl"] = f"{(t.price - last_buy_price) * t.shares:.2f}"
                    last_buy_price = None
                else:
                    row["round_pnl"] = ""
            rows.append(row)
        return pd.DataFrame(rows)

    def grid_efficiency(self) -> pd.DataFrame:
        """每次完整买入→卖出往返的网格效率(实际毛收益 / 理论最大毛收益)。"""
        if not self.trades or len(self.grid_lines) < 2:
            return pd.DataFrame()
        step_avg = (self.grid_lines[-1] - self.grid_lines[0]) / self.cfg.n_grids
        rows = []
        last_buy = None
        for t in self.trades:
            if t.side == "buy":
                last_buy = (t.date, t.price, t.shares)
            elif t.side == "sell" and last_buy is not None:
                buy_date, buy_price, shares = last_buy
                gross = (t.price - buy_price) * shares
                theoretical = step_avg * shares
                rows.append({
                    "buy_date": buy_date,
                    "sell_date": t.date,
                    "buy_price": buy_price,
                    "sell_price": t.price,
                    "shares": shares,
                    "holding_days": (t.date - buy_date).days,
                    "gross_pnl": gross,
                    "theoretical_pnl": theoretical,
                    "efficiency": (gross / theoretical) if theoretical > 0 else np.nan,
                })
                last_buy = None
        return pd.DataFrame(rows)


# ============ 主回测循环 ============

def run(code_or_name: str,
        start: str = "2022-01-01",
        end: Optional[str] = None,
        capital: float = 100_000.0,
        grid_mode: Literal["fixed", "geometric"] = "geometric",
        n_grids: int = 10,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        lower_pct: Optional[float] = None,
        upper_pct: Optional[float] = None,
        lookback: int = 60,
        shares_per_grid: int = 100,
        base_position: int = 0,
        commission: float = 0.0003,
        stamp_tax: float = 0.001,
        slippage: float = 0.001,
        min_commission: float = 5.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        price_limit_override: Optional[float] = None,
        t0: Optional[bool] = None,
        lot_size: int = DEFAULT_LOT,
        allow_fractional: bool = False,
        skip_base_buy: bool = False,
        verbose: bool = False) -> GridResult:
    """跑一次网格回测。返回 `GridResult`,包含完整交易明细、权益曲线、网格效率。

    t0:
        None  — 自动检测(ETF / 可转债 / 国债 / LOF 等 T+0 品种自动开)
        True  — 强制 T+0(当日买入当日可卖)
        False — 强制 T+1(A 股股票默认)
    """
    code, name = resolve_code(code_or_name)
    cfg = GridConfig(
        code=code, name=name, start=start, end=end, capital=capital,
        grid_mode=grid_mode, n_grids=n_grids,
        lower=lower, upper=upper, lower_pct=lower_pct, upper_pct=upper_pct,
        lookback=lookback, shares_per_grid=shares_per_grid,
        base_position=base_position,
        commission=commission, stamp_tax=stamp_tax, slippage=slippage,
        min_commission=min_commission,
        stop_loss=stop_loss, take_profit=take_profit,
        price_limit_override=price_limit_override,
        t0=t0,
        lot_size=lot_size, allow_fractional=allow_fractional,
        skip_base_buy=skip_base_buy, verbose=verbose,
    )
    return _run(cfg)


def _run(cfg: GridConfig) -> GridResult:
    client.ensure_login()
    df = data.get_kline(cfg.code, cfg.start, cfg.end, use_cache=True)
    if df.empty:
        raise ValueError(f"无 K 线数据: {cfg.code} {cfg.start}~{cfg.end}")
    df = df.sort_index()
    if "preclose" not in df.columns:
        df["preclose"] = df["close"].shift(1)

    # 自动区间参考价
    if cfg.lower is None or cfg.upper is None:
        ref_df = df.tail(cfg.lookback) if cfg.lookback > 0 else df
        ref_low = float(ref_df["low"].min())
        ref_high = float(ref_df["high"].max())
    else:
        ref_low = ref_high = 0.0

    grid_lines = build_grid_lines(cfg, ref_low, ref_high)
    price_limit = cfg.price_limit_override or detect_price_limit(cfg.code, cfg.name)
    # 交易制度:None → 按 detect_t0 自动;显式 True/False 强制
    t0_mode = cfg.t0 if cfg.t0 is not None else detect_t0(cfg.code)

    # 整百对齐
    if not cfg.allow_fractional:
        if cfg.lot_size > 1 and cfg.shares_per_grid % cfg.lot_size != 0:
            cfg.shares_per_grid -= cfg.shares_per_grid % cfg.lot_size
            if cfg.shares_per_grid == 0:
                cfg.shares_per_grid = cfg.lot_size
        if cfg.lot_size > 1 and cfg.base_position % cfg.lot_size != 0:
            cfg.base_position -= cfg.base_position % cfg.lot_size

    # 状态
    cash = float(cfg.capital)
    position = 0
    available = 0       # 今日可卖
    pending = 0         # 今日买入、明日才可卖
    cost_basis = 0.0    # 持仓总成本(含费)
    n_buys = n_sells = 0
    trades: list[GridTrade] = []

    # 底仓(首根 K 线开盘价成交)
    if cfg.base_position > 0 and not cfg.skip_base_buy:
        first_open = float(df.iloc[0]["open"])
        fill_price = first_open * (1 + cfg.slippage)
        amount = cfg.base_position * fill_price
        fee = max(cfg.min_commission, amount * cfg.commission)
        if amount + fee <= cash:
            cash -= (amount + fee)
            position = cfg.base_position
            available = cfg.base_position
            cost_basis = amount + fee
            n_buys += 1
            trades.append(GridTrade(
                date=df.index[0], side="buy", price=fill_price,
                grid_idx=-1, shares=cfg.base_position,
                cash_after=cash, position_after=position,
                available_after=available, reason="base",
            ))

    equity_list: list[float] = []
    cash_list: list[float] = []
    pos_list: list[int] = []
    avail_list: list[int] = []
    avg_cost_list: list[float] = []
    stop_triggered = False

    for i, (dt, row) in enumerate(df.iterrows()):
        # 风控触发后:以收盘价市价清仓,之后不再交易
        if stop_triggered:
            if position > 0 and available > 0:
                fill_price = float(row["close"]) * (1 - cfg.slippage)
                rev = fill_price * available
                fee = max(cfg.min_commission, rev * cfg.commission) + rev * cfg.stamp_tax
                cash += (rev - fee)
                trades.append(GridTrade(
                    date=dt, side="sell", price=fill_price, grid_idx=-1,
                    shares=available, cash_after=cash, position_after=0,
                    available_after=0, reason="risk_close",
                ))
                position = available = pending = 0
                cost_basis = 0.0
            eq = cash
            equity_list.append(eq); cash_list.append(cash)
            pos_list.append(0); avail_list.append(0); avg_cost_list.append(0.0)
            continue

        high = float(row["high"])
        low = float(row["low"])
        open_p = float(row["open"])
        pre_close = float(row["preclose"]) if not pd.isna(row["preclose"]) else open_p

        limit_up = pre_close * (1 + price_limit)
        limit_dn = pre_close * (1 - price_limit)

        avg_cost = (cost_basis / position) if position > 0 else 0.0
        mid = (grid_lines[0] + grid_lines[-1]) / 2

        # 收集今日触发的网格 — 低于中轨/均价视为买入信号,反之视为卖出
        triggers: list[tuple[str, int, float]] = []
        for idx, g in enumerate(grid_lines):
            if low <= g <= high:
                ref = avg_cost if avg_cost > 0 else mid
                if g < ref:
                    triggers.append(("buy", idx, g))
                else:
                    triggers.append(("sell", idx, g))

        # 按价格升序处理 — 这样 T+0 时低位买入的份额可立即被高位卖出;
        # 对 T+1 而言,买入记入 pending 不影响 available,行为与 sell-first 等价。
        triggers.sort(key=lambda t: t[2])
        for side, idx, g in triggers:
            if side == "buy":
                fill_price = g * (1 + cfg.slippage)
                if fill_price > limit_up:
                    continue
                if cfg.allow_fractional:
                    max_by_cash = int(cash // (fill_price + 1e-9))
                else:
                    lot = cfg.lot_size if cfg.lot_size > 1 else 1
                    max_by_cash = (int(cash // (fill_price * lot))) * lot
                buy_shares = min(max_by_cash, cfg.shares_per_grid)
                if buy_shares <= 0:
                    continue
                amount = buy_shares * fill_price
                fee = max(cfg.min_commission, amount * cfg.commission)
                if amount + fee > cash:
                    continue
                cash -= (amount + fee)
                position += buy_shares
                # T+0:当日买入当日可卖;T+1:记入 pending 等 EOD 再划拨
                if t0_mode:
                    available += buy_shares
                else:
                    pending += buy_shares
                cost_basis += (amount + fee)
                n_buys += 1
                trades.append(GridTrade(
                    date=dt, side="buy", price=fill_price, grid_idx=idx,
                    shares=buy_shares, cash_after=cash, position_after=position,
                    available_after=available, reason="grid",
                ))
            else:  # sell
                if available < cfg.shares_per_grid:
                    continue
                sell_shares = min(available, cfg.shares_per_grid)
                if (not cfg.allow_fractional) and cfg.lot_size > 1 and sell_shares % cfg.lot_size != 0:
                    sell_shares -= sell_shares % cfg.lot_size
                if sell_shares <= 0:
                    continue
                fill_price = g * (1 - cfg.slippage)
                if fill_price < limit_dn:
                    continue
                rev = sell_shares * fill_price
                fee = max(cfg.min_commission, rev * cfg.commission) + rev * cfg.stamp_tax
                cash += (rev - fee)
                position -= sell_shares
                available -= sell_shares
                if position > 0:
                    cost_basis -= cost_basis * (sell_shares / (position + sell_shares))
                else:
                    cost_basis = 0.0
                n_sells += 1
                trades.append(GridTrade(
                    date=dt, side="sell", price=fill_price, grid_idx=idx,
                    shares=sell_shares, cash_after=cash, position_after=position,
                    available_after=available, reason="grid",
                ))

        # T+1:今日买入的 pending 份额 EOD 划入 available
        if not t0_mode:
            available += pending
            pending = 0

        # 收市权益 + 风控判定
        close = float(row["close"])
        eq = cash + position * close
        if position > 0 and avg_cost > 0:
            ret = (close - avg_cost) / avg_cost
            if cfg.stop_loss is not None and ret <= -cfg.stop_loss:
                stop_triggered = True
            elif cfg.take_profit is not None and ret >= cfg.take_profit:
                stop_triggered = True

        equity_list.append(eq)
        cash_list.append(cash)
        pos_list.append(position)
        avail_list.append(available)
        avg_cost_list.append(cost_basis / position if position > 0 else 0.0)

    # 收尾
    n_full = min(n_buys, n_sells)
    result = GridResult(
        code=cfg.code, name=cfg.name, cfg=cfg,
        grid_lines=grid_lines, price_limit=price_limit, t0=t0_mode,
        trades=trades,
        n_grid_buys=n_buys, n_grid_sells=n_sells, n_full_roundtrips=n_full,
    )
    result.equity = pd.Series(equity_list, index=df.index, name="equity")
    result.cash_history = pd.Series(cash_list, index=df.index, name="cash")
    result.position_history = pd.Series(pos_list, index=df.index, name="position")
    result.available_history = pd.Series(avail_list, index=df.index, name="available")
    result.avg_cost_history = pd.Series(avg_cost_list, index=df.index, name="avg_cost")
    result.daily_returns = result.equity.pct_change().fillna(0)
    # 同期纯持有股票收益(便于对比)
    first_close = float(df.iloc[0]["close"])
    last_close = float(df.iloc[-1]["close"])
    result.stock_return = (last_close / first_close - 1) if first_close > 0 else 0.0
    return result


# ============ 可视化与文本报告 ============

def plot_price_with_grid(result: GridResult, df: Optional[pd.DataFrame] = None,
                         save_path: Optional[str] = None):
    """价格图 + 网格线 + 买卖点。"""
    import matplotlib.pyplot as plt
    if df is None:
        df = data.get_kline(result.code, result.cfg.start, result.cfg.end, use_cache=True)
    if df is None or df.empty:
        raise ValueError("无法获取 K 线数据用于绘图")

    fig, ax = plt.subplots(figsize=(14, 7))
    for g in result.grid_lines:
        ax.axhline(g, color="grey", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.plot(df.index, df["close"], color="black", linewidth=1.0, label="Close")
    ax.fill_between(df.index, df["low"], df["high"], color="lightgrey",
                    alpha=0.3, label="H/L range")
    tdf = result.trades_df() if result.trades else pd.DataFrame()
    if not tdf.empty:
        buys = tdf[tdf["side"] == "buy"]
        sells = tdf[tdf["side"] == "sell"]
        if not buys.empty:
            ax.scatter(buys["date"], buys["price"], marker="^",
                       color="red", s=60, label="Buy", zorder=5)
        if not sells.empty:
            ax.scatter(sells["date"], sells["price"], marker="v",
                       color="green", s=60, label="Sell", zorder=5)
    ax.set_title(
        f"Grid Backtest  {result.code} {result.name}  "
        f"[{result.cfg.start} ~ {df.index[-1].date()}]  "
        f"grid={result.cfg.n_grids}  limit={result.price_limit*100:.0f}%"
    )
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        if result.cfg.verbose:
            print(f"saved → {save_path}")
    return fig, ax


def plot_equity(result: GridResult, save_path: Optional[str] = None):
    """权益曲线 + 持仓/可卖持仓副图。"""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(result.equity.index, result.equity.values,
             color="navy", linewidth=1.2, label="Equity")
    ax1.axhline(result.cfg.capital, color="grey", linestyle="--",
                linewidth=0.8, label="Initial")
    ax1.set_title(
        f"Equity  {result.code} {result.name}  "
        f"final={result.equity.iloc[-1]:,.0f}  "
        f"stock_hold={result.stock_return*100:.2f}%"
    )
    ax1.set_ylabel("Equity")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(result.position_history.index, 0,
                     result.position_history.values,
                     color="steelblue", alpha=0.5, label="Position")
    ax2.plot(result.available_history.index, result.available_history.values,
             color="orange", linewidth=1.0, label="Available (T+1)")
    ax2.set_ylabel("Shares")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        if result.cfg.verbose:
            print(f"saved → {save_path}")
    return fig, (ax1, ax2)


def plot(result: GridResult, save_dir: Optional[str] = None):
    """一次性画所有图到 save_dir;若 save_dir=None 则只显示不保存。"""
    plot_price_with_grid(
        result,
        save_path=os.path.join(save_dir, "price_grid.png") if save_dir else None,
    )
    plot_equity(
        result,
        save_path=os.path.join(save_dir, "equity.png") if save_dir else None,
    )


def write_text_report(result: GridResult, path: str):
    """写文本报告(供 CLI / 离线归档)。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"网格交易回测报告 — {result.code} {result.name}")
    lines.append("=" * 60)
    for k, v in result.summary().items():
        lines.append(f"  {k:>14s}: {v}")
    lines.append("")
    lines.append("网格价格线:")
    for i, g in enumerate(result.grid_lines):
        lines.append(f"  grid[{i:>2d}] = {g:.4f}")
    if result.trades:
        lines.append("")
        lines.append(f"交易明细(共 {len(result.trades)} 笔,最多展示前 30 与后 10 笔):")
        tdf = result.trades_df()
        for _, row in tdf.head(30).iterrows():
            lines.append(
                f"  {row['date'].date()}  {row['side']:>4s}  "
                f"{row['shares']:>5d} @ {row['price']:.4f}  "
                f"reason={row['reason']}"
            )
        if len(tdf) > 40:
            lines.append("  ...")
            for _, row in tdf.tail(10).iterrows():
                lines.append(
                    f"  {row['date'].date()}  {row['side']:>4s}  "
                    f"{row['shares']:>5d} @ {row['price']:.4f}  "
                    f"reason={row['reason']}"
                )
    eff = result.grid_efficiency()
    if not eff.empty:
        lines.append("")
        lines.append("网格效率(每次完整往返):")
        lines.append(f"  平均效率  : {eff['efficiency'].mean():.2%}")
        lines.append(f"  中位效率  : {eff['efficiency'].median():.2%}")
        lines.append(f"  总毛收益  : {eff['gross_pnl'].sum():,.2f}")
        lines.append(f"  理论最大  : {eff['theoretical_pnl'].sum():,.2f}")
        lines.append(f"  完整往返数: {len(eff)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if result.cfg.verbose:
        print(f"saved → {path}")
    return path
