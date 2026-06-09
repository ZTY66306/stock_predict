"""选股器:多因子筛选,支持技术面/形态/基本面条件组合。

用法:
    from baostock_tool.screener import Screen, screen

    # 内置快捷筛选
    picks = screen("low_pe_high_turnover", date="2024-06-01")

    # 自定义
    s = Screen(date="2024-06-01")
    s.add(MA5_gt_MA20).add(macd_golden_cross).add(pe_between(0, 30))
    picks = s.run(limit=20)

所有条件函数签名: (df: pd.DataFrame) -> pd.Series[bool]
df 含至少 close/open/high/low/volume 字段,以及可选 peTTM/pbMRQ/pctChg 等。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import client, data, indicators as ind
from .patterns import BULLISH_PATTERNS, BEARISH_PATTERNS, list_patterns

logger = logging.getLogger(__name__)

Condition = Callable[[pd.DataFrame], pd.Series]
SCREEN_BATCH = 200  # 一次批量取多少只


# ============ 内置条件 ============

def MA5_gt_MA20(df: pd.DataFrame) -> pd.Series:
    return ind.MA(df["close"], 5) > ind.MA(df["close"], 20)


def MA20_gt_MA60(df: pd.DataFrame) -> pd.Series:
    return ind.MA(df["close"], 20) > ind.MA(df["close"], 60)


def macd_golden_cross(df: pd.DataFrame) -> pd.Series:
    dif, dea, _ = ind.MACD(df["close"])
    return ind.golden_cross(dif, dea)


def macd_death_cross(df: pd.DataFrame) -> pd.Series:
    dif, dea, _ = ind.MACD(df["close"])
    return ind.death_cross(dif, dea)


def kdj_oversold(df: pd.DataFrame) -> pd.Series:
    k, d, j = ind.KDJ(df)
    return (j < 20) & (k > d)


def kdj_overbought(df: pd.DataFrame) -> pd.Series:
    k, d, j = ind.KDJ(df)
    return (j > 100) & (k < d)


def rsi_oversold(df: pd.DataFrame) -> pd.Series:
    return ind.RSI(df["close"], 14) < 30


def rsi_overbought(df: pd.DataFrame) -> pd.Series:
    return ind.RSI(df["close"], 14) > 70


def boll_lower_break(df: pd.DataFrame) -> pd.Series:
    _, _, lo = ind.BOLL(df["close"])
    return df["close"] < lo


def boll_upper_break(df: pd.DataFrame) -> pd.Series:
    _, up, _ = ind.BOLL(df["close"])
    return df["close"] > up


def volume_breakout(df: pd.DataFrame, ratio: float = 2.0) -> pd.Series:
    """成交量是 20 日均量的 ratio 倍以上"""
    avg20 = df["volume"].rolling(20, min_periods=5).mean()
    return df["volume"] >= avg20 * ratio


def new_high_n(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["close"] >= df["close"].rolling(n, min_periods=1).max()


def new_low_n(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["close"] <= df["close"].rolling(n, min_periods=1).min()


def momentum_n(df: pd.DataFrame, n: int = 20, thr: float = 0.1) -> pd.Series:
    return df["close"].pct_change(n) > thr


def pe_between(lo: float, hi: float) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        if "peTTM" not in df.columns:
            return pd.Series(False, index=df.index)
        pe = pd.to_numeric(df["peTTM"], errors="coerce")
        return (pe > lo) & (pe < hi)
    _cond.__name__ = f"pe_between_{lo}_{hi}"
    return _cond


def pb_between(lo: float, hi: float) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        if "pbMRQ" not in df.columns:
            return pd.Series(False, index=df.index)
        pb = pd.to_numeric(df["pbMRQ"], errors="coerce")
        return (pb > lo) & (pb < hi)
    _cond.__name__ = f"pb_between_{lo}_{hi}"
    return _cond


def pct_change_between(lo: float, hi: float) -> Condition:
    def _cond(df: pd.DataFrame) -> pd.Series:
        if "pctChg" not in df.columns:
            return pd.Series(False, index=df.index)
        pc = pd.to_numeric(df["pctChg"], errors="coerce")
        return (pc > lo) & (pc < hi)
    _cond.__name__ = f"pctChg_{lo}_{hi}"
    return _cond


# ============ 形态条件(包装 patterns.py) ============

def pattern(name: str) -> Condition:
    """把 K 线形态包装为筛股条件。name: 形态名,如 'engulfing_bullish' / 'morning_star'。"""
    from . import patterns as ptn
    func = ptn.detect
    def _cond(df: pd.DataFrame) -> pd.Series:
        return func(df, name)
    _cond.__name__ = f"pattern_{name}"
    return _cond


def pattern_bullish_score(min_score: int = 2) -> Condition:
    """多形态看涨共振:同时出现 ≥ min_score 个看涨形态时命中。"""
    from . import patterns as ptn
    def _cond(df: pd.DataFrame) -> pd.Series:
        return ptn.pattern_score(df, bullish=True) >= min_score
    _cond.__name__ = f"pattern_bullish_ge_{min_score}"
    return _cond


def pattern_bearish_score(max_score: int = -2) -> Condition:
    """多形态看跌共振。"""
    from . import patterns as ptn
    def _cond(df: pd.DataFrame) -> pd.Series:
        return ptn.pattern_score(df, bullish=False) <= max_score
    _cond.__name__ = f"pattern_bearish_le_{max_score}"
    return _cond


# ============ 量价条件 ============

def turnover_high(quantile: float = 0.8, n: int = 20) -> Condition:
    """换手率处于近 n 日分位以上。"""
    def _cond(df: pd.DataFrame) -> pd.Series:
        if "turn" not in df.columns:
            return pd.Series(False, index=df.index)
        t = pd.to_numeric(df["turn"], errors="coerce")
        th = t.rolling(n, min_periods=5).quantile(quantile)
        return t >= th
    _cond.__name__ = f"turnover_high_{quantile}"
    return _cond


def amount_expand(ratio: float = 1.5, n: int = 10) -> Condition:
    """成交额是 n 日均额的 ratio 倍。"""
    def _cond(df: pd.DataFrame) -> pd.Series:
        if "amount" not in df.columns:
            return pd.Series(False, index=df.index)
        ma = df["amount"].rolling(n, min_periods=3).mean()
        return df["amount"] >= ma * ratio
    _cond.__name__ = f"amount_expand_{ratio}"
    return _cond


# ============ Screen 容器 ============

@dataclass
class Screen:
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    lookback_days: int = 120
    universe: Optional[list[str]] = None     # None = 全市场
    conditions: list[Condition] = field(default_factory=list)
    name: str = "custom"

    def add(self, *conds: Condition) -> "Screen":
        self.conditions.extend(conds)
        return self

    def run(self, limit: Optional[int] = None, show_progress: bool = True) -> pd.DataFrame:
        client.ensure_login()
        from datetime import datetime, timedelta
        end = self.date
        start = (datetime.strptime(self.date, "%Y-%m-%d") - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")

        codes = self.universe or data.get_all_codes(end)
        if not codes:
            logger.warning("无可用股票代码")
            return pd.DataFrame()

        results: list[dict] = []
        iterator = tqdm(codes, desc=f"筛选[{self.name}]") if show_progress else codes
        for code in iterator:
            try:
                df = data.get_kline(code, start, end)
                if df.empty or len(df) < 30:
                    continue
                # 把所有条件逐个 AND
                mask = pd.Series(True, index=df.index)
                for cond in self.conditions:
                    try:
                        m = cond(df)
                        if m is None or m.empty:
                            continue
                        mask &= m.fillna(False)
                    except Exception as e:  # 单条件失败不影响整体
                        logger.debug("条件 %s 在 %s 失败: %s", getattr(cond, "__name__", cond), code, e)
                if mask.iloc[-1]:
                    last = df.iloc[-1]
                    results.append({
                        "code": code,
                        "close": float(last["close"]),
                        "pctChg": float(last.get("pctChg", 0) or 0),
                        "volume": float(last.get("volume", 0) or 0),
                        "peTTM": float(last.get("peTTM", 0) or 0),
                        "pbMRQ": float(last.get("pbMRQ", 0) or 0),
                    })
            except Exception as e:
                logger.debug("处理 %s 失败: %s", code, e)
                continue
            if limit and len(results) >= limit * 5:
                # 提前取够,避免无谓查询
                break

        df_out = pd.DataFrame(results)
        if df_out.empty:
            return df_out
        if limit:
            df_out = df_out.head(limit)
        return df_out.reset_index(drop=True)


# ============ 预设模板 ============

# 所有可用模板的元数据(便于 CLI / 文档)
SCREEN_TEMPLATES: dict[str, str] = {
    "low_pe": "PE_TTM < 20",
    "high_pb": "PB < 1",
    "macd_golden": "MACD 金叉",
    "kdj_oversold": "KDJ 超卖",
    "rsi_oversold": "RSI < 30",
    "volume_breakout": "量能突破(量 ≥ 20 日均量 × 2)",
    "new_high_20": "20 日新高",
    "low_pe_high_turnover": "低 PE + 高换手率",
    "bullish_trend": "多头排列(MA5>MA10>MA20>MA60)",
    "pattern_bullish_reversal": "看涨形态共振(吞没 / 锤头 / 早晨之星 等 ≥ 2 个)",
    "pattern_morning_star": "早晨之星形态",
    "pattern_engulfing": "看涨吞没形态",
    "momentum_value": "动量(20 日 > 5%) + 低估值(PE<30)",
    "high_turnover_breakout": "高换手 + 量能放大",
    "rsi_neutral": "RSI 中性区(40-60)",
    "bbi_bullish": "多空布林线 BBI 上穿(MA3+6+12+24)/4",
}


def screen(template: str, date: Optional[str] = None, limit: int = 30, **kwargs) -> pd.DataFrame:
    """内置模板筛选。模板清单见 SCREEN_TEMPLATES。"""
    date = date or datetime.now().strftime("%Y-%m-%d")
    s = Screen(date=date, name=template, lookback_days=kwargs.get("lookback", 120))
    if template == "low_pe":
        s.add(pe_between(0, 20))
    elif template == "high_pb":
        s.add(pb_between(0, 1))
    elif template == "macd_golden":
        s.add(macd_golden_cross)
    elif template == "kdj_oversold":
        s.add(kdj_oversold)
    elif template == "rsi_oversold":
        s.add(rsi_oversold)
    elif template == "rsi_neutral":
        s.add(lambda df: ((ind.RSI(df["close"], 14) > 40) & (ind.RSI(df["close"], 14) < 60)))
    elif template == "volume_breakout":
        s.add(volume_breakout)
    elif template == "new_high_20":
        s.add(new_high_n)
    elif template == "low_pe_high_turnover":
        s.add(pe_between(0, 25)).add(pct_change_between(0.02, 0.20))
    elif template == "high_turnover_breakout":
        s.add(turnover_high(0.8)).add(amount_expand(1.5))
    elif template == "bullish_trend":
        s.add(MA5_gt_MA20).add(MA20_gt_MA60)
    elif template == "momentum_value":
        s.add(pe_between(0, 30)).add(lambda df: df["close"].pct_change(20) > 0.05)
    elif template == "pattern_bullish_reversal":
        s.add(pattern_bullish_score(2))
    elif template == "pattern_morning_star":
        s.add(pattern("morning_star"))
    elif template == "pattern_engulfing":
        s.add(pattern("engulfing_bullish"))
    elif template == "bbi_bullish":
        bbi = (ind.MA(df["close"], 3) + ind.MA(df["close"], 6) + ind.MA(df["close"], 12) + ind.MA(df["close"], 24)) / 4
        s.add(lambda d, bbi=bbi: (d["close"] > bbi) & (d["close"].shift(1) <= bbi.shift(1)))
    else:
        raise ValueError(f"未知模板: {template};可选: {list(SCREEN_TEMPLATES)}")
    return s.run(limit=limit, show_progress=kwargs.get("progress", True))


# ============ 简易每日选股流水线 ============

def daily_pick(date: Optional[str] = None, top: int = 20) -> pd.DataFrame:
    """结合趋势 + 量能 + 估值的复合选股"""
    date = date or datetime.now().strftime("%Y-%m-%d")
    s = Screen(date=date, name="daily_combo", lookback_days=180)
    s.add(MA5_gt_MA20)
    s.add(MA20_gt_MA60)
    s.add(macd_golden_cross)
    s.add(pe_between(0, 50))
    picks = s.run(limit=top, show_progress=True)
    if not picks.empty:
        picks = picks.sort_values("pctChg", ascending=False).head(top).reset_index(drop=True)
    return picks
