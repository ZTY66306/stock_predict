"""K 线形态识别:纯函数式,返回 pd.Series[bool]。

约定:
    所有函数接受 DataFrame(必须含 open/high/low/close),返回与 df.index 对齐的 bool 序列。
    命中当根 bar(确认日 / 反转日),不包含前导 bar;不画未来函数。
    阈值参数化,容差内可调。
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _check(df: pd.DataFrame) -> None:
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}")


def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]


def _is_bullish(df: pd.DataFrame) -> pd.Series:
    return df["close"] > df["open"]


def _is_bearish(df: pd.DataFrame) -> pd.Series:
    return df["close"] < df["open"]


# ============ 单根形态 ============

def doji(df: pd.DataFrame, tol: float = 0.001) -> pd.Series:
    """十字星:实体长度 ≤ 振幅的 tol。"""
    _check(df)
    rng = df["high"] - df["low"]
    body = _body(df)
    return (body <= rng * tol) & (rng > 0)


def hammer(df: pd.DataFrame, body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> pd.Series:
    """锤头:小实体,下影线 ≥ 实体 × shadow_ratio,上影线极短。
    出现在下降趋势末端视为看涨;在上涨末端视为上吊线(hanging_man)。"""
    _check(df)
    body = _body(df)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    ls = _lower_shadow(df)
    us = _upper_shadow(df)
    return (body <= rng * body_ratio) & (ls >= body * shadow_ratio) & (us <= body * 0.5)


def hanging_man(df: pd.DataFrame, body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> pd.Series:
    """上吊线:与锤头同形,需结合趋势位置判断。"""
    return hammer(df, body_ratio=body_ratio, shadow_ratio=shadow_ratio)


def marubozu_bullish(df: pd.DataFrame, shadow_ratio: float = 0.02) -> pd.Series:
    """光头光脚阳线:几乎没有上下影线。"""
    _check(df)
    body = _body(df)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    ls = _lower_shadow(df)
    us = _upper_shadow(df)
    return _is_bullish(df) & (body >= rng * (1 - shadow_ratio)) & (ls <= body * shadow_ratio) & (us <= body * shadow_ratio)


def marubozu_bearish(df: pd.DataFrame, shadow_ratio: float = 0.02) -> pd.Series:
    """光头光脚阴线。"""
    _check(df)
    body = _body(df)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    ls = _lower_shadow(df)
    us = _upper_shadow(df)
    return _is_bearish(df) & (body >= rng * (1 - shadow_ratio)) & (ls <= body * shadow_ratio) & (us <= body * shadow_ratio)


# ============ 双根形态 ============

def engulfing_bullish(df: pd.DataFrame) -> pd.Series:
    """看涨吞没:前阴后阳,阳线实体完全包住阴线实体。"""
    _check(df)
    pre = df.shift(1)
    return (
        _is_bearish(pre) & _is_bullish(df)
        & (df["close"] > pre["open"])
        & (df["open"] < pre["close"])
    )


def engulfing_bearish(df: pd.DataFrame) -> pd.Series:
    """看跌吞没:前阳后阴,阴线实体完全包住阳线实体。"""
    _check(df)
    pre = df.shift(1)
    return (
        _is_bullish(pre) & _is_bearish(df)
        & (df["close"] < pre["open"])
        & (df["open"] > pre["close"])
    )


def piercing(df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """刺穿线:前阴后阳,阳线深入阴线实体 ≥ 一半。"""
    _check(df)
    pre = df.shift(1)
    pre_body = (pre["open"] - pre["close"]).abs()
    open_above = df["open"] < pre["low"]
    close_mid = df["close"] > (pre["open"] + pre["close"]) / 2
    close_below = df["close"] < pre["open"]
    return (
        _is_bearish(pre) & _is_bullish(df)
        & open_above & close_mid & close_below
    )


def dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    """乌云盖顶:前阳后阴,阴线深入阳线实体一半以上。"""
    _check(df)
    pre = df.shift(1)
    return (
        _is_bullish(pre) & _is_bearish(df)
        & (df["open"] > pre["high"])
        & (df["close"] < (pre["open"] + pre["close"]) / 2)
        & (df["close"] > pre["open"])
    )


# ============ 三根形态 ============

def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """红三兵:连续三根阳线,每根收盘价都创新高,且开盘在前一根实体内。"""
    _check(df)
    p1, p2 = df.shift(1), df.shift(2)
    return (
        _is_bullish(p2) & _is_bullish(p1) & _is_bullish(df)
        & (p1["close"] > p2["close"])
        & (df["close"] > p1["close"])
        & (p1["open"] > p2["open"]) & (p1["open"] < p2["close"])
        & (df["open"] > p1["open"]) & (df["open"] < p1["close"])
    )


def three_black_crows(df: pd.DataFrame) -> pd.Series:
    """三只乌鸦:连续三根阴线,每根收盘价都创新低。"""
    _check(df)
    p1, p2 = df.shift(1), df.shift(2)
    return (
        _is_bearish(p2) & _is_bearish(p1) & _is_bearish(df)
        & (p1["close"] < p2["close"])
        & (df["close"] < p1["close"])
        & (p1["open"] < p2["open"]) & (p1["open"] > p2["close"])
        & (df["open"] < p1["open"]) & (df["open"] > p1["close"])
    )


def morning_star(df: pd.DataFrame) -> pd.Series:
    """早晨之星:大阴 + 小实体(可阴可阳)+ 大阳。"""
    _check(df)
    p1, p2 = df.shift(1), df.shift(2)
    big_bear = _is_bearish(p2) & (_body(p2) > 0)
    small = _body(p1) < _body(p2) * 0.5
    big_bull = _is_bullish(df) & (_body(df) > _body(p2) * 0.5)
    return big_bear & small & big_bull & (df["close"] > (p2["open"] + p2["close"]) / 2)


def evening_star(df: pd.DataFrame) -> pd.Series:
    """黄昏之星:大阳 + 小实体 + 大阴。"""
    _check(df)
    p1, p2 = df.shift(1), df.shift(2)
    big_bull = _is_bullish(p2) & (_body(p2) > 0)
    small = _body(p1) < _body(p2) * 0.5
    big_bear = _is_bearish(df) & (_body(df) > _body(p2) * 0.5)
    return big_bull & small & big_bear & (df["close"] < (p2["open"] + p2["close"]) / 2)


# ============ 形态集合与打分 ============

BULLISH_PATTERNS = {
    "engulfing_bullish": engulfing_bullish,
    "hammer": hammer,
    "marubozu_bullish": marubozu_bullish,
    "piercing": piercing,
    "three_white_soldiers": three_white_soldiers,
    "morning_star": morning_star,
}

BEARISH_PATTERNS = {
    "engulfing_bearish": engulfing_bearish,
    "hanging_man": hanging_man,
    "marubozu_bearish": marubozu_bearish,
    "dark_cloud_cover": dark_cloud_cover,
    "three_black_crows": three_black_crows,
    "evening_star": evening_star,
}


def list_patterns() -> list[str]:
    """返回所有内置形态名。"""
    return sorted(BULLISH_PATTERNS.keys() | BEARISH_PATTERNS.keys() | {"doji"})


def detect(df: pd.DataFrame, name: str) -> pd.Series:
    """按名称调用单个形态。"""
    table = BULLISH_PATTERNS | BEARISH_PATTERNS | {"doji": doji}
    if name not in table:
        raise ValueError(f"未知形态: {name},可选: {list(table)}")
    return table[name](df)


def pattern_score(df: pd.DataFrame, names: Iterable[str] | None = None,
                  bullish: bool = True) -> pd.Series:
    """把多个看涨(或看跌)形态叠加打分,返回 -N~N 整数序列。"""
    names = names or list_patterns()
    score = pd.Series(0, index=df.index, dtype=int)
    for n in names:
        if n in BULLISH_PATTERNS and bullish:
            score += detect(df, n).astype(int)
        elif n in BEARISH_PATTERNS and not bullish:
            score -= detect(df, n).astype(int)
        elif n == "doji":
            # 十字星记 0,不强加方向
            continue
    return score
