"""交易策略库:输入 K 线 DataFrame,返回 signal DataFrame。

信号约定:
    position: 目标仓位 1=满仓, 0=空仓
    signal:   当日事件 +1 买入 / -1 卖出 / 0 无
每只策略都应只依赖 df 列,不隐式访问外部数据。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from . import indicators as ind


StrategyFunc = Callable[[pd.DataFrame, dict], pd.DataFrame]


@dataclass
class Strategy:
    """策略注册容器,用于回测与扫描。"""
    name: str
    description: str
    func: StrategyFunc
    default_params: dict


# ============ 策略实现 ============

def _empty_signal(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "signal": 0,
        "position": 0,
        "note": "",
    }, index=df.index)


def strategy_ma_cross(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """双均线交叉: 短均上穿长均买入,下穿卖出"""
    short = int(params.get("short", 5))
    long_ = int(params.get("long", 20))
    sig = _empty_signal(df)
    ma_s = ind.MA(df["close"], short)
    ma_l = ind.MA(df["close"], long_)
    sig["signal"] = np.where(ind.golden_cross(ma_s, ma_l), 1,
                              np.where(ind.death_cross(ma_s, ma_l), -1, 0))
    sig["position"] = (ma_s > ma_l).astype(int)
    return sig


def strategy_macd(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """MACD 柱状由负转正买入,反之卖出"""
    sig = _empty_signal(df)
    _, _, hist = ind.MACD(df["close"])
    sig["signal"] = np.where(
        (hist > 0) & (hist.shift(1) <= 0), 1,
        np.where((hist < 0) & (hist.shift(1) >= 0), -1, 0),
    )
    sig["position"] = (hist > 0).astype(int)
    return sig


def strategy_kdj(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """KDJ: K 上穿 D 且 J<20 买入; K 下穿 D 且 J>80 卖出"""
    sig = _empty_signal(df)
    k, d, j = ind.KDJ(df)
    cross_up = (k > d) & (k.shift(1) <= d.shift(1)) & (j < 30)
    cross_dn = (k < d) & (k.shift(1) >= d.shift(1)) & (j > 80)
    sig["signal"] = np.where(cross_up, 1, np.where(cross_dn, -1, 0))
    sig["position"] = ((k > d) & (j < 80)).astype(int)
    return sig


def strategy_boll_breakout(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """BOLL 突破: 收盘上穿上轨买入,下穿中轨卖出"""
    sig = _empty_signal(df)
    mid, up, lo = ind.BOLL(df["close"])
    sig["signal"] = np.where(
        (df["close"] > up) & (df["close"].shift(1) <= up.shift(1)), 1,
        np.where((df["close"] < mid) & (df["close"].shift(1) >= mid.shift(1)), -1, 0),
    )
    sig["position"] = (df["close"] > mid).astype(int)
    return sig


def strategy_turtle(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """海龟交易: 20 日新高买入,10 日新低卖出"""
    entry = int(params.get("entry", 20))
    exit_ = int(params.get("exit", 10))
    sig = _empty_signal(df)
    hi = df["close"].rolling(entry, min_periods=1).max()
    lo = df["close"].rolling(exit_, min_periods=1).min()
    sig["signal"] = np.where(
        (df["close"] >= hi) & (df["close"].shift(1) < hi.shift(1)), 1,
        np.where((df["close"] <= lo) & (df["close"].shift(1) > lo.shift(1)), -1, 0),
    )
    sig["position"] = (df["close"] >= hi.shift(1)).astype(int)
    return sig


def strategy_momentum(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """动量策略: N 日涨幅 > threshold 买入,<-threshold 卖出"""
    n = int(params.get("n", 20))
    thr = float(params.get("threshold", 0.05))
    ret = df["close"].pct_change(n)
    sig = _empty_signal(df)
    sig["signal"] = np.where(ret > thr, 1, np.where(ret < -thr, -1, 0))
    sig["position"] = (ret > 0).astype(int)
    return sig


def strategy_mean_reversion(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """均值回归: 偏离 MA20 超过 k 个 std 买入回归,跌破 MA20 卖出"""
    n = int(params.get("n", 20))
    k = float(params.get("k", 2.0))
    ma = ind.MA(df["close"], n)
    std = df["close"].rolling(n, min_periods=1).std(ddof=0)
    z = (df["close"] - ma) / std.replace(0, np.nan)
    sig = _empty_signal(df)
    sig["signal"] = np.where(z < -k, 1, np.where(z > 0, -1, 0))
    sig["position"] = (z < 0).astype(int)
    return sig


def strategy_rsi_oversold(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """RSI 超卖反弹: RSI<30 买入,RSI>70 卖出"""
    n = int(params.get("n", 14))
    buy = int(params.get("buy", 30))
    sell = int(params.get("sell", 70))
    rsi = ind.RSI(df["close"], n)
    sig = _empty_signal(df)
    sig["signal"] = np.where(
        (rsi < buy) & (rsi.shift(1) >= buy), 1,
        np.where((rsi > sell) & (rsi.shift(1) <= sell), -1, 0),
    )
    sig["position"] = (rsi < 50).astype(int)
    return sig


# ============ 注册表 ============

STRATEGIES: dict[str, Strategy] = {
    "ma_cross": Strategy(
        name="ma_cross",
        description="双均线交叉(默认 5/20)",
        func=strategy_ma_cross,
        default_params={"short": 5, "long": 20},
    ),
    "macd": Strategy(
        name="macd",
        description="MACD 柱状拐点",
        func=strategy_macd,
        default_params={},
    ),
    "kdj": Strategy(
        name="kdj",
        description="KDJ 极值反转",
        func=strategy_kdj,
        default_params={},
    ),
    "boll": Strategy(
        name="boll",
        description="BOLL 通道突破",
        func=strategy_boll_breakout,
        default_params={},
    ),
    "turtle": Strategy(
        name="turtle",
        description="海龟交易(20日新高/10日新低)",
        func=strategy_turtle,
        default_params={"entry": 20, "exit": 10},
    ),
    "momentum": Strategy(
        name="momentum",
        description="N 日动量",
        func=strategy_momentum,
        default_params={"n": 20, "threshold": 0.05},
    ),
    "mean_reversion": Strategy(
        name="mean_reversion",
        description="均值回归(z-score)",
        func=strategy_mean_reversion,
        default_params={"n": 20, "k": 2.0},
    ),
    "rsi_oversold": Strategy(
        name="rsi_oversold",
        description="RSI 超卖反弹",
        func=strategy_rsi_oversold,
        default_params={"n": 14, "buy": 30, "sell": 70},
    ),
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"未知策略: {name},可选: {list(STRATEGIES)}")
    return STRATEGIES[name]


def run_strategy(name: str, df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """运行指定策略,返回包含 signal/position 的 DataFrame"""
    strat = get_strategy(name)
    p = {**strat.default_params, **(params or {})}
    sig = strat.func(df, p)
    sig.attrs["strategy"] = name
    sig.attrs["params"] = p
    return sig


# ============ 多策略融合 (Ensemble) ============

from typing import Sequence   # noqa: E402


class EnsembleStrategy:
    """把多套策略的信号融合成 1 套,降低单策略过拟合风险。

    voting:
        weighted — 加权求和(sum(s_i * w_i)),过 entry_threshold 触发买入
        majority — 多数投票:N 套策略里 ≥ majority_min 套给 1 才开仓;
                   0 是平仓;-majority_min 套给 -1 才开空(对称)
        veto     — 多数投票 + 任何 1 套给反向就否决(更保守)
    """

    def __init__(self,
                 strategies: Sequence[Strategy | str],
                 weights: Optional[Sequence[float]] = None,
                 voting: Literal["weighted", "majority", "veto"] = "weighted",
                 entry_threshold: float = 0.3,
                 exit_threshold: float = 0.0,
                 majority_min: int = 2):
        # 标准化 strategies → list[Strategy]
        norm: list[Strategy] = []
        for s in strategies:
            if isinstance(s, str):
                norm.append(get_strategy(s))
            else:
                norm.append(s)
        if not norm:
            raise ValueError("strategies 不能为空")
        self.strategies = norm
        self.n = len(norm)
        self.weights = (list(weights) if weights else [1.0 / self.n] * self.n)
        if len(self.weights) != self.n:
            raise ValueError("weights 长度必须等于 strategies 数量")
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-6):
            # 归一化权重
            s = sum(self.weights)
            self.weights = [w / s for w in self.weights]
        self.voting = voting
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.majority_min = majority_min

    def _aggregate(self, sigs: list[pd.DataFrame]) -> pd.DataFrame:
        """把多套策略的 signal/position 聚合成一份。"""
        ref = sigs[0]
        out = pd.DataFrame(index=ref.index)
        if self.voting == "weighted":
            # 权重化的 signal 分数
            score = sum(sigs[i]["signal"] * w for i, w in enumerate(self.weights))
            out["score"] = score
            out["position"] = (score > self.entry_threshold).astype(int)
            out["signal"] = np.where(
                out["position"].diff() == 1, 1,
                np.where(out["position"].diff() == -1, -1, 0),
            )
        else:
            # majority / veto
            signals = np.vstack([s["signal"].values for s in sigs])
            n_long = (signals == 1).sum(axis=0)
            n_short = (signals == -1).sum(axis=0)
            n_flat = (signals == 0).sum(axis=0)
            if self.voting == "majority":
                out["position"] = np.where(
                    n_long >= self.majority_min, 1,
                    np.where(n_short >= self.majority_min, -1, 0),
                )
            else:  # veto: 必须多数给同向,任何反向就 flat
                out["position"] = np.where(
                    (n_long >= self.majority_min) & (n_short == 0), 1,
                    np.where((n_short >= self.majority_min) & (n_long == 0), -1, 0),
                )
            out["position"] = pd.Series(out["position"], index=ref.index)
            out["signal"] = np.where(
                out["position"].diff() == 1, 1,
                np.where(out["position"].diff() == -1, -1, 0),
            )
            out["score"] = (n_long - n_short) / self.n
        out["note"] = "ensemble:" + "+".join(s.name for s in self.strategies)
        return out

    def run(self, df: pd.DataFrame,
            params_list: Optional[Sequence[dict]] = None) -> pd.DataFrame:
        """跑每套子策略并融合。params_list 可为每套策略指定不同 params。"""
        sigs = []
        params_list = params_list or [None] * self.n
        for s, p in zip(self.strategies, params_list):
            params = {**s.default_params, **(p or {})}
            sigs.append(s.func(df, params))
        return self._aggregate(sigs)
