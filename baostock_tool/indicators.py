"""技术指标库:纯 pandas/numpy 实现,输入 DataFrame (close/high/low/volume 列),返回带指标列的 DataFrame。

每个函数都做 in-place 返回新 DataFrame,不修改原表。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _check(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"输入 DataFrame 缺少列: {missing}")


# ============ 移动平均线 ============

def MA(series: pd.Series, n: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=n, min_periods=1).mean()


def EMA(series: pd.Series, n: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=n, adjust=False, min_periods=1).mean()


def SMA(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """中国式 SMA: 今日 SMA = ( (n-1)*昨日 SMA + 2*今日收盘) / (n+1) ; 等价于 EMA(n*2-1)"""
    return series.ewm(alpha=2 / (n * 2 - 1), adjust=False, min_periods=1).mean()


def add_ma(df: pd.DataFrame, periods: tuple[int, ...] = (5, 10, 20, 30, 60, 120, 250)) -> pd.DataFrame:
    """加多周期 MA 列"""
    _check(df, "close")
    out = df.copy()
    for p in periods:
        out[f"MA{p}"] = MA(out["close"], p)
    return out


# ============ MACD ============

def MACD(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """返回 (DIF, DEA, HIST)"""
    ema_fast = EMA(close, fast)
    ema_slow = EMA(close, slow)
    dif = ema_fast - ema_slow
    dea = EMA(dif, signal)
    hist = (dif - dea) * 2
    return dif, dea, hist


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    _check(df, "close")
    out = df.copy()
    dif, dea, hist = MACD(out["close"], fast, slow, signal)
    out["MACD_DIF"] = dif
    out["MACD_DEA"] = dea
    out["MACD_HIST"] = hist
    return out


# ============ KDJ ============

def KDJ(df: pd.DataFrame, n: int = 9, k_period: int = 3, d_period: int = 3
        ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """KDJ 指标: 传入含 high/low/close 的 DataFrame"""
    _check(df, "high", "low", "close")
    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_n = df["high"].rolling(window=n, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(alpha=1 / k_period, adjust=False).mean()
    d = k.ewm(alpha=1 / d_period, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def add_kdj(df: pd.DataFrame, n: int = 9, k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
    out = df.copy()
    k, d, j = KDJ(out, n, k_period, d_period)
    out["KDJ_K"] = k
    out["KDJ_D"] = d
    out["KDJ_J"] = j
    return out


# ============ RSI ============

def RSI(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI"""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder 平滑
    roll_up = up.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)


def add_rsi(df: pd.DataFrame, periods: tuple[int, ...] = (6, 12, 14, 24)) -> pd.DataFrame:
    _check(df, "close")
    out = df.copy()
    for p in periods:
        out[f"RSI{p}"] = RSI(out["close"], p)
    return out


# ============ BOLL ============

def BOLL(close: pd.Series, n: int = 20, k: float = 2.0
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = MA(close, n)
    std = close.rolling(window=n, min_periods=1).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower


def add_boll(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    _check(df, "close")
    out = df.copy()
    mid, up, lo = BOLL(out["close"], n, k)
    out["BOLL_MID"] = mid
    out["BOLL_UP"] = up
    out["BOLL_LOW"] = lo
    out["BOLL_WIDTH"] = (up - lo) / mid
    out["BOLL_PCT"] = (out["close"] - lo) / (up - lo).replace(0, np.nan)
    return out


# ============ ATR ============

def ATR(df: pd.DataFrame, n: int = 14) -> pd.Series:
    _check(df, "high", "low", "close")
    high = df["high"]
    low = df["low"]
    pre_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - pre_close).abs(),
        (low - pre_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()


def add_atr(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    out = df.copy()
    out[f"ATR{n}"] = ATR(out, n)
    return out


# ============ CCI ============

def CCI(df: pd.DataFrame, n: int = 20) -> pd.Series:
    _check(df, "high", "low", "close")
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma_tp = tp.rolling(window=n, min_periods=1).mean()
    md = tp.rolling(window=n, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - ma_tp) / (0.015 * md.replace(0, np.nan))
    return cci


def add_cci(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["CCI"] = CCI(out, n)
    return out


# ============ OBV ============

def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    obv = (volume * direction).cumsum()
    return obv


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    _check(df, "close", "volume")
    out = df.copy()
    out["OBV"] = OBV(out["close"], out["volume"])
    out["OBV_MA"] = MA(out["OBV"], 20)
    return out


# ============ 收益与波动 ============

def returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """简单收益率"""
    return close.pct_change(periods=periods)


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def volatility(returns_: pd.Series, n: int = 20, annual: int = 252) -> pd.Series:
    """滚动年化波动率"""
    return returns_.rolling(window=n, min_periods=1).std() * np.sqrt(annual)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    _check(df, "close")
    out = df.copy()
    out["ret_1"] = returns(out["close"], 1)
    out["ret_5"] = returns(out["close"], 5)
    out["ret_20"] = returns(out["close"], 20)
    out["log_ret"] = log_returns(out["close"])
    out["vol_20"] = volatility(out["ret_1"], 20)
    return out


# ============ DMI / ADX ============

def DMI(df: pd.DataFrame, n: int = 14
        ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """趋向指标 DMI: 返回 (PDI, MDI, ADX, ADXR)。"""
    _check(df, "high", "low", "close")
    high = df["high"]
    low = df["low"]
    close = df["close"]
    pre_close = close.shift(1)
    up_move = high - high.shift(1)
    dn_move = low.shift(1) - low
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr = pd.concat([
        high - low,
        (high - pre_close).abs(),
        (low - pre_close).abs(),
    ], axis=1).max(axis=1)
    atr_n = tr.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    pdi = 100 * plus_dm_s / atr_n.replace(0, np.nan)
    mdi = 100 * minus_dm_s / atr_n.replace(0, np.nan)
    dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    adxr = (adx + adx.shift(n)) / 2
    return pdi.fillna(0), mdi.fillna(0), adx.fillna(0), adxr.fillna(0)


def add_dmi(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    out = df.copy()
    pdi, mdi, adx, adxr = DMI(out, n)
    out["PDI"] = pdi
    out["MDI"] = mdi
    out["ADX"] = adx
    out["ADXR"] = adxr
    return out


# ============ Williams %R ============

def WILLR(df: pd.DataFrame, n: int = 14) -> pd.Series:
    _check(df, "high", "low", "close")
    high_n = df["high"].rolling(n, min_periods=1).max()
    low_n = df["low"].rolling(n, min_periods=1).min()
    return (high_n - df["close"]) / (high_n - low_n).replace(0, np.nan) * -100


def add_willr(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    out = df.copy()
    out[f"WR{n}"] = WILLR(out, n)
    return out


# ============ ROC ============

def ROC(close: pd.Series, n: int = 12) -> pd.Series:
    return close.pct_change(n) * 100


def add_roc(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    out = df.copy()
    out[f"ROC{n}"] = ROC(out["close"], n)
    return out


# ============ MFI ============

def MFI(df: pd.DataFrame, n: int = 14) -> pd.Series:
    _check(df, "high", "low", "close", "volume")
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw_money = tp * df["volume"]
    direction = np.sign(tp.diff().fillna(0))
    pos_money = (raw_money * (direction > 0)).rolling(n, min_periods=1).sum()
    neg_money = (raw_money * (direction < 0)).rolling(n, min_periods=1).sum()
    mf = 100 - 100 / (1 + pos_money / neg_money.replace(0, np.nan))
    return mf.fillna(50)


def add_mfi(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    out = df.copy()
    out[f"MFI{n}"] = MFI(out, n)
    return out


# ============ TRIX ============

def TRIX(close: pd.Series, n: int = 12) -> pd.Series:
    """三重指数平滑 = TR 与其 1 周期变化的百分率。"""
    tr = close.ewm(span=n, adjust=False, min_periods=1).mean()
    return tr.pct_change() * 100


def add_trix(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    out = df.copy()
    out[f"TRIX{n}"] = TRIX(out["close"], n)
    return out


# ============ SAR(抛物线) ============

def SAR(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02,
        af_max: float = 0.2) -> pd.Series:
    _check(df, "high", "low")
    high = df["high"].values
    low = df["low"].values
    n = len(high)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=df.index)
    bull = True  # 当前趋势方向
    af = af_start
    ep = high[0] if bull else low[0]
    sar[0] = ep
    for i in range(1, n):
        prev_sar = sar[i - 1]
        if bull:
            sar[i] = prev_sar + af * (ep - prev_sar)
            # 不能高于前两根低点
            if i >= 2:
                sar[i] = min(sar[i], low[i - 1], low[i - 2])
            else:
                sar[i] = min(sar[i], low[i - 1])
            if low[i] < sar[i]:
                # 反转:多 -> 空
                bull = False
                sar[i] = ep  # 反转时 SAR 设为前 EP
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            if i >= 2:
                sar[i] = max(sar[i], high[i - 1], high[i - 2])
            else:
                sar[i] = max(sar[i], high[i - 1])
            if high[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
    return pd.Series(sar, index=df.index)


def add_sar(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    out = df.copy()
    out["SAR"] = SAR(out, **kwargs)
    return out


# ============ BIAS 乖离率 ============

def BIAS(close: pd.Series, n: int = 6) -> pd.Series:
    return (close - MA(close, n)) / MA(close, n).replace(0, np.nan) * 100


def add_bias(df: pd.DataFrame, periods: tuple[int, ...] = (6, 12, 24)) -> pd.DataFrame:
    out = df.copy()
    for p in periods:
        out[f"BIAS{p}"] = BIAS(out["close"], p)
    return out


# ============ 全部指标一次性加 ============

# add_all 支持的子模块列表
ALL_INDICATORS = ("ma", "macd", "kdj", "rsi", "boll", "atr", "cci", "obv", "returns",
                  "dmi", "willr", "roc", "mfi", "trix", "sar", "bias")


def add_all(df: pd.DataFrame, include: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
    """加全套常用指标。include=None 取全部;支持子集以加速。"""
    items = include or ALL_INDICATORS
    out = df.copy()
    if "ma" in items:
        out = add_ma(out)
    if "macd" in items:
        out = add_macd(out)
    if "kdj" in items:
        out = add_kdj(out)
    if "rsi" in items:
        out = add_rsi(out)
    if "boll" in items:
        out = add_boll(out)
    if "atr" in items:
        out = add_atr(out)
    if "cci" in items:
        out = add_cci(out)
    if "obv" in items:
        out = add_obv(out)
    if "returns" in items:
        out = add_returns(out)
    if "dmi" in items:
        out = add_dmi(out)
    if "willr" in items:
        out = add_willr(out)
    if "roc" in items:
        out = add_roc(out)
    if "mfi" in items:
        out = add_mfi(out)
    if "trix" in items:
        out = add_trix(out)
    if "sar" in items:
        out = add_sar(out)
    if "bias" in items:
        out = add_bias(out)
    return out


# ============ 信号识别工具 ============

def golden_cross(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
    """金叉: 短均线上穿长均线,返回 True 当天"""
    return (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))


def death_cross(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
    """死叉: 短均线下穿长均线"""
    return (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))


def breakout(series: pd.Series, n: int) -> pd.Series:
    """N 日新高"""
    return series >= series.rolling(window=n, min_periods=1).max()
