"""量化分析:因子 IC、分层回测、收益归因、相关性分析等。

主要工具:
    - factor_ic(factor_df, ret_df): 计算因子与下期收益的 IC (Pearson/Spearman)
    - layered_backtest(factor_df, ret_df, q=5): 因子分层回测
    - performance_attribution: 收益归因(Brinson 简化版)
    - correlation_matrix(prices): 收益率相关性矩阵
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def factor_ic(factor: pd.DataFrame, forward_ret: pd.DataFrame,
              method: str = "spearman") -> pd.DataFrame:
    """逐期计算因子与下期收益的 IC。

    factor / forward_ret: 列=股票, 索引=日期
    method: 'pearson' / 'spearman'

    返回: DataFrame(index=date, columns=[IC, abs_IC, pvalue])
    """
    if method == "spearman":
        corr_fn = lambda x, y: stats.spearmanr(x, y, nan_policy="omit")
    else:
        corr_fn = lambda x, y: stats.pearsonr(x, y)

    dates = factor.index.intersection(forward_ret.index)
    rows = []
    for d in dates:
        f = factor.loc[d].dropna()
        r = forward_ret.loc[d].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 5:
            continue
        try:
            if method == "spearman":
                res = corr_fn(f[common], r[common])
                ic, p = res.statistic, res.pvalue
            else:
                ic, p = corr_fn(f[common], r[common])
        except Exception:
            continue
        rows.append({"date": d, "IC": ic, "abs_IC": abs(ic), "pvalue": p})
    df = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["IC", "abs_IC", "pvalue"])
    return df


def ic_summary(ic: pd.DataFrame) -> pd.Series:
    if ic.empty:
        return pd.Series(dtype=float)
    return pd.Series({
        "IC_mean": ic["IC"].mean(),
        "IC_std": ic["IC"].std(),
        "IC_IR": ic["IC"].mean() / ic["IC"].std() if ic["IC"].std() > 0 else 0,
        "abs_IC_mean": ic["abs_IC"].mean(),
        "IC>0比例": (ic["IC"] > 0).mean(),
        "IC<-0.02 比例": (ic["IC"] < -0.02).mean(),
        "期数": len(ic),
    })


def layered_backtest(factor: pd.DataFrame, forward_ret: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """按因子值分 q 层,计算每层(多头=第 q 层,空头=第 1 层)的累计收益。

    输入: 因子截面值 + 下一期收益
    输出: DataFrame(q, [期均收益, 累计收益, 胜率, 换手])
    """
    common_dates = factor.index.intersection(forward_ret.index)
    if len(common_dates) == 0:
        return pd.DataFrame()

    layer_rets = {i: [] for i in range(1, q + 1)}
    for d in common_dates:
        f = factor.loc[d].dropna()
        r = forward_ret.loc[d].dropna()
        common = f.index.intersection(r.index)
        if len(common) < q * 3:
            continue
        ranks = pd.qcut(f[common], q=q, labels=False, duplicates="drop") + 1
        for k in ranks.unique():
            layer_rets[int(k)].append(r[common[ranks == k]].mean())

    rows = []
    for k in range(1, q + 1):
        rets = pd.Series(layer_rets[k])
        if rets.empty:
            continue
        rows.append({
            "分层": f"Q{k}",
            "期均收益": rets.mean(),
            "累计收益": (1 + rets).prod() - 1,
            "胜率": (rets > 0).mean(),
            "期数": len(rets),
        })
    df = pd.DataFrame(rows).set_index("分层")
    if not df.empty:
        long_ret = df.loc["Q5", "累计收益"] if "Q5" in df.index else 0
        short_ret = df.loc["Q1", "累计收益"] if "Q1" in df.index else 0
        df.attrs["多空差"] = long_ret - short_ret
    return df


def correlation_matrix(prices: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """对收益率计算相关性矩阵"""
    rets = prices.pct_change().dropna(how="all")
    return rets.corr(method=method)


def rolling_beta(y: pd.Series, x: pd.Series, window: int = 60) -> pd.Series:
    """y 对 x 的滚动 beta"""
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var.replace(0, np.nan)


def annualized_return(returns_: pd.Series, periods: int = 252) -> float:
    if returns_.empty:
        return 0.0
    total = (1 + returns_).prod() - 1
    years = len(returns_) / periods
    return (1 + total) ** (1 / years) - 1 if years > 0 else 0.0


def annualized_vol(returns_: pd.Series, periods: int = 252) -> float:
    if returns_.empty:
        return 0.0
    return returns_.std() * math.sqrt(periods)


def sharpe_ratio(returns_: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    if returns_.empty or returns_.std() == 0:
        return 0.0
    excess = returns_ - rf / periods
    return excess.mean() / excess.std() * math.sqrt(periods)


def max_drawdown(equity: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """返回最大回撤 (起点, 终点, 回撤幅度)"""
    if equity.empty:
        return None, None, 0.0
    peak = equity.cummax()
    dd = equity / peak - 1
    end = dd.idxmin()
    start = equity[:end].idxmax() if end is not None else None
    return start, end, float(dd.min())


def rolling_corr(x: pd.Series, y: pd.Series, window: int = 60) -> pd.Series:
    return x.rolling(window).corr(y)


def build_factor_table(prices: pd.DataFrame) -> pd.DataFrame:
    """从价格矩阵构造常见横截面因子(动量、反转、波动、换手等)。

    prices: 列=股票, 索引=日期
    返回: dict-like 因子 DataFrame {factor_name: DataFrame(date x code)}
    """
    rets = prices.pct_change()
    factors = {}
    factors["mom_20"] = prices.pct_change(20)
    factors["mom_5"] = prices.pct_change(5)
    factors["vol_20"] = rets.rolling(20).std()
    factors["vol_ratio"] = rets.rolling(5).std() / rets.rolling(20).std()
    # 价格分位
    factors["price_pos_60"] = prices.rolling(60).apply(lambda x: x.rank(pct=True).iloc[-1] if len(x) == 60 else np.nan, raw=False)
    return factors


# ============ 风险与归因 ============

def downside_deviation(returns_: pd.Series, mar: float = 0.0, periods: int = 252) -> float:
    """下行偏差(年化)。"""
    if returns_.empty:
        return 0.0
    diff = (mar / periods) - returns_
    dd = diff[diff > 0]
    if dd.empty:
        return 0.0
    return float(np.sqrt((dd ** 2).mean()) * math.sqrt(periods))


def sortino(returns_: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    """Sortino 比率。"""
    if returns_.empty:
        return 0.0
    dd = downside_deviation(returns_, rf, periods)
    if dd == 0:
        return 0.0
    excess = returns_.mean() * periods - rf
    return float(excess / dd)


def calmar(returns_: pd.Series, periods: int = 252) -> float:
    """Calmar 比率 = 年化收益 / 最大回撤。"""
    if returns_.empty:
        return 0.0
    eq = (1 + returns_).cumprod()
    _, _, mdd = max_drawdown(eq)
    if mdd == 0:
        return 0.0
    return float(annualized_return(returns_, periods) / abs(mdd))


def value_at_risk(returns_: pd.Series, p: float = 0.05) -> float:
    """历史 VaR(损失为正)。"""
    if returns_.empty:
        return 0.0
    return float(-returns_.quantile(p))


def conditional_var(returns_: pd.Series, p: float = 0.05) -> float:
    """CVaR / Expected Shortfall(尾部均值)。"""
    if returns_.empty:
        return 0.0
    cutoff = returns_.quantile(p)
    return float(-returns_[returns_ <= cutoff].mean())


def performance_attribution(returns: pd.DataFrame, weights: pd.DataFrame,
                            factor_returns: pd.DataFrame) -> pd.DataFrame:
    """简化 Brinson 归因:对每期把组合收益分解到各因子暴露(单因子回归)。

    returns:       列=资产,行=日期(各资产日收益)
    weights:       列=资产,行=日期(组合权重,每日权重和=1)
    factor_returns:列=因子,行=日期

    返回: DataFrame(index=factor, columns=['beta','contribution','ann_contribution'])
    """
    port_ret = (weights.shift(1).fillna(0) * returns).sum(axis=1)
    common = port_ret.index.intersection(factor_returns.index)
    pr = port_ret.loc[common]
    fr = factor_returns.loc[common]
    if pr.empty or fr.empty:
        return pd.DataFrame()
    rows = []
    for f in fr.columns:
        cov = pr.cov(fr[f])
        var = fr[f].var()
        beta = cov / var if var > 0 else 0.0
        contribution = beta * fr[f].mean() * 252
        rows.append({"factor": f, "beta": beta, "ann_contribution": contribution})
    df = pd.DataFrame(rows).set_index("factor")
    if not df.empty:
        df.attrs["portfolio_ann_return"] = pr.mean() * 252
    return df


def regime_detection(returns_: pd.Series, n: int = 20) -> pd.Series:
    """简易市场状态识别:HMM 替代实现。

    规则:
        bull:   近 n 日收益 > +vol  + 20 日均线斜率 > 0
        bear:   近 n 日收益 < -vol  + 斜率 < 0
        side:   其余
    """
    if returns_.empty:
        return pd.Series(dtype=object)
    vol = returns_.rolling(n, min_periods=n // 2).std() * math.sqrt(252)
    cum_ret = (1 + returns_).cumprod()
    sma = cum_ret.rolling(n, min_periods=n // 2).mean()
    slope = sma.diff(n)  # 价差近似斜率
    out = pd.Series("side", index=returns_.index, dtype=object)
    out[(cum_ret > sma) & (slope > 0) & (returns_.rolling(n).mean() > vol / math.sqrt(252))] = "bull"
    out[(cum_ret < sma) & (slope < 0) & (returns_.rolling(n).mean() < -vol / math.sqrt(252))] = "bear"
    return out


def rolling_factor_returns(factor_df: pd.DataFrame, n: int = 60) -> pd.DataFrame:
    """滚动 N 日因子收益(横截面多空:Q5-Q1 收益序列)。"""
    out = {}
    for d in factor_df.index:
        f = factor_df.loc[d].dropna()
        if len(f) < 10:
            continue
        try:
            q1, q5 = f.quantile([0.2, 0.8])
            out[d] = q5 - q1
        except Exception:
            continue
    return pd.Series(out).to_frame("多空差")
