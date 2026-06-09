"""涨跌停统计 / 题材热度(早盘扫描 + 复盘工具)。

提供:
    - daily_limit_up(date)             当日涨停股明细
    - daily_limit_down(date)           当日跌停股明细
    - consecutive_limit_up(date, n)    N 连板股(N≥2)
    - failed_limit_up(date)            炸板股(曾涨停但未封住)
    - limit_up_count_series(start,end) 每日涨停 / 跌停 / 炸板 数 趋势
    - sector_limit_up_count(date)      各行业涨停股数排行
    - concept_limit_up_count(date)     各概念涨停股数排行
    - market_sentiment(date)           市场情绪综合指标

依赖:akshare(可选)。未装时所有函数抛 ImportError。
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import akshare as ak
    _HAS_AKSHARE = True
except ImportError:  # pragma: no cover
    _HAS_AKSHARE = False


def _require_akshare():
    if not _HAS_AKSHARE:
        raise ImportError(
            "本模块依赖 akshare,运行 `pip install akshare` 后重试。"
        )


def _fmt_date(date) -> str:
    """'YYYY-MM-DD' / 'YYYYMMDD' / datetime → 'YYYYMMDD'。"""
    if isinstance(date, str):
        return date.replace("-", "")
    return date.strftime("%Y%m%d")


# ============ 涨停 / 跌停 / 炸板 ============

def daily_limit_up(date) -> pd.DataFrame:
    """当日涨停股池(东财源)。

    date: 'YYYY-MM-DD' 或 'YYYYMMDD'
    返回字段:代码 / 名称 / 涨跌幅 / 封板资金 / 首次封板时间 / 炸板次数 / 涨停统计等
    """
    _require_akshare()
    d = _fmt_date(date)
    df = ak.stock_zt_pool_em(date=d)
    return df.reset_index(drop=True) if df is not None else pd.DataFrame()


def daily_limit_down(date) -> pd.DataFrame:
    """当日跌停股池。"""
    _require_akshare()
    d = _fmt_date(date)
    df = ak.stock_zt_pool_dtgc_em(date=d)
    return df.reset_index(drop=True) if df is not None else pd.DataFrame()


def failed_limit_up(date) -> pd.DataFrame:
    """炸板股池(曾涨停但未封住的股票)。"""
    _require_akshare()
    d = _fmt_date(date)
    df = ak.stock_zt_pool_zbgc_em(date=d)
    return df.reset_index(drop=True) if df is not None else pd.DataFrame()


def consecutive_limit_up(date, n: int = 2) -> pd.DataFrame:
    """N 连板股(连续 ≥ n 个交易日涨停)。

    实现:对当日涨停股按 '连板数' 字段过滤;若字段名不固定则尽量匹配。
    """
    zt = daily_limit_up(date)
    if zt.empty:
        return zt
    # 连板数字段名可能为 '连板数' / '涨停统计' / '连续涨停天数'
    col = next((c for c in zt.columns
                if "连板" in c or "连续" in c and "涨停" in c), None)
    if col is None:
        col = next((c for c in zt.columns if "涨停" in c and ("次数" in c or "天数" in c)), None)
    if col is None:
        # 退而求其次:全部涨停股
        return zt
    out = zt[zt[col] >= n].reset_index(drop=True)
    return out


# ============ 趋势统计 ============

def limit_up_count_series(start_date, end_date) -> pd.DataFrame:
    """区间内每日涨停 / 跌停 / 炸板 数量趋势。

    返回 DataFrame,索引为日期,列 [limit_up, limit_down, failed_limit_up, fail_rate]
    """
    _require_akshare()
    s = _fmt_date(start_date)
    e = _fmt_date(end_date)
    df = ak.stock_zt_pool_zsdt_em(start_date=s, end_date=e)
    if df is None or df.empty:
        return pd.DataFrame()
    # 不同版本字段不同,统一化
    date_col = next((c for c in df.columns if "日期" in c), None)
    zu_col = next((c for c in df.columns if "涨停" in c and "数量" in c), None) \
             or next((c for c in df.columns if "涨停" in c), None)
    zd_col = next((c for c in df.columns if "跌停" in c and "数量" in c), None) \
             or next((c for c in df.columns if "跌停" in c), None)
    zb_col = next((c for c in df.columns if "炸板" in c), None)
    out = pd.DataFrame()
    if date_col:
        out["date"] = pd.to_datetime(df[date_col])
    if zu_col:
        out["limit_up"] = pd.to_numeric(df[zu_col], errors="coerce")
    if zd_col:
        out["limit_down"] = pd.to_numeric(df[zd_col], errors="coerce")
    if zb_col:
        out["failed_limit_up"] = pd.to_numeric(df[zb_col], errors="coerce")
    if "limit_up" in out and "failed_limit_up" in out:
        total = out["limit_up"] + out["failed_limit_up"]
        out["fail_rate"] = (out["failed_limit_up"] / total.replace(0, pd.NA)).fillna(0)
    if "date" in out:
        out = out.set_index("date").sort_index()
    return out


# ============ 板块 / 概念 ============

def sector_limit_up_count(date) -> pd.DataFrame:
    """当日各行业涨停股数排行。

    实现思路:拉涨停股池,按 '所属行业' 字段 groupby count。
    """
    zt = daily_limit_up(date)
    if zt.empty:
        return zt
    col = next((c for c in zt.columns if "行业" in c), None)
    if col is None:
        return pd.DataFrame()
    out = zt.groupby(col).size().reset_index(name="limit_up_count") \
              .sort_values("limit_up_count", ascending=False) \
              .reset_index(drop=True)
    return out


def concept_limit_up_count(date) -> pd.DataFrame:
    """当日各概念 / 题材涨停股数排行。

    注意:概念字段可能为 '概念' / '所属概念' / '涨停原因' 等。
    """
    zt = daily_limit_up(date)
    if zt.empty:
        return zt
    # 优先看 '涨停原因'(东财口径的题材归因)
    col = next((c for c in zt.columns
                if "原因" in c or "概念" in c or "题材" in c), None)
    if col is None:
        return pd.DataFrame()
    # '涨停原因' 可能是字符串,用 ; 分隔多个概念
    rows: list[dict] = []
    for _, row in zt.iterrows():
        reasons = str(row[col])
        for r in reasons.replace(",", ";").split(";"):
            r = r.strip()
            if not r:
                continue
            rows.append({"concept": r, "code": row.get("代码", "")})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).groupby("concept").agg(
        code_count=("code", "nunique")
    ).reset_index().sort_values("code_count", ascending=False).reset_index(drop=True)
    return out


# ============ 综合市场情绪 ============

def market_sentiment(date) -> dict:
    """综合市场情绪指标。

    返回:
        limit_up_count       涨停数
        limit_down_count     跌停数
        failed_limit_up_count 炸板数
        zr_ratio             涨跌停比(涨停 / 跌停,跌停为 0 时返回 inf)
        failed_rate          炸板率(炸板 / (涨停 + 炸板))
        top_sector           涨停最多的行业
        top_concept          涨停最多的概念
        strength             强弱打分(0~1)
    """
    zt = daily_limit_up(date)
    zd = daily_limit_down(date)
    zb = failed_limit_up(date)
    zu_n = len(zt) if not zt.empty else 0
    zd_n = len(zd) if not zd.empty else 0
    zb_n = len(zb) if not zb.empty else 0
    zr_ratio = (zu_n / zd_n) if zd_n > 0 else float("inf")
    failed_rate = (zb_n / (zu_n + zb_n)) if (zu_n + zb_n) > 0 else 0.0
    sec = sector_limit_up_count(date)
    con = concept_limit_up_count(date)
    top_sector = sec.iloc[0].iloc[0] if not sec.empty else ""
    top_concept = con.iloc[0]["concept"] if not con.empty else ""
    # strength: 0~1
    # - 涨停越多越强
    # - 炸板率越低越强
    # - 涨跌停比越高越强
    s_count = min(zu_n / 100.0, 1.0)              # 100 个涨停算满分
    s_failed = max(0.0, 1.0 - failed_rate)         # 炸板率 0% 满分
    s_zr = min(zr_ratio / 5.0, 1.0) if zr_ratio != float("inf") else 1.0
    strength = (s_count + s_failed + s_zr) / 3.0
    return {
        "limit_up_count": zu_n,
        "limit_down_count": zd_n,
        "failed_limit_up_count": zb_n,
        "zr_ratio": zr_ratio,
        "failed_rate": failed_rate,
        "top_sector": top_sector,
        "top_concept": top_concept,
        "strength": strength,
    }


# ============ 便捷:一字板识别 ============

def first_limit_up(date) -> pd.DataFrame:
    """当日一字涨停股(开盘即涨停,中间未打开)。"""
    zt = daily_limit_up(date)
    if zt.empty:
        return zt
    # '封板时间' 为 09:25 / 09:30 通常意味一字板
    col = next((c for c in zt.columns if "封板时间" in c or "首次" in c and "时间" in c), None)
    if col is None:
        return zt
    out = zt[zt[col].astype(str).str.startswith(("09:25", "09:30"))].reset_index(drop=True)
    return out
