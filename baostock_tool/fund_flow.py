"""资金流 / 北向资金 / 龙虎榜(akshare 集成)。

提供 A 股实战的"题材/资金"数据:
    - get_fund_flow(code)         个股资金流(主力/超大/大/中/小单 净流入)
    - get_northbound()            北向资金汇总(沪股通 + 深股通)
    - get_longhubang(start, end)  龙虎榜(机构/游资席位 + 解读)
    - get_sector_fund_flow()      板块资金流排名(行业 / 概念)

依赖:akshare(可选)。未装时所有函数抛 ImportError,指引用户 pip install akshare。
akshare 数据源来自东方财富等公开站点,字段命名以东财为准(中文)。
"""
from __future__ import annotations

import logging
import os
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
            "也可绕过本模块,自行从东财/同花顺抓取数据。"
        )


def _normalize_code(code: str) -> tuple[str, str]:
    """'sh.600000' → ('600000', 'sh');'600000' → ('600000', '')."""
    code = code.strip()
    if "." in code:
        prefix, body = code.split(".", 1)
        return body, prefix.lower()
    return code, ""


# ============ 个股资金流 ============

def get_fund_flow(code: str, start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> pd.DataFrame:
    """个股资金流(主力/超大/大单/中单/小单 净流入,东财源)。

    code: 'sh.600000' / 'sz.000001' / '600000'
    start_date / end_date: 'YYYY-MM-DD' 或 'YYYYMMDD';None 拉最近
    返回:DataFrame,关键字段包括
        主力净流入 / 超大单净流入 / 大单净流入 / 中单净流入 / 小单净流入
    """
    _require_akshare()
    body, market = _normalize_code(code)
    if not market:
        # 用 stock_individual_fund_flow_rank 不需要 market,但不带日期范围
        market = "sh" if body.startswith("6") or body.startswith("9") or body.startswith("5") else "sz"
    try:
        df = ak.stock_individual_fund_flow(stock=body, market=market)
    except Exception as e:
        logger.warning("stock_individual_fund_flow 失败,改用 _rank: %s", e)
        df = ak.stock_individual_fund_flow_rank(indicator="今日")
    if df is None or df.empty:
        return pd.DataFrame()
    # 日期列名不固定,统一找 '日期' / '时间' 类
    date_col = next((c for c in df.columns if "日期" in c or "时间" in c), None)
    if date_col and (start_date or end_date):
        s = (start_date or "").replace("-", "")
        e = (end_date or "").replace("-", "")
        dates = df[date_col].astype(str)
        # 合并 mask 一次性过滤(避免 chain filter 索引错位导致 ValueError)
        mask = pd.Series(True, index=df.index)
        if s:
            mask &= (dates.values >= s)
        if e:
            mask &= (dates.values <= e)
        df = df[mask.values]
    return df.reset_index(drop=True)


# ============ 北向资金 ============

def get_northbound() -> pd.DataFrame:
    """沪深股通资金流向汇总(实时/最近交易日)。

    返回字段:类型 / 板块 / 资金方向 / 成交净买额 / 资金净流入 等
    """
    _require_akshare()
    df = ak.stock_hsgt_fund_flow_summary_em()
    return df.reset_index(drop=True) if df is not None else pd.DataFrame()


def get_northbound_history(start_date: str = "20240101",
                            end_date: str = "20251231") -> pd.DataFrame:
    """北向资金历史日度数据(沪股通 + 深股通 每日净买额)。

    start_date / end_date: 'YYYYMMDD'
    """
    _require_akshare()
    sh = ak.stock_hsgt_hist_em(symbol="沪股通", period="daily",
                                start_date=start_date, end_date=end_date)
    sz = ak.stock_hsgt_hist_em(symbol="深股通", period="daily",
                                start_date=start_date, end_date=end_date)
    parts = []
    for name, df in [("沪股通", sh), ("深股通", sz)]:
        if df is None or df.empty:
            continue
        d = df.copy()
        d["通道"] = name
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ============ 龙虎榜 ============

def get_longhubang(start_date: str, end_date: str) -> pd.DataFrame:
    """龙虎榜每日详情(东财源)。

    start_date / end_date: 'YYYYMMDD'
    返回字段:代码 / 名称 / 上榜日 / 解读 / 收盘价 / 净买额 / 买入总额 / 卖出总额 等
    """
    _require_akshare()
    df = ak.stock_lhb_detail_em(start_date=start_date, end_date=end_date)
    return df.reset_index(drop=True) if df is not None else pd.DataFrame()


def get_longhubang_ggtj(start_date: str = "20240101",
                          end_date: str = "20251231") -> pd.DataFrame:
    """龙虎榜个股统计(累计上榜次数 / 累计净买额 / 解读)。"""
    _require_akshare()
    df = ak.stock_lhb_ggtj_em(start_date=start_date, end_date=end_date)
    return df.reset_index(drop=True) if df is not None else pd.DataFrame()


# ============ 板块资金流 ============

def get_sector_fund_flow(indicator: str = "今日",
                          sector_type: str = "行业资金流") -> pd.DataFrame:
    """板块资金流排名。

    indicator: '今日' / '3日' / '5日' / '10日'
    sector_type: '行业资金流' / '概念资金流' / '地域资金流'
    """
    _require_akshare()
    df = ak.stock_sector_fund_flow_rank(indicator=indicator, sector_type=sector_type)
    return df.reset_index(drop=True) if df is not None else pd.DataFrame()


# ============ 便捷:筛资金流入 + 价格上涨 ============

def strong_capital_inflow(threshold: float = 0.0) -> pd.DataFrame:
    """今日板块资金流排名中,主力净流入 > threshold 的板块。"""
    df = get_sector_fund_flow()
    if df.empty:
        return df
    # 列名可能是 '主力净流入' / '净流入' / '主力资金净流入' 等
    col = next((c for c in df.columns if "主力" in c and "净流入" in c), None)
    if col is None:
        col = next((c for c in df.columns if "净流入" in c), None)
    if col is None:
        return df
    # 试着把列转数值(可能含 '亿' / '万' / '%' / 字符串)
    s = df[col].astype(str).str.replace("亿", "e8").str.replace("万", "e4").str.replace("%", "")
    s = s.str.replace(",", "").str.replace("—", "").str.replace("--", "")
    try:
        df["_净流入_num"] = pd.to_numeric(s, errors="coerce")
    except Exception:
        return df
    out = df[df["_净流入_num"] > threshold].drop(columns=["_净流入_num"])
    return out.reset_index(drop=True)


# ============ 缓存(轻量) ============

_CACHE_DIR = os.path.expanduser("~/.baostock_tool/fund_flow_cache")


def clear_cache() -> int:
    """清空资金流缓存目录,返回删除文件数。"""
    if not os.path.isdir(_CACHE_DIR):
        return 0
    n = 0
    for f in os.listdir(_CACHE_DIR):
        p = os.path.join(_CACHE_DIR, f)
        if os.path.isfile(p):
            os.remove(p)
            n += 1
    return n
