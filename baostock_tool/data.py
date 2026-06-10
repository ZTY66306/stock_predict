"""数据获取层:封装 baostock 全部常用接口,统一返回 pandas DataFrame。

典型用法:
    from baostock_tool.data import get_kline, get_industry, get_index_constituents
    df = get_kline("sh.600000", "2024-01-01", "2024-12-31")
    industries = get_industry()
    hs300 = get_index_constituents("hs300", "2024-06-01")
"""
from __future__ import annotations

import logging
from typing import Iterable, Literal, Optional

import baostock as bs
import pandas as pd
from tqdm import tqdm

from baostock_tool import client, data_cache
from baostock_tool.utils import normalize_kline_df, safe_float, to_date

logger = logging.getLogger(__name__)

# 常用频率映射
FREQUENCY_D = "d"
FREQUENCY_W = "w"
FREQUENCY_M = "m"
FREQUENCY_5 = "5"
FREQUENCY_15 = "15"
FREQUENCY_30 = "30"
FREQUENCY_60 = "60"

# 复权方式
ADJ_NONE = "3"
ADJ_FRONT = "1"   # 前复权
ADJ_BACK = "2"    # 后复权

# K 线常用字段(全量)
KLINE_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)

# ============ 交易日/股票列表 ============

def get_trade_calendar(start: str = "2020-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """获取交易日历。返回字段: calendar_date, is_trading_day"""
    client.ensure_login()
    end = end or to_date(pd.Timestamp.now())
    rs = bs.query_trade_dates(start_date=start, end_date=end)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame(columns=["calendar_date", "is_trading_day"])
    df["is_trading_day"] = df["is_trading_day"].astype(int)
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    return df


def get_all_stocks(date: Optional[str] = None) -> pd.DataFrame:
    """获取指定日期的全市场股票列表。返回 code, tradeStatus, code_name 等。"""
    client.ensure_login()
    date = date or to_date(pd.Timestamp.now())
    rs = bs.query_all_stock(day=date)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()
    if not df.empty and "tradeStatus" in df.columns:
        df = df[df["tradeStatus"] == "1"].reset_index(drop=True)
    return df


def get_stock_basic(code: Optional[str] = None,
                    code_name: Optional[str] = None) -> pd.DataFrame:
    """获取股票基础信息(可按 code 或名称模糊查询)。"""
    client.ensure_login()
    rs = bs.query_stock_basic(code=code or "", code_name=code_name or "")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


# ============ 指数成分股 ============

def get_index_constituents(index: Literal["hs300", "sz50", "zz500"], date: str) -> pd.DataFrame:
    """获取指数成分股。index: hs300(沪深300)/sz50(上证50)/zz500(中证500)"""
    client.ensure_login()
    func = {
        "hs300": bs.query_hs300_stocks,
        "sz50": bs.query_sz50_stocks,
        "zz500": bs.query_zz500_stocks,
    }[index]
    rs = func(day=date)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


# ============ 行业/概念 ============

def get_industry(src: Literal["sw", "industry"] = "sw") -> pd.DataFrame:
    """获取行业分类。src: 'sw' 申万 / 'industry' baostock 内置。"""
    client.ensure_login()
    rs = bs.query_stock_industry() if src == "industry" else bs.query_shenwan_industry()
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


# ============ K 线 ============

def get_kline(
    code: str,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    frequency: str = FREQUENCY_D,
    adjustflag: str = ADJ_FRONT,
    fields: str = KLINE_FIELDS,
    use_cache: bool = True,
    cache_dir: str = ".cache/kline",
    refresh: bool = False,
) -> pd.DataFrame:
    """获取 K 线数据,自动转数值类型、date 索引。

    code: 'sh.600000' / 'sz.000001'
    frequency: d / w / m / 5 / 15 / 30 / 60
    adjustflag: 1 前复权 / 2 后复权 / 3 不复权

    缓存:
        use_cache=True 时先查本地 .cache/kline,命中即返回;
        refresh=True 强制从 baostock 重新拉取并写回。
    """
    client.ensure_login()
    end = end or to_date(pd.Timestamp.now())

    if use_cache and not refresh:
        cached = data_cache.read_cached(code, start, end, frequency, adjustflag, cache_dir)
        if cached is not None and not cached.empty:
            return cached

    rs = bs.query_history_k_data_plus(
        code, fields, start_date=start, end_date=end,
        frequency=frequency, adjustflag=adjustflag,
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        logger.warning("get_kline(%s) 返回空: %s", code, rs.error_msg)
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=rs.fields)
    df = normalize_kline_df(df)

    if use_cache and not df.empty:
        # 写缓存时用稍大区间,便于后续 start/end 切片命中
        try:
            full_start = min(start, df.index.min().strftime("%Y-%m-%d"))
            full_end = max(end, df.index.max().strftime("%Y-%m-%d"))
            data_cache.write_cached(code, df, frequency, adjustflag, cache_dir)
        except Exception as e:
            logger.debug("缓存写入 %s 失败: %s", code, e)

    return df


def get_kline_batch(
    codes: Iterable[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    frequency: str = FREQUENCY_D,
    adjustflag: str = ADJ_FRONT,
    show_progress: bool = True,
) -> pd.DataFrame:
    """批量拉取多只股票的 K 线,返回带 code 列的合并 DataFrame。"""
    client.ensure_login()
    end = end or to_date(pd.Timestamp.now())
    out = []
    iterator = tqdm(list(codes), desc="拉取K线") if show_progress else codes
    for c in iterator:
        df = get_kline(c, start, end, frequency, adjustflag)
        if not df.empty:
            out.append(df.assign(code=c))
    if not out:
        return pd.DataFrame()
    return pd.concat(out).sort_values(["code", "date"]).reset_index()


# ============ 财务数据 ============

def get_profit(code: str, year: int, quarter: int) -> pd.DataFrame:
    """利润表(单季度)。quarter: 1/2/3/4"""
    client.ensure_login()
    rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def get_balance(code: str, year: int, quarter: int) -> pd.DataFrame:
    """资产负债表"""
    client.ensure_login()
    rs = bs.query_balance_data(code=code, year=year, quarter=quarter)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def get_cash_flow(code: str, year: int, quarter: int) -> pd.DataFrame:
    """现金流量表"""
    client.ensure_login()
    rs = bs.query_cash_flow_data(code=code, year=year, quarter=quarter)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def get_forecast(code: str, start: str = "2020-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """业绩预告"""
    client.ensure_login()
    end = end or to_date(pd.Timestamp.now())
    rs = bs.query_forecast_report(code=code, start_date=start, end_date=end)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def get_top_holders(code: str, year: int, quarter: int) -> pd.DataFrame:
    """前十大股东"""
    client.ensure_login()
    rs = bs.query_top_holders(code=code, year=year, quarter=quarter)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


# ============ 宏观数据 ============

def get_macro(start: str = "2010-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """宏观经济数据(CPI/PPI/GDP 等)"""
    client.ensure_login()
    end = end or to_date(pd.Timestamp.now())
    rs = bs.query_macro_data(start_date=start, end_date=end)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def get_money_supply(start: str = "2010-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """货币供应量月度数据"""
    client.ensure_login()
    end = end or to_date(pd.Timestamp.now())
    rs = bs.query_money_supply_data_month(start_date=start, end_date=end)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def get_required_reserve(start: str = "2010-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """存款准备金率"""
    client.ensure_login()
    end = end or to_date(pd.Timestamp.now())
    rs = bs.query_required_reserve_ratio(start_date=start, end_date=end)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def get_deposit_rate(start: str = "2010-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """存款利率"""
    client.ensure_login()
    end = end or to_date(pd.Timestamp.now())
    rs = bs.query_deposit_rate_file(start_date=start, end_date=end)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


# ============ 便捷:获取全市场代码 ============

def get_all_codes(date: Optional[str] = None) -> list[str]:
    """获取全市场交易股票代码列表。"""
    df = get_all_stocks(date)
    if df.empty or "code" not in df.columns:
        return []
    return df["code"].tolist()


def get_index_codes(index: Literal["hs300", "sz50", "zz500"], date: str) -> list[str]:
    df = get_index_constituents(index, date)
    if df.empty:
        return []
    col = "code" if "code" in df.columns else df.columns[0]
    return df[col].tolist()


# ============ 复权因子 / 分红 / 估值 / 成长 / 运营 / 业绩快报 ============

def get_adjust_factor(code: str, start: str = "2020-01-01",
                       end: Optional[str] = None) -> pd.DataFrame:
    """复权因子(用于复权价自校验)。"""
    client.ensure_login()
    end = end or to_date(pd.Timestamp.now())
    rs = bs.query_adjust_factor(code=code, start_date=start, end_date=end)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()
    if not df.empty and "dividOperateDate" in df.columns:
        df["dividOperateDate"] = pd.to_datetime(df["dividOperateDate"])
    return df


def get_dividend(code: str, year: int, year_type: str = "report") -> pd.DataFrame:
    """分红送股。year_type: 'report'(预案) / 'operate'(实施)"""
    client.ensure_login()
    rs = bs.query_dividend_data(code=code, year=year, yearType=year_type)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def get_valuation(code: str, start: str = "2020-01-01",
                  end: Optional[str] = None) -> pd.DataFrame:
    """历史估值数据(PB/PE/PS/PCF)。"""
    client.ensure_login()
    end = end or to_date(pd.Timestamp.now())
    rs = bs.query_valuation_data(code=code, start_date=start, end_date=end)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()
    if not df.empty and "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"])
        df = df.set_index("day").sort_index()
    return df


def get_growth(code: str, year: int, quarter: int) -> pd.DataFrame:
    """成长性数据(同比/环比)。"""
    client.ensure_login()
    rs = bs.query_growth_data(code=code, year=year, quarter=quarter)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def get_operation(code: str, year: int, quarter: int) -> pd.DataFrame:
    """运营能力(周转率)。"""
    client.ensure_login()
    rs = bs.query_operation_data(code=code, year=year, quarter=quarter)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def get_performance_express(code: str, start: str = "2020-01-01",
                             end: Optional[str] = None) -> pd.DataFrame:
    """业绩快报。"""
    client.ensure_login()
    end = end or to_date(pd.Timestamp.now())
    rs = bs.query_performance_express_report(code=code, start_date=start, end_date=end)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def get_securities_info(code: str) -> dict:
    """个股元信息(IPO 日 / 流通股本等)。返回 dict;失败返回空 dict。

    实现说明:不同 baostock 版本对单只股票元信息接口不一致 ——
    老版本提供 `query_securities_info(code)`,新版本(>=0.8.9)改用
    `query_stock_basic(code=...)` 统一承担。这里用 try/except 同时兼容两种接口,
    保证在不同 baostock 版本下都能正常返回。

    返回字段(成功时):code, code_name, ipoDate, outDate, type, status
    """
    client.ensure_login()
    # 优先用新版接口
    try:
        rs = bs.query_stock_basic(code=code)
        if rs.error_code == "0" and rs.next():
            return dict(zip(rs.fields, rs.get_row_data()))
        return {}
    except AttributeError:
        pass
    # 兜底:老版接口
    try:
        rs = bs.query_securities_info(code=code)
        if rs.error_code == "0" and rs.next():
            return dict(zip(rs.fields, rs.get_row_data()))
        return {}
    except AttributeError:
        return {}


def get_industry_detail(code: str, date: str) -> pd.DataFrame:
    """个股所属行业明细。"""
    client.ensure_login()
    rs = bs.query_industry_detail(code=code, date=date)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def get_security_benchmarks(date: str) -> pd.DataFrame:
    """基准指数列表。"""
    client.ensure_login()
    rs = bs.query_bench_mark(date=date)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


# ============ 缓存管理便捷函数 ============

def clear_kline_cache(cache_dir: str = ".cache/kline") -> int:
    """清空 K 线缓存,返回删除文件数。"""
    return data_cache.clear(cache_dir)


def kline_cache_info(cache_dir: str = ".cache/kline") -> dict:
    """查看 K 线缓存统计。"""
    return data_cache.cache_info(cache_dir)
