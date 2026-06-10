"""Dashboard 共用工具:数据缓存、UI 组件、配置。

所有数据拉取都走 st.cache_data,避免每次点按钮都重新拉 baostock / akshare。
所有 I/O 调用通过线程池 + 超时,避免单线程 streamlit 主线程被网络请求卡死。
"""
from __future__ import annotations

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

from baostock_tool import data

logger = logging.getLogger(__name__)

# ============ 线程池 + 超时 ============
# Streamlit 单线程跑 ScriptRunner,任何一次阻塞的 baostock/akshare 调用都会让
# 所有新 WebSocket 堆在 accept 队列里卡死(症状:recv-q 暴涨、浏览器报
# "Connection timed out")。下面用共享线程池 + 显式 timeout 把阻塞调用隔离掉,
# 主线程最多等 N 秒,超时后返回 fallback,服务继续 accept。
#
# 调参:导出环境变量 `DASHBOARD_IO_TIMEOUT=秒数` 即可全站生效,默认 20s。
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dash-io")
_DEFAULT_TIMEOUT = float(os.environ.get("DASHBOARD_IO_TIMEOUT", "20"))


def _call_with_timeout(fn, timeout: float, *args, **kwargs):
    """在线程池里跑 fn,timeout 秒后抛 FuturesTimeout。"""
    fut = _EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout)
    except FuturesTimeout:
        fut.cancel()
        logger.warning(
            "%s 超时(>%.1fs)", getattr(fn, "__name__", repr(fn)), timeout
        )
        raise


# ============ 数据缓存 ============

@st.cache_data(ttl=3600, show_spinner="拉取 K 线...")
def cached_kline(code: str, start: str, end: str,
                 frequency: str = "d", adjustflag: str = "1") -> pd.DataFrame:
    """缓存 K 线 1 小时。"""
    try:
        return _call_with_timeout(
            data.get_kline, 20.0,
            code, start, end, frequency=frequency, adjustflag=adjustflag,
        )
    except FuturesTimeout:
        st.warning(f"K 线拉取超时(20s): {code} {start}~{end}")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"K 线拉取失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner="查询股票元信息...")
def cached_securities_info(code: str) -> dict:
    """缓存元信息 1 天。"""
    try:
        return _call_with_timeout(data.get_securities_info, 10.0, code) or {}
    except FuturesTimeout:
        st.warning(f"元信息查询超时(10s): {code}")
        return {}
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner="查询股票列表...")
def cached_all_stocks(date: str) -> list[str]:
    """缓存全市场股票列表 1 天。"""
    try:
        return _call_with_timeout(data.get_all_codes, 30.0, date)
    except FuturesTimeout:
        st.warning(f"全市场股票列表拉取超时(30s): {date}")
        return []
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner="指数成分股...")
def cached_index_codes(index: str, date: str) -> list[str]:
    try:
        return _call_with_timeout(data.get_index_codes, 15.0, index, date)
    except FuturesTimeout:
        st.warning(f"指数成分股拉取超时(15s): {index} {date}")
        return []
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner="查询资金流...")
def cached_fund_flow(code: str) -> pd.DataFrame:
    """资金流缓存 5 分钟(实时数据)。"""
    from baostock_tool import fund_flow as ff
    if not ff._HAS_AKSHARE:
        return pd.DataFrame()
    try:
        return _call_with_timeout(ff.get_fund_flow, 15.0, code)
    except FuturesTimeout:
        st.warning(f"资金流拉取超时(15s): {code}")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"资金流拉取失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner="北向资金...")
def cached_northbound() -> pd.DataFrame:
    from baostock_tool import fund_flow as ff
    if not ff._HAS_AKSHARE:
        return pd.DataFrame()
    try:
        return _call_with_timeout(ff.get_northbound, 15.0)
    except FuturesTimeout:
        st.warning("北向资金拉取超时(15s)")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"北向资金拉取失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner="涨跌停...")
def cached_limit_up(date: str) -> pd.DataFrame:
    from baostock_tool import market_overview as mo
    if not mo._HAS_AKSHARE:
        return pd.DataFrame()
    try:
        return _call_with_timeout(mo.daily_limit_up, 20.0, date)
    except FuturesTimeout:
        st.warning(f"涨停池拉取超时(20s): {date}")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"涨停池拉取失败: {e}")
        return pd.DataFrame()


# ============ 通用 UI 组件 ============

def stock_picker(label: str = "股票", key: Optional[str] = None,
                 default: str = "", help: str = "") -> str:
    """股票代码/名称输入器。返回 sh.600000 格式的标准代码。"""
    raw = st.text_input(
        label,
        value=default,
        key=key,
        help=help or "支持代码(sh.600000)或中文名(浦发银行)",
        placeholder="sh.600000 或 浦发银行",
    ).strip()
    if not raw:
        return ""
    if "." in raw:
        return raw.lower()
    # 当成名字解析
    try:
        info = cached_securities_info(raw)
        if info and "code" in info:
            return str(info["code"]).lower()
        # fallback: 模糊查询
        df = data.get_stock_basic(code_name=raw)
        if not df.empty:
            return str(df.iloc[0]["code"]).lower()
    except Exception:
        pass
    st.warning(f"未找到股票: {raw} (请检查代码或名称)")
    return ""


def period_selector(default: str = "1y", key: Optional[str] = None) -> tuple[str, str]:
    """时间区间选择器,返回 (start, end) 'YYYY-MM-DD'。"""
    presets = {
        "3m": 90,
        "6m": 180,
        "1y": 365,
        "2y": 365 * 2,
        "3y": 365 * 3,
        "5y": 365 * 5,
    }
    cols = st.columns([1, 2])
    with cols[0]:
        choice = st.selectbox("时间区间", list(presets.keys()) + ["自定义"],
                                index=list(presets.keys()).index(default)
                                if default in presets else 2, key=key)
    end = datetime.now().strftime("%Y-%m-%d")
    if choice == "自定义":
        with cols[1]:
            d = st.date_input("选择起止", value=(datetime.now() - timedelta(days=365),
                                                datetime.now()),
                               key=(key or "") + "_custom")
        if isinstance(d, tuple) and len(d) == 2:
            return d[0].strftime("%Y-%m-%d"), d[1].strftime("%Y-%m-%d")
        return "", end
    days = presets[choice]
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return start, end


def show_df(df: pd.DataFrame, height: int = 400, key: Optional[str] = None):
    """统一 dataframe 展示。"""
    if df is None or df.empty:
        st.info("暂无数据")
        return
    st.dataframe(df, height=height, key=key, width="stretch")


def show_metrics(metrics: dict, cols: int = 4):
    """用 st.metric 展示一组关键数字。"""
    items = list(metrics.items())
    rows = [items[i:i + cols] for i in range(0, len(items), cols)]
    for row in rows:
        cs = st.columns(len(row))
        for c, (k, v) in zip(cs, row):
            if isinstance(v, dict):
                c.metric(k, v.get("value", ""), delta=v.get("delta"),
                          delta_color=v.get("color", "normal"))
            else:
                c.metric(k, v)


def plotly_or_matplotlib(fig, use_plotly: bool = True):
    """统一图表展示。"""
    if use_plotly:
        try:
            import plotly.graph_objects as go
            st.plotly_chart(fig, width="stretch")
            return
        except ImportError:
            pass
    st.pyplot(fig)


def download_button(df: pd.DataFrame, filename: str, label: str = "下载 CSV"):
    """DataFrame 下载按钮。"""
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, csv, filename, "text/csv", width="stretch")


def error_boundary(func):
    """装饰器:页面级异常捕获,在 Streamlit 里显示友好错误。"""
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ 运行时错误: {type(e).__name__}: {e}")
            with st.expander("🔍 详细堆栈"):
                import traceback
                st.code(traceback.format_exc())
    return wrapper


def akshare_warning():
    """在用 akshare 的页面前检查并提示。"""
    from baostock_tool import fund_flow as ff, market_overview as mo
    if not ff._HAS_AKSHARE:
        st.warning(
            "⚠️ **本功能依赖 akshare**,未检测到。运行 `pip install akshare` 后重启 dashboard。"
        )
        return False
    return True


def section_header(title: str, subtitle: str = ""):
    """统一的章节标题。"""
    if subtitle:
        st.markdown(f"### {title}\n*{subtitle}*")
    else:
        st.markdown(f"### {title}")
    st.divider()


def check_login():
    """确保 baostock 已登录(避免每个页面都写)。"""
    try:
        from baostock_tool import client
        _call_with_timeout(client.ensure_login, 10.0)
        return True
    except FuturesTimeout:
        st.error("❌ baostock 登录超时(10s),网络或 baostock 服务可能不可用。")
        return False
    except Exception as e:
        st.error(f"❌ baostock 登录失败: {e}\n\n请到终端跑 `python -m baostock_tool.cli login`")
        return False
