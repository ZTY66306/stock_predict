"""Dashboard 共用工具:数据缓存、UI 组件、配置。

所有数据拉取都走 st.cache_data,避免每次点按钮都重新拉 baostock / akshare。
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

from baostock_tool import data


# ============ 数据缓存 ============

@st.cache_data(ttl=3600, show_spinner="拉取 K 线...")
def cached_kline(code: str, start: str, end: str,
                 frequency: str = "d", adjustflag: str = "1") -> pd.DataFrame:
    """缓存 K 线 1 小时。"""
    return data.get_kline(code, start, end, frequency=frequency, adjustflag=adjustflag)


@st.cache_data(ttl=86400, show_spinner="查询股票元信息...")
def cached_securities_info(code: str) -> dict:
    """缓存元信息 1 天。"""
    try:
        return data.get_securities_info(code) or {}
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner="查询股票列表...")
def cached_all_stocks(date: str) -> list[str]:
    """缓存全市场股票列表 1 天。"""
    try:
        return data.get_all_codes(date)
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner="指数成分股...")
def cached_index_codes(index: str, date: str) -> list[str]:
    try:
        return data.get_index_codes(index, date)
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner="查询资金流...")
def cached_fund_flow(code: str) -> pd.DataFrame:
    """资金流缓存 5 分钟(实时数据)。"""
    from baostock_tool import fund_flow as ff
    if not ff._HAS_AKSHARE:
        return pd.DataFrame()
    try:
        return ff.get_fund_flow(code)
    except Exception as e:
        st.warning(f"资金流拉取失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner="北向资金...")
def cached_northbound() -> pd.DataFrame:
    from baostock_tool import fund_flow as ff
    if not ff._HAS_AKSHARE:
        return pd.DataFrame()
    try:
        return ff.get_northbound()
    except Exception as e:
        st.warning(f"北向资金拉取失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner="涨跌停...")
def cached_limit_up(date: str) -> pd.DataFrame:
    from baostock_tool import market_overview as mo
    if not mo._HAS_AKSHARE:
        return pd.DataFrame()
    try:
        return mo.daily_limit_up(date)
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
    st.dataframe(df, height=height, key=key, use_container_width=True)


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
            st.plotly_chart(fig, use_container_width=True)
            return
        except ImportError:
            pass
    st.pyplot(fig)


def download_button(df: pd.DataFrame, filename: str, label: str = "下载 CSV"):
    """DataFrame 下载按钮。"""
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, csv, filename, "text/csv", use_container_width=True)


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
        client.ensure_login()
        return True
    except Exception as e:
        st.error(f"❌ baostock 登录失败: {e}\n\n请到终端跑 `python -m baostock_tool.cli login`")
        return False
