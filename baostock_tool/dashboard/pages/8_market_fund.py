"""资金流 / 北向 / 涨跌停 / 题材 实时面板(需 akshare)。"""
from __future__ import annotations

from datetime import datetime, timedelta

import streamlit as st

from baostock_tool.dashboard import utils

st.set_page_config(page_title="资金流/涨跌停", page_icon="📡", layout="wide")

utils.section_header("📡 资金流 / 涨跌停 / 题材 实时面板",
                      "依赖 akshare(未装时本页面功能受限)")

if not utils.akshare_warning():
    st.stop()

# ===== 选项卡 =====
tab1, tab2, tab3, tab4 = st.tabs(["💰 资金流", "🌐 北向资金", "🚀 涨跌停", "🔥 题材/连板"])

# ============ 1. 资金流 ============
with tab1:
    utils.section_header("个股资金流")
    code = utils.stock_picker("股票", default="sh.600000", key="ff_code")
    if code:
        df = utils.cached_fund_flow(code)
        if df.empty:
            st.warning("无数据(可能非交易日 / 网络问题)")
        else:
            st.success(f"✅ 拉到 {len(df)} 条")
            utils.show_df(df.head(30), height=400)
            utils.download_button(df.head(60), f"{code}_fund_flow.csv")

# ============ 2. 北向资金 ============
with tab2:
    utils.section_header("🌐 北向资金汇总")
    if st.button("🔄 刷新", key="nb_refresh"):
        st.cache_data.clear()
    df = utils.cached_northbound()
    if df.empty:
        st.warning("无数据")
    else:
        st.success(f"✅ 拉到 {len(df)} 条")
        utils.show_df(df, height=400)

# ============ 3. 涨跌停 ============
with tab3:
    utils.section_header("🚀 当日涨停股池")
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("日期", value=(datetime.now() - timedelta(days=1)).date(),
                              key="lu_date")
    with col2:
        if st.button("🔄 刷新", key="lu_refresh"):
            st.cache_data.clear()

    df_lu = utils.cached_limit_up(str(date))
    if df_lu.empty:
        st.warning("无涨停数据(可能非交易日 / akshare 接口问题)")
    else:
        st.success(f"✅ 当日涨停 {len(df_lu)} 只")
        utils.show_df(df_lu, height=500)
        utils.download_button(df_lu, f"limit_up_{date}.csv")

# ============ 4. 题材/连板 ============
with tab4:
    utils.section_header("🔥 题材涨停排行")
    from baostock_tool import market_overview as mo
    if st.button("🔄 刷新", key="theme_refresh"):
        st.cache_data.clear()
    date_t = st.date_input("日期", value=(datetime.now() - timedelta(days=1)).date(),
                            key="theme_date")
    try:
        sec = mo.sector_limit_up_count(str(date_t))
        if not sec.empty:
            st.markdown("**行业涨停排行**")
            st.dataframe(sec.head(20), width="stretch", height=400)
        con = mo.concept_limit_up_count(str(date_t))
        if not con.empty:
            st.markdown("**概念/题材涨停排行**")
            st.dataframe(con.head(20), width="stretch", height=400)
        # 市场情绪
        s = mo.market_sentiment(str(date_t))
        st.markdown("**市场情绪综合**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("涨停数", s["limit_up_count"])
        c2.metric("跌停数", s["limit_down_count"])
        c3.metric("炸板率", f"{s['failed_rate']*100:.1f}%")
        c4.metric("**情绪打分**", f"**{s['strength']:.2%}**")
    except Exception as e:
        st.warning(f"获取失败: {e}")
