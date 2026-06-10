"""选股页面:16 个内置模板 + 自定义条件组合。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from baostock_tool import screener
from baostock_tool.dashboard import utils

st.set_page_config(page_title="选股", page_icon="🔍", layout="wide")
utils.check_login() or st.stop()

utils.section_header("🔍 选股", "16 个内置模板 + 自定义条件组合")

# ===== 模板选择 =====
utils.section_header("模板选择")
tmpl = st.selectbox("选股模板", list(screener.SCREEN_TEMPLATES.keys()),
                     format_func=lambda k: f"{k}  —  {screener.SCREEN_TEMPLATES[k]}")

col1, col2, col3 = st.columns(3)
with col1:
    date = st.date_input("选股日期", value=pd.Timestamp.now().date())
with col2:
    lookback = st.number_input("回看天数", value=120, min_value=30, max_value=500)
with col3:
    limit = st.number_input("最多选股", value=30, min_value=5, max_value=200)

if st.button("🚀 跑选股", type="primary", width="stretch"):
    with st.spinner("拉数据 + 跑条件中..."):
        try:
            picks = screener.screen(tmpl, date=str(date),
                                    lookback=lookback, limit=limit)
        except Exception as e:
            st.error(f"❌ 选股失败: {e}")
            st.stop()
    if picks.empty:
        st.warning("⚠️ 没有股票命中,可以放宽 lookback 或换模板试试")
    else:
        st.success(f"✅ 命中 {len(picks)} 只")
        st.dataframe(picks, width="stretch", height=500)
        utils.download_button(picks, f"screener_{tmpl}_{date}.csv")


# ===== 自定义条件 =====
utils.section_header("自定义条件", "把多个条件 AND 起来,跑出自己的选股规则")
preset_conds = {
    "MA5 > MA20": screener.MA5_gt_MA20,
    "MA20 > MA60": screener.MA20_gt_MA60,
    "MACD 金叉": screener.macd_golden_cross,
    "MACD 死叉": screener.macd_death_cross,
    "KDJ 超卖": screener.kdj_oversold,
    "KDJ 超买": screener.kdj_overbought,
    "RSI < 30": screener.rsi_oversold,
    "RSI > 70": screener.rsi_overbought,
    "BOLL 下破": screener.boll_lower_break,
    "BOLL 上破": screener.boll_upper_break,
    "量能突破(2x)": screener.volume_breakout,
    "20 日新高": screener.new_high_n,
    "20 日新低": screener.new_low_n,
    "形态:看涨吞没": lambda df: screener.pattern("engulfing_bullish")(df),
    "形态:早晨之星": lambda df: screener.pattern("morning_star")(df),
    "形态:看涨共振(>=2)": screener.pattern_bullish_score(2),
}

s = screener.Screen(date=str(date), lookback_days=lookback)
chosen = st.multiselect("选条件(AND 关系)", list(preset_conds.keys()))
for c in chosen:
    s.add(preset_conds[c])

# 量化条件
with st.expander("➕ 加量化条件(PE / PB / 涨幅 / 换手率 / 量能)"):
    pe_on = st.checkbox("加 PE 区间")
    if pe_on:
        pe_lo, pe_hi = st.slider("PE_TTM 范围", 0, 200, (0, 30))
        s.add(screener.pe_between(pe_lo, pe_hi))
    pb_on = st.checkbox("加 PB 区间")
    if pb_on:
        pb_lo, pb_hi = st.slider("PB 范围", 0.0, 20.0, (0.0, 5.0))
        s.add(screener.pb_between(pb_lo, pb_hi))
    pc_on = st.checkbox("加涨跌幅区间(%)")
    if pc_on:
        pc_lo, pc_hi = st.slider("涨跌幅", -20.0, 20.0, (0.0, 5.0))
        s.add(screener.pct_change_between(pc_lo / 100, pc_hi / 100))

if st.button("🎯 跑自定义选股", type="secondary", width="stretch"):
    if not chosen and not (pe_on or pb_on or pc_on):
        st.warning("请至少勾一个条件")
        st.stop()
    with st.spinner("拉数据 + 跑条件中..."):
        try:
            picks = s.run(limit=limit, show_progress=False)
        except Exception as e:
            st.error(f"❌ 自定义选股失败: {e}")
            st.stop()
    if picks.empty:
        st.warning("⚠️ 没股票命中,可放宽条件")
    else:
        st.success(f"✅ 命中 {len(picks)} 只")
        st.dataframe(picks, width="stretch", height=500)
        utils.download_button(picks, f"custom_screen_{date}.csv")
