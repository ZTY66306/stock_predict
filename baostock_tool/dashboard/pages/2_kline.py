"""K 线 + 技术指标 + 形态识别 一体化页面。"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from baostock_tool import data, indicators as ind, patterns as ptn
from baostock_tool.dashboard import utils

st.set_page_config(page_title="K 线 + 指标", page_icon="📈", layout="wide")
utils.check_login() or st.stop()

utils.section_header("📈 K 线 + 技术指标", "支持加全套 16 类技术指标 + 12 种 K 线形态")

# ===== 输入区 =====
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    code = utils.stock_picker("股票", default="sh.600000",
                                help="代码或中文名")
with col2:
    frequency = st.selectbox("频率", ["d", "w", "m", "5", "15", "30", "60"], index=0)
with col3:
    adjustflag = st.selectbox("复权", [("1", "前复权"), ("2", "后复权"), ("3", "不复权")],
                                format_func=lambda x: x[1], index=0)[0]

start, end = utils.period_selector(default="1y")

if not code:
    st.info("👆 输入股票代码或名称开始")
    st.stop()

# ===== 拉数据 =====
df = utils.cached_kline(code, start, end, frequency=frequency, adjustflag=adjustflag)
if df.empty:
    st.error(f"❌ {code} 在 {start}~{end} 没数据")
    st.stop()

info = utils.cached_securities_info(code)
name = info.get("code_name", code) if info else code

st.success(f"✅ {code} {name}  共 {len(df)} 个交易日  区间 {df.index[0].date()} ~ {df.index[-1].date()}")

# ===== 指标选择 =====
utils.section_header("🔧 技术指标")
all_inds = ind.ALL_INDICATORS
ind_cols = st.columns(4)
selected_inds = []
for i, name_i in enumerate(all_inds):
    with ind_cols[i % 4]:
        if st.checkbox(name_i, value=(name_i in ["ma", "macd", "kdj", "rsi", "boll"]),
                         key=f"ind_{name_i}"):
            selected_inds.append(name_i)

if st.button("📊 加指标 + 画图", type="primary", width="stretch"):
    with st.spinner("计算中..."):
        df_plot = ind.add_all(df, include=tuple(selected_inds) if selected_inds else None)

    utils.section_header("📈 价格 + 指标图")
    # 价格 + MA
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                     gridspec_kw={"height_ratios": [2, 1]},
                                     sharex=True)
    ax1.plot(df_plot.index, df_plot["close"], color="black", linewidth=1.0, label="收盘价")
    for c in df_plot.columns:
        if c.startswith("MA"):
            ax1.plot(df_plot.index, df_plot[c], linewidth=0.7, label=c)
    if "BOLL_UP" in df_plot.columns:
        ax1.plot(df_plot.index, df_plot["BOLL_UP"], color="red", linewidth=0.5, alpha=0.5)
        ax1.plot(df_plot.index, df_plot["BOLL_LOW"], color="green", linewidth=0.5, alpha=0.5)
    ax1.set_title(f"{code} {name}  Close + MA + BOLL")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # MACD / RSI / KDJ 等副图(若有)
    if "MACD_HIST" in df_plot.columns:
        colors = ["red" if v >= 0 else "green" for v in df_plot["MACD_HIST"]]
        ax2.bar(df_plot.index, df_plot["MACD_HIST"], color=colors, alpha=0.6, label="MACD_HIST")
        ax2.plot(df_plot.index, df_plot["MACD_DIF"], color="blue", linewidth=0.7, label="DIF")
        ax2.plot(df_plot.index, df_plot["MACD_DEA"], color="orange", linewidth=0.7, label="DEA")
        ax2.set_title("MACD")
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)
    elif "RSI14" in df_plot.columns:
        ax2.plot(df_plot.index, df_plot["RSI14"], color="purple", linewidth=1.0)
        ax2.axhline(70, color="red", linestyle="--", linewidth=0.5)
        ax2.axhline(30, color="green", linestyle="--", linewidth=0.5)
        ax2.set_title("RSI(14)")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 数据表
    with st.expander("📋 看带指标的完整数据"):
        st.dataframe(df_plot.tail(60), width="stretch", height=400)
        utils.download_button(df_plot.reset_index(),
                              f"{code}_indicators.csv", "📥 下载完整数据 CSV")

# ===== 形态识别 =====
utils.section_header("🎯 K 线形态识别")
ptn_list = ptn.list_patterns()
ptn_cols = st.columns(6)
selected_ptns = []
for i, p in enumerate(ptn_list):
    with ptn_cols[i % 6]:
        if st.checkbox(p, value=False, key=f"ptn_{p}"):
            selected_ptns.append(p)

if st.button("🔍 识别形态", type="secondary") and selected_ptns:
    with st.spinner("扫描中..."):
        rows = []
        for p_name in selected_ptns:
            mask = ptn.detect(df, p_name)
            for dt in df.index[mask]:
                rows.append({
                    "日期": dt.date(),
                    "形态": p_name,
                    "收盘价": float(df.loc[dt, "close"]),
                })
        if rows:
            df_hits = pd.DataFrame(rows).sort_values("日期", ascending=False)
            st.success(f"✅ 命中 {len(df_hits)} 次")
            st.dataframe(df_hits, width="stretch", height=300)
            utils.download_button(df_hits, f"{code}_patterns.csv")
        else:
            st.info("区间内无形态命中")
