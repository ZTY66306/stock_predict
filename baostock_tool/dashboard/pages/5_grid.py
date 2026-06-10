"""网格交易复盘页面:支持 T+0/T+1、手动/自动区间、止损止盈。"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from baostock_tool import grid_backtest as gb
from baostock_tool.dashboard import utils

st.set_page_config(page_title="网格复盘", page_icon="🕸️", layout="wide")
utils.check_login() or st.stop()

utils.section_header("🕸️ 网格交易复盘", "支持代码/中文名 + T+0/T+1 + 手动/自动区间 + 止损止盈")

# ===== 输入 =====
col1, col2 = st.columns(2)
with col1:
    code = utils.stock_picker("股票", default="sh.510300",
                                help="ETF/可转债 走 T+0;股票 走 T+1")
with col2:
    mode = st.selectbox("网格模式", ["geometric", "fixed"],
                         format_func=lambda x: "等比(推荐)" if x == "geometric" else "等差")

start, end = utils.period_selector(default="2y")

utils.section_header("🎯 网格区间")
mode_range = st.radio("区间设定", ["手动", "自动(按 lookback 区间百分比)"], horizontal=True)

if mode_range == "手动":
    col1, col2 = st.columns(2)
    with col1:
        lower = st.number_input("网格下沿", value=3.0, step=0.1, format="%.2f")
    with col2:
        upper = st.number_input("网格上沿", value=4.0, step=0.1, format="%.2f")
    lower_pct = upper_pct = None
    lookback = 60
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        lower_pct = st.number_input("下沿比例", value=0.85, step=0.05, format="%.2f")
    with col2:
        upper_pct = st.number_input("上沿比例", value=1.15, step=0.05, format="%.2f")
    with col3:
        lookback = st.number_input("回看天数", value=60, min_value=20, max_value=500)
    lower = upper = None

col1, col2, col3, col4 = st.columns(4)
with col1:
    n_grids = st.slider("网格数", 3, 50, 10)
with col2:
    shares_per_grid = st.number_input("每格股数", value=100, step=100)
with col3:
    base_position = st.number_input("底仓(股)", value=0, step=100)
with col4:
    t0 = st.selectbox("交易制度", ["auto", "true", "false"],
                       format_func=lambda x: {"auto": "自动识别", "true": "强制 T+0",
                                                "false": "强制 T+1"}[x])

with st.expander("⚙️ 风控 + 成本"):
    col1, col2, col3 = st.columns(3)
    with col1:
        stop_loss = st.number_input("止损比例(0=关)", value=0.0, step=0.05, format="%.2f")
    with col2:
        take_profit = st.number_input("止盈比例(0=关)", value=0.0, step=0.05, format="%.2f")
    with col3:
        cash = st.number_input("初始资金", value=100000.0, step=10000.0)

if not code:
    st.info("👆 选股票开始")
    st.stop()

if st.button("🚀 跑网格回测", type="primary", width="stretch"):
    with st.spinner("回测中..."):
        try:
            t0_flag = None if t0 == "auto" else (t0 == "true")
            result = gb.run(
                code, start=start, end=end,
                grid_mode=mode, n_grids=n_grids,
                lower=lower, upper=upper,
                lower_pct=lower_pct, upper_pct=upper_pct,
                lookback=lookback,
                shares_per_grid=shares_per_grid, base_position=base_position,
                stop_loss=stop_loss or None, take_profit=take_profit or None,
                capital=cash, t0=t0_flag,
            )
        except Exception as e:
            st.error(f"❌ 回测失败: {e}")
            st.stop()

    st.success("✅ 回测完成")
    s = result.summary()

    # 关键指标
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总收益", s["总收益率"])
    c2.metric("年化", s["年化收益"])
    c3.metric("最大回撤", s["最大回撤"])
    c4.metric("交易制度", s["交易制度"])
    c5.metric("完整往返", s["完整往返"])

    # 网格区间显示
    st.info(f"📊 网格区间: **{s['网格区间']}**, 涨跌幅限制 **{s['涨跌幅限制']}**, 期末权益 **{s['期末权益']}**")

    # 图:价格 + 网格线 + 买卖点
    utils.section_header("📈 价格 + 网格线 + 买卖点")
    import pandas as pd
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                     gridspec_kw={"height_ratios": [2, 1]},
                                     sharex=True)
    df = utils.cached_kline(code, start, end)
    for g in result.grid_lines:
        ax1.axhline(g, color="grey", linestyle=":", alpha=0.4, linewidth=0.8)
    ax1.plot(df.index, df["close"], color="black", linewidth=1.0, label="收盘价")
    ax1.fill_between(df.index, df["low"], df["high"], color="lightgrey", alpha=0.3, label="H/L")
    tdf = result.trades_df()
    if not tdf.empty:
        buys = tdf[tdf["side"] == "buy"]
        sells = tdf[tdf["side"] == "sell"]
        if not buys.empty:
            ax1.scatter(buys["date"], buys["price"], marker="^", color="red", s=60, label="买入", zorder=5)
        if not sells.empty:
            ax1.scatter(sells["date"], sells["price"], marker="v", color="green", s=60, label="卖出", zorder=5)
    ax1.set_title(f"{code}  {result.name}  网格 {result.cfg.n_grids} 格")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # 权益 + 持仓
    ax2.plot(result.equity.index, result.equity.values, color="navy", linewidth=1.2, label="权益")
    ax2.axhline(cash, color="grey", linestyle="--", linewidth=0.7, label="初始资金")
    ax2.fill_between(result.position_history.index, 0, result.position_history.values,
                     color="steelblue", alpha=0.4, label="持仓")
    ax2.set_title("资金曲线 + 持仓")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 网格效率
    with st.expander("📊 网格效率分析"):
        eff = result.grid_efficiency()
        if not eff.empty:
            utils.show_df(eff, height=300)
        else:
            st.info("无完整往返")

    with st.expander(f"💹 交易明细({len(result.trades)} 笔)"):
        if not tdf.empty:
            utils.show_df(tdf, height=400)
            utils.download_button(tdf, f"{code}_grid_trades.csv")
