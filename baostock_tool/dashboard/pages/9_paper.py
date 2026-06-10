"""实盘模拟器页面:多股多策略 + 状态持久化 + webhook。"""
from __future__ import annotations

import json
import os

import streamlit as st

from baostock_tool import paper_trader as ptr
from baostock_tool.dashboard import utils

st.set_page_config(page_title="实盘模拟", page_icon="📊", layout="wide")
utils.check_login() or st.stop()

utils.section_header("📊 实盘模拟器(PaperTrader)",
                      "把策略接到每日信号,跟踪虚拟持仓,状态 JSON 持久化")

# ===== 配置 =====
col1, col2 = st.columns(2)
with col1:
    codes_input = st.text_input("股票代码(逗号分隔)", value="sh.600000, sh.510300")
with col2:
    strat = st.selectbox("共用策略", list(["ma_cross", "macd", "kdj", "rsi_oversold",
                                                "boll", "turtle", "momentum", "mean_reversion"]),
                          index=0)

col1, col2, col3 = st.columns(3)
with col1:
    cash = st.number_input("初始资金", value=200000.0, step=10000.0)
with col2:
    position_pct = st.slider("单次用 % 资金", 0.5, 1.0, 0.95, 0.05)
with col3:
    date = st.date_input("运行日期", value=__import__("datetime").date.today())

col1, col2 = st.columns(2)
with col1:
    state_path = st.text_input("状态文件", value="./paper_state.json")
with col2:
    webhook = st.text_input("Webhook URL(可选)", value="")

# ===== 跑 =====
if st.button("🚀 跑一次模拟", type="primary", use_container_width=True):
    codes = [c.strip() for c in codes_input.split(",") if c.strip()]
    if not codes:
        st.warning("请输入至少 1 个股票代码")
        st.stop()
    strategies = {c: strat for c in codes}
    try:
        trader = ptr.PaperTrader(
            strategies=strategies, initial_cash=cash,
            state_path=state_path if state_path else None,
            webhook_url=webhook if webhook else None,
            position_size_pct=position_pct,
        )
        report = trader.run_once(date=str(date))
        trader.save_state()
    except Exception as e:
        st.error(f"❌ 失败: {e}")
        st.stop()

    st.success(f"✅ {date} 跑完")
    s = report.summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总权益", s["总权益"])
    c2.metric("盈亏比", s["盈亏比"])
    c3.metric("持仓数", s["持仓数"])
    c4.metric("信号数", s["信号数"])

    # 信号
    st.markdown("### 🎯 今日信号")
    sig_df = report.signals_df()
    if not sig_df.empty:
        active = sig_df[sig_df["signal"] != 0]
        if not active.empty:
            st.dataframe(active, use_container_width=True)
        else:
            st.info("无触发信号")

    # 持仓
    st.markdown("### 💼 当前持仓")
    pos_df = report.positions_df()
    pos_active = pos_df[pos_df["shares"] > 0]
    if not pos_active.empty:
        pos_disp = pos_active[["code", "shares", "avg_cost", "last_price",
                                 "cost_basis", "realized_pnl"]].copy()
        # 算浮动盈亏
        pos_disp["unrealized_pnl"] = (pos_disp["last_price"] - pos_disp["avg_cost"]) * pos_disp["shares"]
        st.dataframe(pos_disp, use_container_width=True)
    else:
        st.info("无持仓")

    # 日志
    if report.log:
        with st.expander("📋 运行日志"):
            for line in report.log:
                st.text(line)

    # 保存
    if state_path and os.path.exists(state_path):
        with open(state_path, "rb") as f:
            st.download_button("📥 下载状态文件", f.read(), "paper_state.json",
                                "application/json", use_container_width=True)

# ===== 历史 =====
if state_path and os.path.exists(state_path):
    st.divider()
    st.markdown("### 📜 历史运行")
    try:
        with open(state_path) as f:
            state = json.load(f)
        st.json({
            "上次运行": state.get("last_run"),
            "当前 cash": state.get("cash"),
            "持仓数": sum(1 for p in state.get("positions", {}).values()
                          if p.get("shares", 0) > 0),
        })
        hist = state.get("history", [])
        if hist:
            st.dataframe(hist, use_container_width=True)
    except Exception as e:
        st.warning(f"读状态文件失败: {e}")
