"""智能定投 DCA 页面:4 策略对比 + 智能加仓参数。"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from baostock_tool import dca
from baostock_tool.dashboard import utils

st.set_page_config(page_title="智能定投", page_icon="💰", layout="wide")
utils.check_login() or st.stop()

utils.section_header("💰 智能定投 DCA", "4 种策略横向对比 + 智能加仓规则")

code = utils.stock_picker("股票", default="sh.510300",
                          help="建议用 ETF / 宽基指数")
start, end = utils.period_selector(default="3y")

col1, col2, col3 = st.columns(3)
with col1:
    amount = st.number_input("每期投入(元)", value=2000, step=500)
with col2:
    frequency = st.selectbox("频率", ["monthly", "biweekly", "weekly"], index=0)
with col3:
    show_compare = st.checkbox("跑 4 策略对比", value=True)

# 智能加仓参数(只在 smart / dip_buy 时用)
with st.expander("⚙️ 智能加仓参数(可选)", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        ma_window = st.number_input("均线回看天数(smart 用)", value=120, min_value=20)
    with col2:
        dip_threshold = st.number_input("跌幅加仓阈值(dip_buy 用,负数)",
                                          value=-0.05, step=0.01, format="%.2f")

# ===== 单策略 =====
if not show_compare:
    utils.section_header("单策略回测")
    strategy_choice = st.selectbox("策略", ["lump_sum", "pure", "dip_buy", "smart"])

    if st.button("🚀 跑单策略", type="primary", use_container_width=True):
        df_check = utils.cached_kline(code, start, end)
        if df_check.empty:
            st.error(f"❌ {code} 无数据")
            st.stop()
        with st.spinner("回测中..."):
            try:
                result = dca.dca_backtest(
                    code, start, end,
                    amount_per_period=amount, frequency=frequency,
                    strategy=strategy_choice, ma_window=ma_window,
                    dip_threshold=dip_threshold,
                )
            except Exception as e:
                st.error(f"❌ 失败: {e}")
                st.stop()

        st.success("✅ 跑完")
        s = result.summary()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总收益", s["总收益率"])
        c2.metric("年化", s["年化收益"])
        c3.metric("累计投入", s["累计投入"])
        c4.metric("平均成本", s["平均成本"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                         gridspec_kw={"height_ratios": [2, 1]},
                                         sharex=True)
        ax1.plot(result.value_history.index, result.value_history.values,
                 color="navy", linewidth=1.2, label="市值")
        ax1.plot(result.cost_basis_history.index, result.cost_basis_history.values,
                 color="grey", linestyle="--", linewidth=0.8, label="累计投入")
        ax1.fill_between(result.cost_basis_history.index,
                         result.cost_basis_history.values, result.value_history.values,
                         where=result.value_history.values >= result.cost_basis_history.values,
                         color="red", alpha=0.15, label="浮盈")
        ax1.fill_between(result.cost_basis_history.index,
                         result.cost_basis_history.values, result.value_history.values,
                         where=result.value_history.values < result.cost_basis_history.values,
                         color="green", alpha=0.15, label="浮亏")
        ax1.set_title(f"{code}  {strategy_choice}  期末市值 {result.final_value:,.0f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        avg_cost = result.cost_basis_history / result.shares_history.replace(0, float("nan"))
        ax2.plot(df_check.index, df_check["close"].values, color="black", linewidth=0.7,
                 label="收盘价")
        ax2.plot(avg_cost.index, avg_cost.values, color="orange", linewidth=1.0,
                 label="平均成本")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ===== 4 策略对比 =====
if show_compare:
    if st.button("🚀 跑 4 策略对比", type="primary", use_container_width=True):
        df_check = utils.cached_kline(code, start, end)
        if df_check.empty:
            st.error(f"❌ {code} 无数据")
            st.stop()
        with st.spinner("跑 4 策略..."):
            try:
                cmp = dca.compare_strategies(
                    code, start, end,
                    amount_per_period=amount, frequency=frequency,
                    ma_window=ma_window, dip_threshold=dip_threshold,
                )
            except Exception as e:
                st.error(f"❌ 失败: {e}")
                st.stop()

        st.success("✅ 跑完")
        st.markdown("### 📊 4 策略对比")
        # 格式化百分比列
        disp = cmp.copy()
        for col in ["总收益率", "年化收益"]:
            disp[col] = disp[col].apply(lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x)
        disp["平均成本"] = disp["平均成本"].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # 4 策略市值曲线对比
        fig, ax = plt.subplots(figsize=(14, 6))
        for s_name in ["lump_sum", "pure", "dip_buy", "smart"]:
            r = dca.dca_backtest(code, start, end,
                                  amount_per_period=amount, frequency=frequency,
                                  strategy=s_name, ma_window=ma_window,
                                  dip_threshold=dip_threshold)
            ax.plot(r.value_history.index, r.value_history.values,
                    linewidth=1.2, label=f"{s_name}  ({r.total_return*100:.1f}%)")
        ax.set_title(f"{code} 4 策略市值曲线对比")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
