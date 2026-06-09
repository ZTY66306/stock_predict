"""单股回测页面:8 套策略 + 自定义参数 + 全套风险指标。"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import streamlit as st

from ... import backtest, strategy
from .. import utils

st.set_page_config(page_title="回测", page_icon="🎯", layout="wide")
utils.check_login() or st.stop()

utils.section_header("🎯 单股回测", "8 套经典策略 + 全套风险指标")

# ===== 输入 =====
col1, col2 = st.columns(2)
with col1:
    code = utils.stock_picker("股票", default="sh.600000")
with col2:
    strat = st.selectbox("策略", list(strategy.STRATEGIES.keys()),
                          format_func=lambda k: f"{k}  —  {strategy.STRATEGIES[k].description}")

start, end = utils.period_selector(default="2y")

with st.expander("⚙️ 高级参数", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        cash = st.number_input("初始资金", value=100000.0, step=10000.0)
        commission = st.number_input("手续费率", value=0.0003, step=0.0001, format="%.4f")
        slippage = st.number_input("滑点", value=0.001, step=0.001, format="%.3f")
    with col2:
        stop_loss = st.number_input("止损比例(0=关)", value=0.0, step=0.05, format="%.2f")
        take_profit = st.number_input("止盈比例(0=关)", value=0.0, step=0.05, format="%.2f")
    params_text = st.text_input("策略参数 JSON(覆盖默认)",
                                 value=json.dumps(strategy.STRATEGIES[strat].default_params))
    try:
        params = json.loads(params_text) if params_text.strip() else None
    except Exception as e:
        st.error(f"参数 JSON 解析失败: {e}")
        params = None

if not code:
    st.info("👆 选股票开始")
    st.stop()

if st.button("🚀 跑回测", type="primary", use_container_width=True):
    df = utils.cached_kline(code, start, end)
    if df.empty:
        st.error(f"❌ {code} 无数据")
        st.stop()

    with st.spinner("跑策略 + 回测中..."):
        try:
            sig = strategy.run_strategy(strat, df, params)
            cfg = backtest.BacktestConfig(
                initial_cash=cash, commission=commission, slippage=slippage,
                stop_loss=stop_loss or None, take_profit=take_profit or None,
            )
            result = backtest.BacktestEngine(cfg).run(df, sig)
        except Exception as e:
            st.error(f"❌ 回测失败: {e}")
            st.stop()

    # ===== 关键指标 =====
    st.success("✅ 回测完成")
    metrics = result.risk_metrics()
    cols = st.columns(5)
    cols[0].metric("总收益", f"{metrics['总收益率']*100:.2f}%")
    cols[1].metric("年化", f"{metrics['年化收益']*100:.2f}%")
    cols[2].metric("夏普", f"{metrics['夏普比率']:.2f}")
    cols[3].metric("最大回撤", f"{metrics['最大回撤']*100:.2f}%")
    cols[4].metric("胜率", f"{metrics['胜率']*100:.1f}%")

    # 权益曲线 + 回撤
    utils.section_header("📈 权益曲线")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                     gridspec_kw={"height_ratios": [2, 1]},
                                     sharex=True)
    ax1.plot(result.equity.index, result.equity.values, color="navy", linewidth=1.2)
    ax1.axhline(cash, color="grey", linestyle="--", linewidth=0.7, label="初始资金")
    ax1.set_title(f"{code}  {strat}  最终: {result.equity.iloc[-1]:,.0f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    peak = result.equity.cummax()
    dd = (result.equity / peak - 1) * 100
    ax2.fill_between(result.equity.index, dd, 0, color="red", alpha=0.4)
    ax2.set_title("回撤 (%)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 完整指标表
    with st.expander("📋 全部风险指标"):
        utils.show_df(metrics.to_frame("值"), height=400)

    # 交易明细
    with st.expander(f"💹 交易明细({len(result.trades)} 笔)"):
        if result.trades:
            utils.show_df(result.trades_df(), height=400)
            utils.download_button(result.trades_df(), f"{code}_trades.csv")
        else:
            st.info("无交易")
