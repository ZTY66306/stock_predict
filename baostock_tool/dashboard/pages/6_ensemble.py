"""多策略融合 + 滚动稳健性页面。"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import streamlit as st

from baostock_tool import backtest, optimizer, strategy
from baostock_tool.dashboard import utils

st.set_page_config(page_title="多策略融合", page_icon="🧩", layout="wide")
utils.check_login() or st.stop()

# ===== 选项卡 =====
tab1, tab2 = st.tabs(["🧩 多策略融合", "🛡️ 滚动稳健性"])

# ============ 多策略融合 ============
with tab1:
    utils.section_header("🧩 多策略融合(EnsembleStrategy)",
                          "把多套策略的信号按 weighted / majority / veto 规则融合")

    col1, col2 = st.columns(2)
    with col1:
        code = utils.stock_picker("股票", default="sh.600000")
    with col2:
        voting = st.selectbox("投票模式", ["weighted", "majority", "veto"],
                                format_func=lambda x: {
                                    "weighted": "加权求和",
                                    "majority": "多数投票",
                                    "veto": "Veto(更保守)"
                                }[x])

    chosen = st.multiselect("选策略", list(strategy.STRATEGIES.keys()),
                              default=["ma_cross", "macd", "kdj", "rsi_oversold"])
    if not chosen:
        st.warning("至少选 1 套策略")
        st.stop()

    # 权重
    weights = None
    if voting == "weighted" and len(chosen) > 1:
        st.markdown("**策略权重(自动归一化,可不填):**")
        ws = []
        cols = st.columns(len(chosen))
        for i, (c, col) in enumerate(zip(chosen, cols)):
            with col:
                w = st.number_input(c, value=1.0, step=0.1, key=f"w_{c}_{voting}")
                ws.append(w)
        weights = json.dumps(ws)

    threshold = 0.3
    majority_min = 2
    if voting == "weighted":
        threshold = st.slider("入场阈值(weighted)", 0.0, 1.0, 0.3, 0.05)
    else:
        majority_min = st.slider("最少同意数(majority/veto)", 1, len(chosen), min(2, len(chosen)))

    start, end = utils.period_selector(default="2y")

    if st.button("🚀 跑融合", type="primary", width="stretch"):
        df = utils.cached_kline(code, start, end)
        if df.empty:
            st.error(f"❌ {code} 无数据")
            st.stop()
        with st.spinner("跑融合中..."):
            try:
                w = json.loads(weights) if weights else None
                ens = strategy.EnsembleStrategy(chosen, weights=w, voting=voting,
                                                  entry_threshold=threshold,
                                                  majority_min=majority_min)
                sig = ens.run(df)
                cfg = backtest.BacktestConfig(initial_cash=100000)
                result = backtest.BacktestEngine(cfg).run(df, sig)
            except Exception as e:
                st.error(f"❌ 失败: {e}")
                st.stop()

        st.success("✅ 跑完")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总收益", f"{result.total_return*100:.2f}%")
        c2.metric("夏普", f"{result.sharpe:.2f}")
        c3.metric("最大回撤", f"{result.max_drawdown*100:.2f}%")
        c4.metric("交易", len(result.trades))

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(result.equity.index, result.equity.values, color="navy", linewidth=1.2)
        ax.axhline(100000, color="grey", linestyle="--", linewidth=0.7, label="初始资金")
        ax.set_title(f"{code}  Ensemble({voting})  {len(chosen)} 策略")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("📋 完整指标"):
            utils.show_df(result.risk_metrics().to_frame("值"))

# ============ 滚动稳健性 ============
with tab2:
    utils.section_header("🛡️ 滚动稳健性测试(RollingRobustness)",
                          "同一组参数在多个滑动窗口上跑,看表现是否一致")

    col1, col2 = st.columns(2)
    with col1:
        code2 = utils.stock_picker("股票", default="sh.600000", key="rob_code")
    with col2:
        strat2 = st.selectbox("策略", list(strategy.STRATEGIES.keys()),
                                format_func=lambda k: f"{k}  —  {strategy.STRATEGIES[k].description}",
                                key="rob_strat")

    col1, col2, col3 = st.columns(3)
    with col1:
        window = st.number_input("窗口大小(交易日)", value=252, min_value=60, step=20)
    with col2:
        step = st.number_input("滑动步长", value=63, min_value=10, step=10)
    with col3:
        params_text = st.text_input("策略参数 JSON",
                                     value=json.dumps(strategy.STRATEGIES[strat2].default_params),
                                     key="rob_params")
    try:
        params = json.loads(params_text) if params_text.strip() else {}
    except Exception as e:
        st.error(f"参数 JSON 解析失败: {e}")
        params = {}

    start, end = utils.period_selector(default="3y", key="rob_period")

    if st.button("🚀 跑稳健性", type="primary", width="stretch"):
        df = utils.cached_kline(code2, start, end)
        if df.empty:
            st.error(f"❌ {code2} 无数据")
            st.stop()
        with st.spinner(f"跑 {len(df)} 天数据上多 fold 稳健性..."):
            try:
                rr = optimizer.RollingRobustness(strat2, params=params,
                                                   window=window, step=step).run(df)
            except Exception as e:
                st.error(f"❌ 失败: {e}")
                st.stop()

        st.success("✅ 跑完")
        s = rr.summary()
        utils.show_df(s.to_frame("值"), height=300)

        # 稳健性评分
        score = rr.robustness_score()
        st.markdown("### 🎯 稳健性评分(0~1,越大越稳)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("盈利 fold 占比", f"{score['profit_rate']:.2%}")
        c2.metric("夏普稳定性", f"{score['stability']:.2%}")
        c3.metric("回撤温和度", f"{score['dd_score']:.2%}")
        c4.metric("**总评分**", f"**{score['overall']:.2%}**")

        # folds 明细
        st.markdown("### 📊 各 fold 明细")
        fdf = rr.folds_df()
        st.dataframe(fdf, width="stretch", height=400)
        utils.download_button(fdf, f"{code2}_robustness_folds.csv")
