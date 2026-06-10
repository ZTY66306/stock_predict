"""Dashboard 主入口。

启动方式 3 种:
    1) streamlit run baostock_tool/dashboard/app.py
    2) baostock-tool-dashboard  (装好后,见 pyproject [project.scripts])
    3) python -m streamlit run baostock_tool/dashboard/app.py

启动后,浏览器打开 http://localhost:8501。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd

from baostock_tool.dashboard import utils


def main() -> None:
    """供 `baostock-tool-dashboard` 脚本入口调用:转交给 streamlit。"""
    here = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(here, "app.py")
    # 用同一 Python 解释器 -m streamlit run 启动
    os.execvp(sys.executable, [sys.executable, "-m", "streamlit", "run", app_path,
                                "--browser.gatherUsageStats", "false"])


# ============ 以下是真实的 Streamlit 页面代码 ============
# (被 `streamlit run app.py` 时执行;被 `python app.py` 走 main 路径)

import streamlit as st

# 全局页面配置
st.set_page_config(
    page_title="A 股量化研究看板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": """
        # baostock_tool Web 看板

        基于 baostock + akshare 的 A 股量化研究工具集。

        包含:K 线 / 指标 / 选股 / 回测 / 网格复盘 / 配对 / 多策略融合 /
        稳健性 / DCA / 资金流 / 涨跌停 / 实盘模拟 等 20+ 工具。

        版本:见 `baostock_tool.__version__`
        """,
    },
)


# ============ 首页(主入口本身就是首页) ============

utils.section_header(
    "📊 baostock_tool 看板",
    "A 股量化研究一站式 Web UI · 基于 Streamlit",
)

# 登录状态
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if utils.check_login():
        st.success("✅ baostock 已登录")
    else:
        st.stop()
with col2:
    from baostock_tool import fund_flow as ff, market_overview as mo
    if ff._HAS_AKSHARE:
        st.success("✅ akshare 已装(资金流/涨跌停可用)")
    else:
        st.warning("⚠️ akshare 未装(资金流/涨跌停不可用)")

st.divider()

# 快速入口
st.markdown("### 🚀 快速入口")
st.markdown("左侧栏是完整功能列表,常用入口:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.page_link("pages/2_kline.py", label="📈 K 线 + 指标", icon="📈")
    st.caption("看历史 K 线、加技术指标、识别形态")
with col2:
    st.page_link("pages/4_backtest.py", label="🎯 回测", icon="🎯")
    st.caption("选策略 + 参数 + 风险指标 + 出图")
with col3:
    st.page_link("pages/5_grid.py", label="🕸️ 网格复盘", icon="🕸️")
    st.caption("T+0/T+1 网格交易回测,适合 ETF / 可转债")
with col4:
    st.page_link("pages/7_dca.py", label="💰 智能定投", icon="💰")
    st.caption("4 种定投策略横向对比")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.page_link("pages/3_screener.py", label="🔍 选股", icon="🔍")
    st.caption("16 个内置模板 + 自定义条件")
with col2:
    st.page_link("pages/6_ensemble.py", label="🧩 多策略融合", icon="🧩")
    st.caption("投票 / 加权 / Veto + 稳健性")
with col3:
    st.page_link("pages/8_market_fund.py", label="📡 资金流/涨跌停", icon="📡")
    st.caption("北向 + 龙虎榜 + 板块涨停")
with col4:
    st.page_link("pages/9_paper.py", label="📊 实盘模拟", icon="📊")
    st.caption("虚拟持仓 / 状态持久化 / webhook")

st.divider()

# 项目信息
with st.expander("ℹ️ 关于本看板", expanded=False):
    st.markdown(f"""
    - **包名:** `baostock_tool`
    - **本次启动:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    - **数据源:** baostock(A 股 K 线/财报/宏观)+ akshare(资金流/涨跌停)
    - **缓存策略:** K 线 1 小时 / 资金流 5 分钟 / 元信息 1 天
    - **CLI 等价命令:** 所有 dashboard 页面底层都是调 `baostock_tool.*` 模块,
      你在终端跑 `python -m baostock_tool.cli <sub>` 也能达到同样效果。

    **常见问题:**
    - Q: 点击按钮后没反应? A: 看页面顶部有没有红色错误条,通常是 baostock / 网络问题
    - Q: 想清缓存? A: 按 `C` 然后选 "Clear cache",或重启 dashboard
    - Q: 想加新页面? A: 在 `baostock_tool/dashboard/pages/` 加 `NN_xxx.py` 即可(Streamlit 自动发现)
    """)

# 模块清单
st.markdown("### 📦 已封装模块")
mods = [
    ("data", "数据获取(K 线/财报/宏观/成分股)"),
    ("indicators", "技术指标(MA/MACD/KDJ/RSI/BOLL...)"),
    ("patterns", "K 线形态识别"),
    ("strategy", "交易策略(8 套经典)"),
    ("backtest", "回测引擎 + 风险指标"),
    ("position", "仓位管理(Kelly/波动率目标)"),
    ("optimizer", "走步优化 + 滚动稳健性"),
    ("screener", "选股器(16 模板)"),
    ("portfolio", "组合回测 + 风险平价"),
    ("quant", "IC / 分层 / Brinson 归因 / Regime"),
    ("predict", "ML 预测(XGBoost / LightGBM 可选)"),
    ("report", "可视化与 HTML 报告"),
    ("grid_backtest", "网格交易(T+0/T+1)"),
    ("pairs_trading", "配对交易(协整/价差/z-score)"),
    ("fund_flow", "资金流/北向/龙虎榜(akshare)"),
    ("dca", "智能定投(4 策略对比)"),
    ("market_overview", "涨跌停统计/题材热度(akshare)"),
    ("paper_trader", "实盘模拟器(状态持久化/webhook)"),
]
df_mods = pd.DataFrame(mods, columns=["模块", "能力"])
st.dataframe(df_mods, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
