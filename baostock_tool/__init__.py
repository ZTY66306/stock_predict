"""baostock_tool:基于 baostock 的股市预测/量化/查询/回测/筛选工具集。

子模块:
    client     - baostock 登录管理
    data       - 原始数据获取(K线/财报/宏观/成分股)
    data_cache - 本地 K 线缓存
    indicators - 技术指标(MA/MACD/KDJ/RSI/BOLL 等)
    patterns   - K 线形态识别
    strategy   - 交易策略
    backtest   - 回测引擎(风险指标 / 仓位)
    position   - 仓位管理(Kelly / 波动率目标)
    optimizer  - 走步优化 + 网格搜索
    screener   - 选股器
    portfolio  - 组合回测 / 风险平价 / 多空
    quant      - 量化分析(IC/分层/归因/regime)
    predict    - 机器学习预测(可选 XGBoost/LightGBM)
    report     - 可视化与报告
    utils      - 工具函数
    grid_backtest - 网格交易回测(A 股 T+0 / T+1 / 涨跌停 / 整百)
    pairs_trading  - 配对交易(协整 / 价差 / z-score 回测)
    fund_flow  - 资金流 / 北向资金 / 龙虎榜(akshare 集成)
    dca       - 智能定投(纯 / 均线偏离 / 跌多加仓 / 一次性对比)
    market_overview - 涨跌停统计 / 题材热度(akshare 集成)
    paper_trader  - 实盘模拟器(状态持久化 / 日报 / webhook)
"""
from __future__ import annotations

__version__ = "0.2.0"

# 暴露所有子模块
from . import (
    client, data, data_cache, indicators, patterns, strategy,
    backtest, position, optimizer, screener, portfolio,
    quant, predict, report, utils, grid_backtest,
    pairs_trading, fund_flow, dca, market_overview, paper_trader,
)

# 顶层便捷符号
from .predict import (
    AVAILABLE_MODELS, StackingEnsemble, WalkForwardML,
    select_features, cross_sectional_score,
)
from .patterns import list_patterns
from .position import kelly_fraction, volatility_target, fixed_fractional
from .optimizer import WalkForwardOptimizer, grid_search, RollingRobustness
from .portfolio import Portfolio, long_short_backtest, risk_parity_weights
from .screener import SCREEN_TEMPLATES
from .strategy import EnsembleStrategy
from . import grid_backtest as _grid

__all__ = [
    "client", "data", "data_cache", "indicators", "patterns", "strategy",
    "backtest", "position", "optimizer", "screener", "portfolio",
    "quant", "predict", "report", "utils", "grid_backtest",
    "pairs_trading", "fund_flow", "dca", "market_overview", "paper_trader",
    "AVAILABLE_MODELS", "StackingEnsemble", "WalkForwardML",
    "select_features", "cross_sectional_score",
    "list_patterns",
    "kelly_fraction", "volatility_target", "fixed_fractional",
    "WalkForwardOptimizer", "grid_search", "RollingRobustness",
    "Portfolio", "long_short_backtest", "risk_parity_weights",
    "SCREEN_TEMPLATES", "EnsembleStrategy",
    "grid_backtest",
    "__version__",
]
