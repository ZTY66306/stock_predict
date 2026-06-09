# baostock_tool

> 基于 [baostock](http://baostock.com) 的 **A 股综合量化工具集**:数据获取 / 技术指标 / 选股 / 策略 / 回测 / 仓位 / 组合 / 量化研究 / 机器学习 / 报告可视化,纯 Python,既可作为库调用,也自带 16 个 CLI 子命令。

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-beta-orange)

---

## 目录

- [项目定位](#项目定位)
- [功能概览](#功能概览)
- [安装](#安装)
- [5 分钟上手](#5-分钟上手)
- [CLI 速查](#cli-速查)
- [选股模板](#选股模板)
- [自定义选股 / 策略](#自定义选股--策略)
- [仓位管理](#仓位管理)
- [组合回测](#组合回测)
- [Walk-Forward 优化](#walk-forward-优化)
- [量化研究(IC / 分层 / Regime)](#量化研究ic--分层--regime)
- [K 线形态识别](#k-线形态识别)
- [ML 预测](#ml-预测)
- [网格交易回测](#网格交易回测) ⬅️ 新增
- [数据缓存](#数据缓存)
- [项目结构](#项目结构)
- [测试](#测试)
- [维护本指南](#维护本指南) ⬅️ 项目更新时同步这里
- [注意事项](#注意事项)
- [License](#license)

---

## 项目定位

`baostock_tool` 是对官方 [baostock](http://baostock.com) SDK 的"业务层封装",把常见 A 股量化研究流程抽成可直接 `import` 的模块,**避免每次新项目都重复造轮子**。它适合:

- 想用纯本地 Python 复现论文 / 研报策略的同学;
- 做日内/日级回测,需要 Sortino / Calmar / VaR / CVaR / Alpha / Beta / IR 等**全套风险指标**;
- 想一站式打通 **数据 → 指标 → 选股 → 策略 → 回测 → 报告** 的研究流水线;
- 需要把研究脚本包装成 **CLI**,扔到服务器/定时任务里跑。

不适合:

- 实时行情(baostock 已停用,本工具专注历史数据);
- 港股 / 美股(目前只覆盖 A 股 + 申万行业 + 沪深 300 / 上证 50 / 中证 500 成分股)。

---

## 功能概览

| 模块 | 能力要点 |
| --- | --- |
| `data` | K 线 / 复权 / 申万行业 / 指数成分 / 交易日历 / **三大财报 + 业绩快报 + 业绩预告 + 前十大股东** / **复权因子 / 分红 / 估值 / 成长 / 运营** / 宏观 / 利率 / 货币供应 / 基准指数 / 个股元信息 |
| `data_cache` | 本地 **CSV 缓存层**,按 `code × freq × adjflag` 切分,自动合并增量、避免重复请求 |
| `indicators` | MA / EMA / SMA(中式) / MACD / KDJ / RSI / BOLL / ATR / CCI / OBV / 收益 / 波动 / **DMI/ADX / Williams %R / ROC / MFI / TRIX / SAR / BIAS**,支持 `add_all` 一次性加全套 |
| `patterns` | 吞没 / 十字星 / 锤头 / 上吊线 / 光头光脚(阳/阴) / 刺穿 / 乌云盖顶 / 红三兵 / 三只乌鸦 / 早晨之星 / 黄昏之星 / **多形态打分** |
| `screener` | **16 个内置模板** + 自定义条件组合(技术面 + 形态 + 量价 + 基本面) + `daily_pick` 流水线 |
| `strategy` | 8 套经典策略(MA / MACD / KDJ / BOLL / **海龟 / 动量 / 均值回归 / RSI 超卖**) |
| `backtest` | 事件驱动回测引擎 + 等权组合回测(手续费 / 滑点 / 印花税 / **最低 5 元** / 整百 lot / 止损止盈) + **Sortino / Calmar / VaR / CVaR / Alpha / Beta / IR / 月度热力** |
| `position` | 凯利公式 / 波动率目标 / 固定风险法 / 回撤风控仓位 + 统一 `PositionSizer` 接口 |
| `optimizer` | **网格搜索 + Walk-Forward 走步优化** + OOS 聚合指标 |
| `portfolio` | **等权 / 风险平价 / 信号驱动 / 多空轮动** 组合回测 + 调仓日历 |
| `quant` | 因子 IC / 分层回测 / 相关性 / **Brinson 归因 / Regime 检测** |
| `predict` | sklearn 涨跌分类 / 收益回归 / **Stacking 集成 / 特征选择 / 横截面打分 / 走步 ML**;`kind="xgb"` / `kind="lgbm"` **可选** |
| `report` | 收益曲线 / 回撤 / K线信号 / IC 分布 / **月度热力 / 滚动夏普 / 持仓时间线 / 蜡烛图 / 水下回撤** / 文本与 **jinja2 HTML 报告** |
| `grid_backtest` | **网格交易复盘**:支持股票代码/中文名输入、A 股规则识别(T+1/整百/涨跌停)、等差/等比网格、手动/自动区间、底仓/止损/止盈、网格效率分析 |
| `cli` | **17 个子命令**,一行搞定数据 / 选股 / 回测 / 风险 / 优化 / 预测 / **网格复盘** / 缓存管理 |

---

## 安装

**生产环境(发布后):**

```bash
pip install baostock-tool
```

**开发模式:**

```bash
git clone <repo> && cd baostock-tool
pip install -e ".[ml,dev]"
```

可选 ML 加速 — 装上后 `predict.DirectionClassifier(kind="xgb")` / `kind="lgbm"` 可用:

```bash
pip install -e ".[ml]"
```

依赖详见 `pyproject.toml`(核心:`baostock`、`pandas`、`numpy`、`matplotlib`、`scipy`、`scikit-learn>=1.2`、`tqdm`、`jinja2`)。

> **登录提示:** 大部分数据接口要求先登录 baostock。脚本里 `client.ensure_login()` 会自动处理;CLI 里跑一次 `python -m baostock_tool.cli login` 即可,登录态是进程级的。

---

## 5 分钟上手

```python
from baostock_tool import data, indicators, strategy, backtest, report

# 1) 拉数据(自动走本地缓存)
df = data.get_kline("sh.600000", "2024-01-01", "2025-12-31")

# 2) 加全套指标
df = indicators.add_all(df)

# 3) 跑策略
sig = strategy.run_strategy("ma_cross", df, {"short": 5, "long": 20})

# 4) 回测 + 全套风险指标
result = backtest.BacktestEngine().run(df, sig)
print(result.risk_metrics())    # 总收益 / 年化 / 夏普 / Sortino / Calmar / VaR / CVaR / 胜率 / 盈亏比

# 5) 出报告
report.plot_equity(result, save_path="equity.png")
report.plot_monthly_heatmap(result.daily_returns, save_path="heatmap.png")
report.write_text_report(result, "report.txt", "sh.600000", "ma_cross")
```

> 也可以用 `pip install` 时生成的命令行二进制 `baostock-tool`,例如 `baostock-tool login`、`baostock-tool kline sh.600000 ...`,等价于 `python -m baostock_tool.cli ...`。

---

## CLI 速查

CLI 共 **17 个子命令**:`login / logout / industry / kline / constituents / screen / backtest / predict / quant / risk / optimize / pattern / cache / info / macro / ranking / grid`。

```bash
# ======== 基础数据 ========
python -m baostock_tool.cli login
python -m baostock_tool.cli kline sh.600000 --start 2024-01-01 --end 2025-12-31 --n 20
python -m baostock_tool.cli constituents hs300 --date 2025-12-01
python -m baostock_tool.cli industry
python -m baostock_tool.cli info sh.600000 --industry
python -m baostock_tool.cli macro
python -m baostock_tool.cli cache info
python -m baostock_tool.cli cache clear

# ======== 选股(CLI 当前支持 9 个模板,其余模板请用 screener.screen(...) 见"选股模板") ========
python -m baostock_tool.cli screen macd_golden --limit 20
python -m baostock_tool.cli screen kdj_oversold --limit 20
python -m baostock_tool.cli screen volume_breakout --limit 30

# ======== 回测 + 报告 ========
python -m baostock_tool.cli backtest sh.600000 --strategy ma_cross \
    --start 2023-01-01 --stop-loss 0.08 --take-profit 0.30 --report ./output

# ======== 风险指标 + 基准 ========
python -m baostock_tool.cli risk sh.600000 --strategy ma_cross --benchmark sh.000300

# ======== ML 预测 ========
python -m baostock_tool.cli predict sh.600000 --model clf --horizon 1
python -m baostock_tool.cli predict sh.600000 --model reg --horizon 5

# ======== 横截面打分排序 ========
python -m baostock_tool.cli ranking --index hs300 --top 20 --save ./scores.csv

# ======== Walk-Forward 优化 ========
python -m baostock_tool.cli optimize sh.600000 --strategy ma_cross \
    --walk-forward --grid '{"short":[3,5,8],"long":[10,20,30]}' --report ./output

# ======== 形态识别 ========
python -m baostock_tool.cli pattern sh.600000 --patterns "morning_star,engulfing_bullish"

# ======== 因子分析 ========
python -m baostock_tool.cli quant --index hs300 --lookback 180 --report ./output

# ======== 网格交易复盘(支持代码或中文名,如"浦发银行" / "sh.600000") ========
python -m baostock_tool.cli grid sh.600000 \
    --start 2022-01-01 --end 2024-12-31 \
    --mode geometric --grids 10 --lower 9.5 --upper 11.0 \
    --shares 200 --report ./output/grid_sh600000

python -m baostock_tool.cli grid 浦发银行 \
    --start 2022-01-01 --end 2024-12-31 \
    --mode geometric --grids 15 --lower-pct 0.85 --upper-pct 1.15 \
    --lookback 60 --base 1000 --stop-loss 0.08 --take-profit 0.30 \
    --report ./output/grid_pf
```

> **小贴士:** CLI 子命令的 `--grid`、`--params` 接受 JSON 字符串;`--report <dir>` 会自动把文本报告、收益曲线、回撤图、K线图等写到该目录(默认是 `output/`)。

---

## 选股模板

`screener.SCREEN_TEMPLATES` 定义了 **16 个内置模板**。通过 `python -m baostock_tool.cli screen` 调用的模板是其子集(目前 CLI 的 `choices` 暴露了 9 个),其余请用 `screener.screen(...)` 调用:

| 模板 | 含义 | CLI `screen` |
| --- | --- | :---: |
| `low_pe` | 0 < PE_TTM < 20 | ✅ |
| `high_pb` | PB < 1 | ✅ |
| `macd_golden` | MACD 金叉 | ✅ |
| `kdj_oversold` | KDJ 超卖 | ✅ |
| `rsi_oversold` | RSI < 30 | ✅ |
| `rsi_neutral` | RSI 中性区(40–60) | ❌ |
| `volume_breakout` | 量能突破(量 ≥ 20 日均量 × 2) | ✅ |
| `new_high_20` | 20 日新高 | ✅ |
| `low_pe_high_turnover` | 低 PE(0–25) + 涨幅 2%–20% | ❌ |
| `high_turnover_breakout` | 高换手(80% 分位以上) + 量能放大(1.5×) | ❌ |
| `bullish_trend` | 多头排列(MA5>MA10>MA20>MA60) | ✅ |
| `momentum_value` | 动量(20 日 > 5%) + 低估值(PE<30) | ❌ |
| `pattern_bullish_reversal` | 看涨形态共振(吞没 / 锤头 / 早晨之星 等 ≥ 2 个) | ❌ |
| `pattern_morning_star` | 早晨之星形态 | ❌ |
| `pattern_engulfing` | 看涨吞没形态 | ❌ |
| `bbi_bullish` | BBI(MA3+6+12+24)/4 上穿 | ❌ |

> 上表里标 ❌ 的模板仍可在 Python 里通过 `from baostock_tool.screener import screen; screen("pattern_bullish_reversal", date="2025-06-01")` 使用;要全部 16 个模板在 CLI 里可用,把 `cli.py` 的 `cmd_screen` `choices` 列表扩到 `SCREEN_TEMPLATES.keys()` 即可。

---

## 自定义选股 / 策略

**自定义选股:**

```python
from baostock_tool import screener

s = screener.Screen(date="2025-06-01", lookback_days=120)
s.add(screener.MA5_gt_MA20)
s.add(screener.macd_golden_cross)
s.add(screener.pe_between(0, 30))
s.add(screener.pattern("morning_star"))           # K 线形态
s.add(screener.pattern_bullish_score(2))           # 至少 2 个看涨形态共振
picks = s.run(limit=20)
```

**自定义策略:**

```python
from baostock_tool import backtest

def my_strategy(df, params):
    import numpy as np, pandas as pd
    n = params.get("n", 10)
    ma = df["close"].rolling(n).mean()
    sig = pd.DataFrame(index=df.index)
    sig["signal"] = np.where(df["close"] > ma, 1, 0)
    sig["position"] = (df["close"] > ma).astype(int)
    return sig

df = ...   # K 线
sig = my_strategy(df, {"n": 10})
result = backtest.BacktestEngine().run(df, sig)
print(result.risk_metrics())     # 全套风险与绩效指标
```

---

## 仓位管理

```python
from baostock_tool import position

# 凯利公式(默认 cap=0.25,防极端值)
f = position.kelly_fraction(win_rate=0.6, win_loss_ratio=2.0)         # → 0.25

# 波动率目标仓位(自动滞后一日,避免未来函数)
weights = position.volatility_target(returns, target_vol=0.15, lookback=60)

# 固定风险法
size = position.fixed_fractional(capital=100000, risk_per_trade=0.02, stop_loss_pct=0.05)

# 统一接口(method ∈ {'all-in','kelly','vol-target','fixed-frac','dd-guard'})
sizer = position.PositionSizer(method="kelly", win_rate=0.55, win_loss_ratio=1.5)
w = sizer.size({"capital": 100000})
```

---

## 组合回测

```python
from baostock_tool import portfolio

# 等权 + 周一调仓
p = portfolio.Portfolio(prices, rebalance_freq="W-MON").run()

# 风险平价权重
cov = prices.pct_change().cov() * 252
w = portfolio.risk_parity_weights(cov)
p_rp = portfolio.Portfolio(prices, rebalance_freq="ME",
                            weight_fn=lambda dt, pr: w).run()

# 多空动量:每期 top_k/2 做多、top_k/2 做空,等权
signal = prices.pct_change(20)
p_ls = portfolio.long_short_backtest(prices, signal, top_k=4, rebalance_freq="W-MON")
```

---

## Walk-Forward 优化

```python
from baostock_tool import optimizer

# 简单网格搜索
df_gs = optimizer.grid_search(df, "ma_cross",
                              param_grid={"short": [3, 5, 8], "long": [10, 20, 30]},
                              metric="calmar_ratio")

# 走步优化(防过拟合,聚合 OOS 风险指标)
opt = optimizer.WalkForwardOptimizer(
    strategy_name="ma_cross",
    param_grid={"short": [3, 5, 8], "long": [10, 20, 30]},
    n_splits=5, metric="calmar_ratio",
)
res = opt.run(df)
print(res.summary())              # 每 fold 训练/验证得分 + 最优参数
print(res.aggregate_metrics)      # 拼接 OOS 的全套风险指标
```

---

## 量化研究(IC / 分层 / Regime)

```python
from baostock_tool import quant
ic = quant.factor_ic(factor_df, fwd_ret_df, method="spearman")
print(quant.ic_summary(ic))                       # IC_mean / IC_std / IC_IR / t 统计
layered = quant.layered_backtest(factor_df, fwd_ret_df, q=5)
print(layered)

# Regime 检测(简单滚动 z-score 划分 bull/bear/side)
regimes = quant.regime_detection(returns_series, n=20)
```

---

## K 线形态识别

```python
from baostock_tool import data, patterns as ptn

df = data.get_kline("sh.600000", "2024-01-01", "2025-12-31")
print(ptn.list_patterns())                          # 所有形态

# 命中某形态
hits = df.index[ptn.detect(df, "morning_star")]

# 多形态看涨打分(≥ 2 可视为较强信号)
score = ptn.pattern_score(df, bullish=True)
```

支持的形态(共 12 个):`doji / hammer / hanging_man / marubozu_bullish / marubozu_bearish / engulfing_bullish / engulfing_bearish / piercing / dark_cloud_cover / three_white_soldiers / three_black_crows / morning_star / evening_star`。

---

## ML 预测

```python
from baostock_tool import data, predict

df = data.get_kline("sh.600000", "2022-01-01", "2025-12-31")

# 涨跌分类(sklearn GBDT)
clf = predict.DirectionClassifier(kind="gbdt")
metrics = clf.fit_walk_forward(df, horizon=1)
print(metrics.summary())
print(predict.predict_next(df, clf))

# 收益回归
reg = predict.PriceRegressor(kind="gbdt")
reg_metrics = reg.fit_walk_forward(df, horizon=5, target="return")

# 可选: 装上 xgboost 后 DirectionClassifier(kind="xgb") / lightgbm 后 kind="lgbm"
# Stacking 集成
ens = predict.StackingEnsemble(task="clf", base_kinds=("gbdt", "rf"))
ens.fit(X, y)

# 特征选择
X_sel, names = predict.select_features(X, y, k=20, method="mutual_info")

# 横截面打分
scores = predict.cross_sectional_score(
    codes, start, end,
    model_factory=lambda: predict.DirectionClassifier(kind="gbdt"),
    horizon=5, target="direction",
)
```

`predict.AVAILABLE_MODELS` 默认 `("gbdt", "rf", "logistic", "ridge")`,装了 xgboost / lightgbm 后追加 `("xgb",)` / `("lgbm",)`。

---

## 网格交易回测

为 A 股量身定制:支持**代码或中文名**输入,自动识别交易规则(**T+0 / T+1** / 整百 / 涨跌停 / 印花税),输出完整交易明细、网格效率、收益曲线与文本报告。

**支持的交易制度与涨跌幅(自动识别):**

| 代码段 | 制度 | 涨跌幅 |
| --- | --- | --- |
| 沪深主板 / 中小板 (`sh.6/9`, `sz.0`) | T+1 | 10% |
| 科创板 (`sh.688`) / 创业板 (`sz.30`) | T+1 | 20% |
| 北交所 (`bj.`) | T+1 | 30% |
| ST / *ST | T+1 | 5% |
| 沪深 ETF / LOF (`sh.5`, `sz.1`) | **T+0** | 10% |
| 可转债 (`sh.110/113`, `sz.123`) | **T+0** | 30% (熔断机制) |
| 国债 / 地方债 (`sh.01/02/10`, `sz.10`) | **T+0** | 无 |

> T+0 时,当日买入的份额**立即可卖** — 这是 ETF / 可转债网格能高频"做差价"的关键。可以用 `t0={None,True,False}` 强制指定,默认自动识别。

**支持的网格参数:**

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `grid_mode` | `fixed`(等差) / `geometric`(等比) | geometric |
| `n_grids` | 网格数 | 10 |
| `lower` / `upper` | 手动下/上沿(价) | — |
| `lower_pct` / `upper_pct` | 自动:`lower = lookback 期最低 × lower_pct` | — |
| `lookback` | 自动区间用回看窗口(交易日) | 60 |
| `shares_per_grid` | 每格成交股数(自动整百对齐) | 100 |
| `base_position` | 初始底仓(股) | 0 |
| `stop_loss` / `take_profit` | 持仓回撤止损 / 浮盈止盈(相对均价) | None |
| `commission` / `stamp_tax` / `slippage` / `min_commission` | 成本 | 0.0003 / 0.001 / 0.001 / 5 |
| `price_limit_override` | 手动指定涨跌幅,跳过代码识别 | None |
| `t0` | `None`=自动识别 / `True`=强制 T+0 / `False`=强制 T+1 | None(自动) |

**Python 用法:**

```python
from baostock_tool import grid_backtest as gb

# 1) 中文名 + 手动区间
r = gb.run("浦发银行", start="2022-01-01", end="2024-12-31",
           grid_mode="geometric", n_grids=10,
           lower=7.5, upper=11.0, shares_per_grid=200)
print(r.summary())            # 一行核心指标 + 同期股票收益 + 超额

# 2) 自动区间(用 lookback=60 日 low/high 的 0.85~1.15 倍)
r = gb.run("sh.600000", start="2022-01-01", end="2024-12-31",
           grid_mode="geometric", n_grids=15,
           lower_pct=0.85, upper_pct=1.15, lookback=60,
           shares_per_grid=300, base_position=0)

# 3) 加底仓 + 止损止盈
r = gb.run("sz.000001", start="2022-01-01", end="2024-12-31",
           grid_mode="fixed", n_grids=8,
           lower_pct=0.85, upper_pct=1.15, lookback=60,
           shares_per_grid=200, base_position=1000,
           stop_loss=0.08, take_profit=0.30)

# 4) T+0 品种(沪深 300 ETF) — 自动识别为 T+0,允许当日买卖
r = gb.run("sh.510300", start="2022-01-01", end="2024-12-31",
           grid_mode="geometric", n_grids=10,
           lower_pct=0.95, upper_pct=1.05, lookback=30,
           shares_per_grid=1000)            # 不传 t0 → 自动按代码识别
print(r.summary()["交易制度"])            # → "T+0"

# 5) 强制 T+0 / T+1 对比(同一只 ETF)
r_t0 = gb.run("sh.510300", ..., t0=True)
r_t1 = gb.run("sh.510300", ..., t0=False)

# 4) 完整往返的网格效率(实际毛收益 / 理论最大毛收益)
print(r.grid_efficiency())    # buy_date / sell_date / efficiency / holding_days ...

# 5) 交易明细(底仓/网格/止损止盈,带 round_pnl 字段)
print(r.trades_df().head(20))

# 6) 画图 + 文本报告
gb.write_text_report(r, "report.txt")
gb.plot(r, save_dir="./output/grid")
```

**CLI 用法:**

```bash
python -m baostock_tool.cli grid 浦发银行 \
    --start 2022-01-01 --end 2024-12-31 \
    --mode geometric --grids 10 --lower 7.5 --upper 11.0 \
    --shares 200 --report ./output/grid

# T+0 ETF 复盘(自动识别 T+0;可用 --t0 {auto,true,false} 强制)
python -m baostock_tool.cli grid sh.510300 \
    --start 2022-01-01 --end 2024-12-31 \
    --mode geometric --grids 10 \
    --lower-pct 0.95 --upper-pct 1.05 --lookback 30 \
    --shares 1000 --t0 auto --report ./output/grid_etf
```

会同时输出 `grid_report.txt` / `price_grid.png` / `equity.png` / `grid_trades.csv` / `grid_efficiency.csv`。

**复盘重点关注:**

- **`完整往返`**:完整买入→卖出次数(用于估算网格频率)。
- **`网格效率`**:每次往返毛收益 / 理论最大毛收益(等比约 88%、等差 80%~95% 都算正常;过低说明被滑点/手续费吃掉太多)。
- **`超额(网格-股票)`**:网格策略相对纯持有股票的额外收益(震荡市期望为正、单边市期望为负)。
- **`期末持仓` vs `完整往返`**:如果期末持仓远大于往返数,说明行情偏单边、网格没怎么"做差价"。

---

## 数据缓存

```python
# 默认开启,数据存在 .cache/kline/ 下;第二次同参数秒回
df = data.get_kline("sh.600000", "2024-01-01", "2025-12-31", use_cache=True)

# 强制刷新(覆盖本地)
df = data.get_kline("sh.600000", "2024-01-01", "2025-12-31", refresh=True)

# 缓存统计 / 清空
print(data.kline_cache_info())     # 文件数 / 占用
n = data.clear_kline_cache()       # 清空
```

缓存按 `code × freq × adjustflag` 切分,写缓存时会用拉取到的实际区间,**便于后续更小窗口的 start/end 切片直接命中**。

---

## 项目结构

```
baostock-tool/
├── pyproject.toml
├── README.md                       # 本文件
├── baostock_tool/
│   ├── __init__.py                 # 统一导出
│   ├── client.py                   # baostock 登录管理(单例 + 上下文)
│   ├── data.py                     # 数据获取(K线/财报/估值/分红/复权因子/行业/成分股/宏观)
│   ├── data_cache.py               # 本地 K 线 CSV 缓存
│   ├── indicators.py               # 技术指标(MA/MACD/KDJ/RSI/BOLL/ATR/CCI/OBV/DMI/Williams/ROC/MFI/TRIX/SAR/BIAS)
│   ├── patterns.py                 # K 线形态识别
│   ├── screener.py                 # 选股器(16 个模板 + 自定义条件)
│   ├── strategy.py                 # 策略库(8 套经典)
│   ├── backtest.py                 # 事件驱动回测引擎(全套风险指标)
│   ├── position.py                 # 仓位管理(Kelly/波动率目标/固定风险/回撤风控)
│   ├── optimizer.py                # 网格搜索 + Walk-Forward
│   ├── portfolio.py                # 组合回测(等权/风险平价/多空/信号驱动)
│   ├── quant.py                    # 量化分析(IC/分层/归因/Regime)
│   ├── predict.py                  # ML 预测(集成/特征选择/横截面打分/可选 XGB/LGBM)
│   ├── report.py                   # 报告与可视化(jinja2 HTML)
│   ├── grid_backtest.py            # 网格交易回测(T+1 / 涨跌停 / 整百 / 止损止盈)
│   ├── cli.py                      # 命令行入口(17 个子命令)
│   └── utils.py                    # 工具函数
├── examples/
│   ├── 01_quickstart.py
│   ├── 02_backtest.py
│   ├── 03_screen_and_quant.py
│   ├── 04_ml_predict.py
│   ├── 05_risk_and_walkforward.py  # 风险指标 + Walk-Forward
│   ├── 06_portfolio_rotation.py    # 组合回测 + 风险平价
│   └── 07_grid_backtest.py         # 网格交易复盘(手动/自动区间 + 风控)
├── scripts/                        # 可选的 README 同步 hook 安装脚本
└── tests/                          # 离线单元测试(不联网,70+ 用例)
```

---

## 测试

```bash
pytest tests/ -v
```

> `tests/` 是**离线单元测试**(`pytest>=7`),覆盖指标、形态、风险指标、仓位、缓存、组合、优化、预测特征工程、网格回测(含 T+0/T+1)等,**不依赖 baostock 网络**。

---

## 维护本指南

> 📌 **本节是给项目维护者看的**:当源码 / CLI / API 变更时,请同步更新对应的章节,避免文档与代码漂移。

| 项目结构 / API 变化 | 同步本指南的位置 |
| --- | --- |
| 新增 / 删除 `baostock_tool/*.py` 子模块 | [项目结构](#项目结构) + [功能概览](#功能概览) + 顶部 TOC |
| `screener.SCREEN_TEMPLATES` 增删模板 | [选股模板](#选股模板) 表格 |
| `cli.py` 的子命令 / 参数 / `choices` | [CLI 速查](#cli-速查) + 顶部"17 个子命令"计数 |
| `cli.py cmd_screen` 的 `choices` 与 `SCREEN_TEMPLATES` 不一致 | [选股模板](#选股模板) 表格中的"CLI `screen`" 列 |
| `strategy.STRATEGIES` 新增策略 | [功能概览](#功能概览) 的 `strategy` 行 + [5 分钟上手](#5-分钟上手) |
| `predict.AVAILABLE_MODELS` 变化 | [ML 预测](#ml-预测) 末段 |
| `BacktestConfig` / `BacktestResult` 新增字段 | [功能概览](#功能概览) 的 `backtest` 行 + [5 分钟上手](#5-分钟上手) 输出 |
| `grid_backtest.GridConfig` 增删字段 / `run()` 参数变化 | [网格交易回测](#网格交易回测) 表格 + 章节示例 |
| `grid_backtest.detect_price_limit` / `detect_t0` 规则变化 | [网格交易回测](#网格交易回测) "支持的交易制度与涨跌幅"表 |
| 仓库版本号(`__version__` / `pyproject.toml`) | 顶部徽标附近的版本说明(如有) |
| `tests/` 用例数变化 | [测试](#测试) 段的"离线单元测试"行 |

**项目根目录提供了一键审计脚本**(可选,见 `.claude/commands` 配置),你也可以直接跑:

```bash
# 列出当前 CLI 子命令实际数量
python -c "import ast,pathlib; t=ast.parse(pathlib.Path('baostock_tool/cli.py').read_text()); print(sum(isinstance(n,ast.FunctionDef) and n.name.value=='add_parser' for n in ast.walk(t)))"

# 列出 SCREEN_TEMPLATES 全部键
python -c "from baostock_tool.screener import SCREEN_TEMPLATES; print(list(SCREEN_TEMPLATES))"
```

> **持续同步:** 仓库自带一个轻量级 hook 检测脚本 `scripts/check_readme_sync.py`,可被三种方式调用:
>
> | 接入方式 | 一行安装 | 触发时机 |
> | --- | --- | --- |
> | Claude Code `PostToolUse` hook | `bash scripts/install_readme_sync_hook.sh` | 编辑源码 / 配置时由 Claude 自动跑 |
> | Git `pre-commit` hook | 同上(同一脚本会一并安装) | `git commit` 时自检 |
> | 手动 / CI | `python3 scripts/check_readme_sync.py <file>` | 任意时机 |
>
> 安装脚本会写入 `.claude/settings.json` 和 `.git/hooks/pre-commit`;若已存在不会覆盖。卸载只需删除对应条目。
>
> 如果你看到提示但本指南不需要更新,可以直接忽略;反过来,如果指南陈旧而 hook 没触发,手动对照上表逐项检查即可。

---

## 注意事项

- **baostock 接口偶有波动**,批量查询时已内置 `try/except` 容错,单只股票失败不会影响整体。
- **实时行情 baostock 已停用**,本工具专注日 / 周 / 月 / 分钟级历史数据。
- **真实交易前请用本工具的 `BacktestEngine` 充分回测**,并谨慎设定止损 / 止盈;回测不等同于实盘,**滑点 / 成交价 / 涨跌停** 都有简化。
- **本地缓存**默认在 `.cache/kline/`,建议加到 `.gitignore`;`--refresh` 强制刷新。
- **ML 模块**对 sklearn 版本敏感,**1.2+ 推荐**;XGBoost / LightGBM 为可选依赖,缺失自动回退到 sklearn GBDT。
- **依赖 Python 3.9+**(matplotlib>=3.6 + scikit-learn>=1.2 都需要)。

---

## License

MIT
