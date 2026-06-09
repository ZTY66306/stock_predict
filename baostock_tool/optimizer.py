"""走步优化 (Walk-Forward Optimization)。

把时间序列切成 N 个 fold,每 fold 内:
    train    : 用于"在网格中选最优参数"(以 metric 最大为准)
    validate : 模拟 OOS 表现
最后拼接所有 OOS 段得到一个 out-of-sample 累计权益曲线。
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import backtest, strategy


MetricFn = Callable[[backtest.BacktestResult], float]


def _default_metric(result: backtest.BacktestResult) -> float:
    """默认评分 = 收益 / 回撤比,稳健性优先。"""
    mdd = abs(result.max_drawdown)
    if mdd == 0:
        return 0.0
    return result.total_return / mdd


def _build_params_grid(grid: dict) -> list[dict]:
    """把 {param: [values]} 笛卡尔积成 [{param: value}, ...]。"""
    keys = list(grid.keys())
    if not keys:
        return [{}]
    return [dict(zip(keys, combo)) for combo in itertools.product(*[grid[k] for k in keys])]


@dataclass
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_score: float
    test_score: float
    test_equity: pd.Series


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold] = field(default_factory=list)
    oos_equity: Optional[pd.Series] = None
    aggregate_metrics: Optional[pd.Series] = None
    best_params_per_fold: Optional[pd.DataFrame] = None
    strategy_name: str = ""
    param_grid: Optional[dict] = None

    def summary(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            rows.append({
                "训练区间": f"{f.train_start.date()} ~ {f.train_end.date()}",
                "验证区间": f"{f.test_start.date()} ~ {f.test_end.date()}",
                "最优参数": f.best_params,
                "训练得分": round(f.train_score, 4),
                "OOS得分": round(f.test_score, 4),
            })
        return pd.DataFrame(rows)


class WalkForwardOptimizer:
    """对内置 strategy 走步优化。"""

    def __init__(self, strategy_name: str, param_grid: dict,
                 n_splits: int = 5, engine: Optional[backtest.BacktestEngine] = None,
                 cfg: Optional[backtest.BacktestConfig] = None,
                 metric: MetricFn | str = "calmar_ratio"):
        self.strategy_name = strategy_name
        self.param_grid = param_grid
        self.n_splits = max(2, n_splits)
        self.engine = engine or backtest.BacktestEngine(cfg or backtest.BacktestConfig())
        self.cfg = self.engine.cfg
        if isinstance(metric, str):
            self.metric = self._named_metric(metric)
        else:
            self.metric = metric

    def _named_metric(self, name: str) -> MetricFn:
        if name == "calmar_ratio":
            return lambda r: r.calmar
        if name == "sharpe":
            return lambda r: r.sharpe
        if name == "total_return":
            return lambda r: r.total_return
        if name == "sortino":
            return lambda r: r.sortino
        if name == "return_over_dd":
            return _default_metric
        raise ValueError(f"未知 metric: {name}")

    def _time_splits(self, n: int) -> list[tuple[int, int, int, int]]:
        """返回 (train_start, train_end, test_start, test_end) 索引。"""
        chunk = n // self.n_splits
        if chunk < 30:
            raise ValueError("样本太少,无法切分")
        splits = []
        for i in range(self.n_splits):
            test_start = chunk * i + chunk // 2
            test_end = min(n, chunk * (i + 1) + chunk // 2)
            train_start = 0
            train_end = test_start
            if train_end - train_start < 30 or test_end - test_start < 10:
                continue
            splits.append((train_start, train_end, test_start, test_end))
        return splits

    def _evaluate(self, df: pd.DataFrame, params: dict, idx: tuple[int, int]) -> tuple[backtest.BacktestResult, float]:
        s, e = idx
        sub = df.iloc[s:e]
        sig = strategy.run_strategy(self.strategy_name, sub, params)
        result = self.engine.run(sub, sig)
        return result, self.metric(result)

    def run(self, df: pd.DataFrame) -> WalkForwardResult:
        if df.empty or len(df) < 60:
            raise ValueError("数据不足,需 ≥ 60 根 bar")
        grid = _build_params_grid(self.param_grid)
        splits = self._time_splits(len(df))
        if not splits:
            raise ValueError("时间切分失败")

        oos_equities: list[pd.Series] = []
        folds: list[WalkForwardFold] = []
        for s, e, ts, te in splits:
            best_score = -np.inf
            best_params: dict = {}
            best_test: Optional[backtest.BacktestResult] = None
            for p in grid:
                tr_score = self._evaluate(df, p, (s, e))[1]
                if tr_score > best_score:
                    best_score = tr_score
                    best_params = p
            train_result, train_score = self._evaluate(df, best_params, (s, e))
            test_result, test_score = self._evaluate(df, best_params, (ts, te))
            folds.append(WalkForwardFold(
                train_start=df.index[s], train_end=df.index[e - 1],
                test_start=df.index[ts], test_end=df.index[te - 1],
                best_params=best_params, train_score=train_score,
                test_score=test_score, test_equity=test_result.equity,
            ))
            oos_equities.append(test_result.equity)

        # 拼接 OOS 权益(取首段起点归一化)
        if oos_equities:
            base = oos_equities[0].iloc[0]
            chained = []
            cur = base
            for eq in oos_equities:
                norm = eq / eq.iloc[0] * cur
                chained.append(norm)
                cur = norm.iloc[-1]
            oos_eq = pd.concat(chained)
            oos_eq.name = "oos_equity"
        else:
            oos_eq = pd.Series(dtype=float)

        best_params_df = pd.DataFrame([
            {**f.best_params, "OOS得分": f.test_score}
            for f in folds
        ])
        if not oos_eq.empty:
            # 重新组装一个 BacktestResult 以复用 risk_metrics
            fake_cfg = backtest.BacktestConfig(initial_cash=base, annual_trading_days=self.cfg.annual_trading_days,
                                               risk_free_rate=self.cfg.risk_free_rate)
            agg = backtest.BacktestResult(equity=oos_eq, cfg=fake_cfg)
            agg_metrics = agg.risk_metrics()
        else:
            agg_metrics = pd.Series(dtype=float)

        return WalkForwardResult(
            folds=folds, oos_equity=oos_eq, aggregate_metrics=agg_metrics,
            best_params_per_fold=best_params_df,
            strategy_name=self.strategy_name, param_grid=self.param_grid,
        )


def grid_search(df: pd.DataFrame, strategy_name: str, param_grid: dict,
                cfg: Optional[backtest.BacktestConfig] = None,
                metric: MetricFn | str = "calmar_ratio") -> pd.DataFrame:
    """简单网格搜索(不做 walk-forward),返回按指标降序的 DataFrame。"""
    if isinstance(metric, str):
        if metric == "calmar_ratio":
            metric_fn = lambda r: r.calmar
        elif metric == "sharpe":
            metric_fn = lambda r: r.sharpe
        elif metric == "total_return":
            metric_fn = lambda r: r.total_return
        elif metric == "sortino":
            metric_fn = lambda r: r.sortino
        else:
            raise ValueError(metric)
    else:
        metric_fn = metric
    grid = _build_params_grid(param_grid)
    engine = backtest.BacktestEngine(cfg or backtest.BacktestConfig())
    rows = []
    for p in grid:
        sig = strategy.run_strategy(strategy_name, df, p)
        result = engine.run(df, sig)
        rows.append({"params": p, "score": metric_fn(result),
                     "total_return": result.total_return,
                     "max_drawdown": result.max_drawdown,
                     "sharpe": result.sharpe,
                     "trades": len(result.trades)})
    df_out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df_out


# ============ 滚动稳健性 (Rolling Robustness) ============

@dataclass
class RollingFold:
    """一次滚动窗口的回测结果。"""
    start: pd.Timestamp
    end: pd.Timestamp
    total_return: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    n_trades: int
    win_rate: float


@dataclass
class RollingRobustnessResult:
    """滚动稳健性测试的结果。"""
    folds: list[RollingFold] = field(default_factory=list)
    strategy_name: str = ""
    params: dict = field(default_factory=dict)
    window: int = 0
    step: int = 0

    def summary(self) -> pd.Series:
        """汇总各 fold 的指标 + 一致性评分。"""
        if not self.folds:
            return pd.Series(dtype=object)
        sharpes = [f.sharpe for f in self.folds]
        returns = [f.total_return for f in self.folds]
        mdds = [f.max_drawdown for f in self.folds]
        wins = [f.win_rate for f in self.folds]
        n_profit = sum(1 for r in returns if r > 0)
        n_loss = sum(1 for r in returns if r < 0)
        return pd.Series({
            "策略": self.strategy_name,
            "参数": self.params,
            "窗口大小": self.window,
            "步长": self.step,
            "fold 数": len(self.folds),
            "中位夏普": f"{np.median(sharpes):.2f}",
            "夏普 std": f"{np.std(sharpes, ddof=0):.2f}",
            "夏普 min / max": f"{min(sharpes):.2f} / {max(sharpes):.2f}",
            "中位收益": f"{np.median(returns)*100:.2f}%",
            "收益 std": f"{np.std(returns, ddof=0)*100:.2f}%",
            "盈利 fold 数": f"{n_profit} / {len(self.folds)} ({n_profit/len(self.folds):.0%})",
            "亏损 fold 数": f"{n_loss} / {len(self.folds)}",
            "中位最大回撤": f"{np.median(mdds)*100:.2f}%",
            "最差回撤": f"{min(mdds)*100:.2f}%",
            "中位胜率": f"{np.median(wins)*100:.2f}%",
        })

    def folds_df(self) -> pd.DataFrame:
        if not self.folds:
            return pd.DataFrame()
        return pd.DataFrame([asdict(f) for f in self.folds])

    def robustness_score(self) -> dict:
        """综合稳健性评分(0~1,越大越稳健)。

        由三个分量等权平均:
            1) 盈利 fold 占比(>0 的 fold / 总 fold)
            2) 1 - 夏普变异系数 CV(std / |mean|),clip 到 0~1
            3) 最差回撤的"温和度":max_dd 在 -30% 内计 1,-60% 计 0
        """
        if not self.folds:
            return {}
        sharpes = np.array([f.sharpe for f in self.folds])
        returns = np.array([f.total_return for f in self.folds])
        mdds = np.array([f.max_drawdown for f in self.folds])
        n = len(self.folds)
        profit_rate = float((returns > 0).sum() / n)
        mean_abs = abs(sharpes.mean()) if sharpes.mean() != 0 else 1e-6
        cv = sharpes.std(ddof=0) / mean_abs
        stability = float(max(0.0, min(1.0, 1.0 - cv)))
        worst_dd = float(mdds.min())  # 最负数
        # dd_score: -30% 以内 = 1,-60% 以上 = 0(线性)
        dd_score = float(np.clip((worst_dd + 0.60) / 0.30, 0.0, 1.0))
        overall = (profit_rate + stability + dd_score) / 3.0
        return {
            "profit_rate": profit_rate,
            "stability": stability,
            "dd_score": dd_score,
            "overall": overall,
        }


class RollingRobustness:
    """滚动稳健性测试:同一组参数,在多个滑动窗口上跑回测,看表现是否一致。

    与 WalkForward 的区别:
        WF    — 每窗口内重选参数(测优化是否过拟合)
        Robust — 同一组参数在多窗口上测(测参数本身是否稳健)
    """

    def __init__(self, strategy_name: str, params: dict,
                 window: int = 252, step: int = 63,
                 engine: Optional[backtest.BacktestEngine] = None,
                 cfg: Optional[backtest.BacktestConfig] = None):
        self.strategy_name = strategy_name
        self.params = params
        self.window = window
        self.step = step
        self.engine = engine or backtest.BacktestEngine(cfg or backtest.BacktestConfig())
        self.cfg = self.engine.cfg

    def run(self, df: pd.DataFrame) -> RollingRobustnessResult:
        if df.empty or len(df) < self.window + 10:
            raise ValueError(f"数据不足,需 ≥ {self.window + 10} 根 bar")
        folds: list[RollingFold] = []
        n = len(df)
        starts = list(range(0, n - self.window + 1, self.step))
        if not starts:
            raise ValueError("窗口/步长组合无法产生任何 fold")
        for s in starts:
            e = s + self.window
            sub = df.iloc[s:e]
            try:
                sig = strategy.run_strategy(self.strategy_name, sub, self.params)
                result = self.engine.run(sub, sig)
                folds.append(RollingFold(
                    start=sub.index[0], end=sub.index[-1],
                    total_return=result.total_return,
                    sharpe=result.sharpe,
                    sortino=result.sortino,
                    calmar=result.calmar,
                    max_drawdown=result.max_drawdown,
                    n_trades=len(result.trades),
                    win_rate=result.win_rate,
                ))
            except Exception as e:
                # 单 fold 失败不影响整体
                continue
        return RollingRobustnessResult(
            folds=folds, strategy_name=self.strategy_name,
            params=self.params, window=self.window, step=self.step,
        )
