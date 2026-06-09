"""走步优化 (Walk-Forward Optimization)。

把时间序列切成 N 个 fold,每 fold 内:
    train    : 用于"在网格中选最优参数"(以 metric 最大为准)
    validate : 模拟 OOS 表现
最后拼接所有 OOS 段得到一个 out-of-sample 累计权益曲线。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
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
