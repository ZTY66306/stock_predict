"""05 风险指标 + Walk-Forward 优化示例。

不需要联网时,使用 synthetic 数据展示完整指标计算与 OOS 拼接。
"""
import sys
sys.path.insert(0, "/home/ubuntu/work/stock")

import numpy as np
import pandas as pd

from baostock_tool import backtest, strategy, optimizer, report
from baostock_tool.utils import ensure_dir


def make_synthetic(n: int = 600) -> pd.DataFrame:
    """构造一只带趋势 + 噪声的虚拟 K 线。"""
    np.random.seed(2024)
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = 10 + np.cumsum(np.random.randn(n) * 0.2)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1_000_000, 5_000_000, n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=idx)


def main():
    df = make_synthetic()
    print(f"=== 合成数据 {len(df)} 根 bar ===")

    # 1) 基础回测 + 风险指标
    print("\n=== MA5/MA20 策略风险指标 ===")
    sig = strategy.run_strategy("ma_cross", df, {"short": 5, "long": 20})
    cfg = backtest.BacktestConfig(
        initial_cash=100000,
        commission=0.0003,
        stop_loss=0.08,
        take_profit=0.30,
        risk_free_rate=0.025,
    )
    res = backtest.BacktestEngine(cfg).run(df, sig)
    print(res.risk_metrics().to_string())
    print("\n月度收益:")
    print(res.monthly_returns().to_string(float_format=lambda x: f"{x*100:.2f}%"))

    # 2) Walk-Forward 优化
    print("\n=== Walk-Forward 网格搜索: short ∈ {3,5,8}, long ∈ {10,20,30} ===")
    opt = optimizer.WalkForwardOptimizer(
        strategy_name="ma_cross",
        param_grid={"short": [3, 5, 8], "long": [10, 20, 30]},
        n_splits=4, metric="calmar_ratio",
    )
    wf = opt.run(df)
    print(wf.summary().to_string(index=False))
    print("\nOOS 聚合指标:")
    if wf.aggregate_metrics is not None:
        print(wf.aggregate_metrics.to_string())

    # 3) 出图
    out = ensure_dir("/home/ubuntu/work/stock/output/05_risk_wf")
    report.plot_equity(res, save_path=f"{out}/equity.png", title="基础 MA 策略")
    report.plot_drawdown(res, save_path=f"{out}/dd.png")
    report.plot_monthly_heatmap(res.daily_returns, save_path=f"{out}/heatmap.png")
    report.plot_rolling_sharpe(res.daily_returns, save_path=f"{out}/rolling_sharpe.png")
    if wf.oos_equity is not None and not wf.oos_equity.empty:
        # OOS 权益画图(用 BacktestResult 包装)
        tmp = backtest.BacktestResult(equity=wf.oos_equity, cfg=cfg)
        report.plot_equity(tmp, save_path=f"{out}/oos_equity.png", title="OOS 拼接权益")
        report.plot_drawdown(tmp, save_path=f"{out}/oos_dd.png")
    report.write_text_report(res, f"{out}/report.txt", "synthetic", "ma_cross")
    print(f"\n报告与图表已写入 {out}")


if __name__ == "__main__":
    main()
