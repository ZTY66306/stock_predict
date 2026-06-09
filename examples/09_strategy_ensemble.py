"""09_strategy_ensemble.py — 多策略融合示例

展示:
    1) 把 4 套经典策略(MA/MACD/KDJ/RSI 超卖)做加权融合
    2) 与单策略对比
    3) 切换 voting 模式(weighted / majority / veto)

运行:
    python examples/09_strategy_ensemble.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baostock_tool import data, backtest, report, strategy as st


def run_one(label, code, voting, threshold, majority_min=None, weights=None):
    df = data.get_kline(code, "2022-01-01", "2024-12-31")
    if df.empty:
        print(f"[{label}] 无数据")
        return None
    if majority_min is not None:
        ens = st.EnsembleStrategy(
            ["ma_cross", "macd", "kdj", "rsi_oversold"],
            weights=weights, voting=voting, majority_min=majority_min,
        )
    else:
        ens = st.EnsembleStrategy(
            ["ma_cross", "macd", "kdj", "rsi_oversold"],
            weights=weights, voting=voting, entry_threshold=threshold,
        )
    sig = ens.run(df)
    cfg = backtest.BacktestConfig(initial_cash=100_000)
    result = backtest.BacktestEngine(cfg).run(df, sig)
    print(f"  [{label:>14s}] 总收益={result.total_return*100:>6.2f}%, "
          f"夏普={result.sharpe:>5.2f}, 交易={len(result.trades):>3d}, "
          f"最大回撤={result.max_drawdown*100:>6.2f}%")
    return result


def main():
    out_dir = "./output/ensemble"
    os.makedirs(out_dir, exist_ok=True)

    code = "sh.600000"
    print(f"=== 标的: {code} ===")
    print("=== 各 voting 模式对比 ===")

    # 1) 单策略基线
    df = data.get_kline(code, "2022-01-01", "2024-12-31")
    for name in ["ma_cross", "macd", "kdj", "rsi_oversold"]:
        sig = st.run_strategy(name, df)
        r = backtest.BacktestEngine().run(df, sig)
        print(f"  [{name:>14s}] 总收益={r.total_return*100:>6.2f}%, "
              f"夏普={r.sharpe:>5.2f}, 交易={len(r.trades):>3d}, "
              f"最大回撤={r.max_drawdown*100:>6.2f}%")

    # 2) weighted 等权
    r_weighted = run_one("weighted 等权", code, "weighted", threshold=0.3)

    # 3) weighted 自定义权重(MA 偏多)
    r_weighted_c = run_one("weighted 自定义", code, "weighted", threshold=0.3,
                            weights=[0.5, 0.2, 0.2, 0.1])

    # 4) majority 2/4
    r_majority = run_one("majority 2/4", code, "majority", threshold=None,
                          majority_min=2)

    # 5) veto 2/4
    r_veto = run_one("veto 2/4", code, "veto", threshold=None,
                      majority_min=2)

    # 6) 出图
    if r_weighted is not None:
        report.plot_equity(r_weighted,
                            save_path=os.path.join(out_dir, "ensemble_weighted.png"))


if __name__ == "__main__":
    main()
