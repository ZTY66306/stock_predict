"""02 回测:用 MA5/MA20 交叉策略对浦发银行近三年做回测。"""
import sys
sys.path.insert(0, "/home/ubuntu/work/stock")

from baostock_tool import data, strategy, backtest, report
from baostock_tool.utils import default_start, today_str


def main():
    code = "sh.600000"
    df = data.get_kline(code, "2023-01-01", today_str())
    sig = strategy.run_strategy("ma_cross", df, {"short": 5, "long": 20})

    engine = backtest.BacktestEngine(backtest.BacktestConfig(
        initial_cash=100000,
        commission=0.0003,
        slippage=0.001,
        stop_loss=0.08,
        take_profit=0.30,
    ))
    result = engine.run(df, sig)

    print("=" * 60)
    print("  MA5/MA20 交叉策略回测")
    print("=" * 60)
    print(result.summary().to_string())
    print("\n交易记录:")
    print(result.trades_df().to_string(index=False))

    # 画图
    import os
    out = "/home/ubuntu/work/stock/output/02_backtest"
    os.makedirs(out, exist_ok=True)
    report.plot_equity(result, title=f"{code} MA交叉", save_path=f"{out}/equity.png")
    report.plot_drawdown(result, save_path=f"{out}/dd.png")
    report.plot_kline_with_signals(df, sig, save_path=f"{out}/kline.png")
    report.write_text_report(result, f"{out}/report.txt", code, "ma_cross")


if __name__ == "__main__":
    main()
