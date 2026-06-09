"""03 选股 + 量化:多因子选股、计算 IC、分层回测。"""
import sys
sys.path.insert(0, "/home/ubuntu/work/stock")

import pandas as pd
from baostock_tool import data, screener, quant, report
from baostock_tool.utils import today_str


def screen_demo():
    print("=== 内置模板:多头排列 ===")
    picks = screener.screen("bullish_trend", limit=20)
    print(picks.to_string())


def custom_screen_demo():
    print("\n=== 自定义筛选:KDJ 超卖 + MACD 金叉 ===")
    s = screener.Screen(date=today_str(), lookback_days=120, name="custom")
    s.add(screener.kdj_oversold).add(screener.macd_golden_cross)
    picks = s.run(limit=20)
    print(picks.to_string())


def quant_demo():
    print("\n=== 沪深 300 成分股动量因子 IC 分析 ===")
    date = today_str()
    codes = data.get_index_codes("hs300", date)
    print(f"成分股数量: {len(codes)}")

    lookback = 180
    start = (pd.Timestamp(date) - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")

    prices = {}
    for c in codes[:50]:  # 演示取前 50 只,真实场景取全部
        df = data.get_kline(c, start, date, show_progress=False)
        if not df.empty:
            prices[c] = df["close"]
    prices = pd.DataFrame(prices).sort_index().ffill()

    fwd_ret = prices.pct_change(5).shift(-5)
    factor = prices.pct_change(20)

    ic = quant.factor_ic(factor, fwd_ret, method="spearman")
    print("\nIC 摘要:")
    print(quant.ic_summary(ic).to_string())

    layered = quant.layered_backtest(factor, fwd_ret, q=5)
    print("\n分层回测:")
    print(layered.to_string())

    import os
    out = "/home/ubuntu/work/stock/output/03_quant"
    os.makedirs(out, exist_ok=True)
    report.plot_ic(ic, save_path=f"{out}/ic.png")
    report.plot_layered_returns(layered, save_path=f"{out}/layered.png")


if __name__ == "__main__":
    screen_demo()
    custom_screen_demo()
    quant_demo()
