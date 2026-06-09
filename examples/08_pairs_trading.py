"""08_pairs_trading.py — 配对交易示例

展示:
    1) 从多只银行股中按协整选对
    2) 跑配对回测(z-score 阈值进出场)
    3) 与单股持有做对比

运行:
    python examples/08_pairs_trading.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from baostock_tool import data, pairs_trading as pt


def main():
    out_dir = "./output/pairs"
    os.makedirs(out_dir, exist_ok=True)

    # 1) 拉一批同行业股票
    codes = ["sh.600000", "sh.600036", "sh.601398", "sh.601939",
             "sh.601288", "sh.601988", "sz.000001", "sz.002142"]
    print("拉取 8 只银行股 K 线(2022-01-01 ~ 2024-12-31)...")
    prices = pd.DataFrame()
    for c in codes:
        df = data.get_kline(c, "2022-01-01", "2024-12-31")
        if not df.empty:
            prices[c] = df["close"]
    prices = prices.dropna(how="any")
    print(f"  -> 实际可用 {prices.shape[1]} 只, {len(prices)} 个交易日")

    # 2) 选对(cointest 法,先按相关预筛,再 EG 协整检验)
    print("\n--- 协整选对 (top 5) ---")
    pairs = pt.select_pairs(prices, method="cointest", top_n=5)
    print(pairs.to_string(index=False))

    # 3) 跑第一对的回测
    if not pairs.empty:
        row = pairs.iloc[0]
        a, b = row["code_a"], row["code_b"]
        print(f"\n--- 配对回测: {a} vs {b} ---")
        result = pt.pairs_backtest(
            prices[a], prices[b],
            entry_z=2.0, exit_z=0.5, lookback=60,
            capital=200_000, t0=False,
        )
        print(result.summary().to_string())

        # 4) 文本报告 + 三联图
        pt.write_text_report(result, os.path.join(out_dir, "pairs_report.txt"))
        pt.plot(result, save_path=os.path.join(out_dir, "pairs.png"))
        print(f"\n报告已写入 {out_dir}/")

    # 5) 距离法选对(适合完全无脑先看)
    print("\n--- 距离法选对 (top 3) ---")
    pairs_d = pt.select_pairs(prices, method="distance", top_n=3)
    print(pairs_d.to_string(index=False))


if __name__ == "__main__":
    main()
