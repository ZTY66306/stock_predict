"""12_dca.py — 智能定投示例

展示:
    1) 4 种策略横向对比(pure / smart / dip_buy / lump_sum)
    2) 智能定投的回测报告
    3) 与一次性投入的成本对比

运行:
    python examples/12_dca.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baostock_tool import dca


def main():
    out_dir = "./output/dca"
    os.makedirs(out_dir, exist_ok=True)

    code = "sh.510300"
    print(f"=== 标的: {code} (沪深 300 ETF) ===")

    # 1) 4 策略对比
    print("\n--- 4 种策略对比 (每月 2000 元, 2018-01-01 ~ 2024-12-31) ---")
    cmp = dca.compare_strategies(code, "2018-01-01", "2024-12-31",
                                  amount_per_period=2000, frequency="monthly")
    print(cmp.to_string(index=False))

    # 2) 单策略详细回测
    print("\n--- 智能定投 (smart) 详细回测 ---")
    r = dca.dca_backtest(code, "2018-01-01", "2024-12-31",
                          amount_per_period=2000, frequency="monthly",
                          strategy="smart", ma_window=120)
    print(r.summary().to_string())

    # 3) 出报告 + 图
    dca.write_text_report(r, os.path.join(out_dir, "dca_smart.txt"))
    dca.plot(r, save_path=os.path.join(out_dir, "dca_smart.png"))

    # 4) 对比 4 策略的市值曲线
    results = {}
    for s in ["lump_sum", "pure", "dip_buy", "smart"]:
        results[s] = dca.dca_backtest(code, "2018-01-01", "2024-12-31",
                                       amount_per_period=2000, frequency="monthly",
                                       strategy=s)
    dca.plot_compare(results, save_path=os.path.join(out_dir, "dca_compare.png"))
    print(f"\n报告已写入 {out_dir}/")


if __name__ == "__main__":
    main()
