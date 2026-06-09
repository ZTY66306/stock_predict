"""14_paper_trader.py — 实盘模拟器示例

展示:
    1) 跟踪多只股票,各跑各的策略
    2) 每日跑一次生成推荐
    3) 状态持久化(进程重启不丢)
    4) 日报输出
    5) 连续多日演示

运行:
    python examples/14_paper_trader.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baostock_tool import paper_trader as ptr


def main():
    out_dir = "./output/paper"
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "state.json")

    # 1) 创建一个跟踪 3 只股票的 trader
    trader = ptr.PaperTrader(
        strategies={
            "sh.600000": "ma_cross",
            "sh.510300": "ma_cross",
            "sh.601318": "rsi_oversold",
        },
        params={
            "sh.600000": {"short": 5, "long": 20},
            "sh.510300": {"short": 10, "long": 30},
            "sh.601318": {"n": 14, "buy": 30, "sell": 70},
        },
        initial_cash=200_000,
        state_path=state_path,
        log_path=os.path.join(out_dir, "trader.log"),
        position_size_pct=0.95,    # 单次用 95% 资金(留点缓冲)
    )

    # 2) 连续 5 个交易日演示
    print("=== 连续 5 个交易日的模拟 ===")
    for i in range(5):
        d = f"2024-12-{15 + i:02d}"
        report = trader.run_once(date=d)
        print(f"\n[{d}]  现金={report.cash:,.0f}  市值={report.market_value:,.0f}  "
              f"总权益={report.total_equity:,.0f}  盈亏比={report.pnl_ratio*100:+.2f}%")
        if report.signals:
            for s in report.signals:
                if s.signal != 0:
                    print(f"  {s.code} {s.name}  signal={s.signal:+d}  "
                          f"shares={s.shares_to_trade}  reason={s.reason}")
        trader.save_state()

    # 3) 出最终报告
    print("\n=== 最终持仓 ===")
    report = trader.run_once(date="2024-12-20")
    print(report.positions_df().to_string(index=False))
    ptr.write_text_report(report, os.path.join(out_dir, "final_report.txt"))

    # 4) 演示状态恢复:新建 trader,自动加载上次的 cash / positions
    print("\n=== 状态恢复测试 ===")
    trader2 = ptr.PaperTrader(
        strategies={"sh.600000": "ma_cross", "sh.510300": "ma_cross",
                     "sh.601318": "rsi_oversold"},
        initial_cash=200_000, state_path=state_path,
    )
    print(f"加载后 cash = {trader2.cash:,.2f}")
    print(f"加载后 sh.600000 shares = {trader2.positions['sh.600000'].shares}")


if __name__ == "__main__":
    main()
