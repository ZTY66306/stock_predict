"""07_grid_backtest.py — 网格交易复盘示例

展示:
    1) 手动区间网格(明确 lower/upper)
    2) 自动区间网格(用 lookback 期的 low/high × 比例)
    3) 加底仓 + 止损 / 止盈
    4) 等差 vs 等比网格对比
    5) 输出文本报告 + 图表

运行:
    python examples/07_grid_backtest.py
"""
from __future__ import annotations

import os
import sys

# 让脚本可以直接 python examples/07_grid_backtest.py 运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baostock_tool import grid_backtest as gb


def main():
    out_dir = "./output/grid"
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) 手动区间(以浦发银行历史价为例) ----------
    print("=" * 60)
    print("示例 1:手动区间网格(几何, 10 格, 每格 200 股)")
    print("=" * 60)
    r1 = gb.run(
        "浦发银行",
        start="2022-01-01", end="2024-12-31",
        grid_mode="geometric", n_grids=10,
        lower=7.5, upper=11.0,
        shares_per_grid=200, base_position=0,
    )
    print(r1.summary().to_string())
    gb.write_text_report(r1, os.path.join(out_dir, "r1_manual.txt"))
    gb.plot(r1, save_dir=os.path.join(out_dir, "r1_manual"))

    # ---------- 2) 自动区间(用最近 60 日的 0.85~1.15 倍) ----------
    print()
    print("=" * 60)
    print("示例 2:自动区间网格(0.85~1.15 倍 lookback=60 日高/低)")
    print("=" * 60)
    r2 = gb.run(
        "sh.600000",
        start="2022-01-01", end="2024-12-31",
        grid_mode="geometric", n_grids=15,
        lower_pct=0.85, upper_pct=1.15, lookback=60,
        shares_per_grid=300, base_position=0,
    )
    print(r2.summary().to_string())
    gb.plot(r2, save_dir=os.path.join(out_dir, "r2_auto"))

    # ---------- 3) 加底仓 + 风控 ----------
    print()
    print("=" * 60)
    print("示例 3:底仓 1000 股 + 8% 止损 + 30% 止盈")
    print("=" * 60)
    r3 = gb.run(
        "sz.000001",
        start="2022-01-01", end="2024-12-31",
        grid_mode="fixed", n_grids=8,
        lower_pct=0.85, upper_pct=1.15, lookback=60,
        shares_per_grid=200, base_position=1000,
        stop_loss=0.08, take_profit=0.30,
    )
    print(r3.summary().to_string())
    gb.plot(r3, save_dir=os.path.join(out_dir, "r3_with_risk"))

    # ---------- 4) 等差 vs 等比对比 ----------
    print()
    print("=" * 60)
    print("示例 4:等差 vs 等比(同一只股,同一区间)")
    print("=" * 60)
    r_geo = gb.run(
        "sh.600519",
        start="2022-01-01", end="2024-12-31",
        grid_mode="geometric", n_grids=12,
        lower_pct=0.80, upper_pct=1.20, lookback=60,
        shares_per_grid=100,
    )
    r_fix = gb.run(
        "sh.600519",
        start="2022-01-01", end="2024-12-31",
        grid_mode="fixed", n_grids=12,
        lower_pct=0.80, upper_pct=1.20, lookback=60,
        shares_per_grid=100,
    )
    print(f"  等比: 总收益={r_geo.equity.iloc[-1]/r_geo.cfg.capital-1:.2%}, "
          f"交易={len(r_geo.trades)}")
    print(f"  等差: 总收益={r_fix.equity.iloc[-1]/r_fix.cfg.capital-1:.2%}, "
          f"交易={len(r_fix.trades)}")

    # ---------- 5) T+0 品种(ETF)复盘 ----------
    print()
    print("=" * 60)
    print("示例 5:T+0 品种(沪深 300 ETF)网格 — 自动识别 T+0")
    print("=" * 60)
    r_etf = gb.run(
        "sh.510300",  # 沪深 300 ETF
        start="2022-01-01", end="2024-12-31",
        grid_mode="geometric", n_grids=10,
        lower_pct=0.95, upper_pct=1.05, lookback=30,
        shares_per_grid=1000,  # ETF 走 1000 份/手
    )
    print(r_etf.summary().to_string())
    gb.plot(r_etf, save_dir=os.path.join(out_dir, "r5_etf_t0"))

    # ---------- 5b) T+0 强制对比 ----------
    print()
    print("=" * 60)
    print("示例 5b:同一只 ETF,强制 T+0 vs 强制 T+1 对比")
    print("=" * 60)
    r_etf_t0 = gb.run(
        "sh.510300",
        start="2022-01-01", end="2024-12-31",
        grid_mode="geometric", n_grids=10,
        lower_pct=0.95, upper_pct=1.05, lookback=30,
        shares_per_grid=1000, t0=True,
    )
    r_etf_t1 = gb.run(
        "sh.510300",
        start="2022-01-01", end="2024-12-31",
        grid_mode="geometric", n_grids=10,
        lower_pct=0.95, upper_pct=1.05, lookback=30,
        shares_per_grid=1000, t0=False,
    )
    print(f"  强制 T+0: 总收益={r_etf_t0.equity.iloc[-1]/r_etf_t0.cfg.capital-1:.2%}, "
          f"完整往返={r_etf_t0.n_full_roundtrips}, 交易={len(r_etf_t0.trades)}")
    print(f"  强制 T+1: 总收益={r_etf_t1.equity.iloc[-1]/r_etf_t1.cfg.capital-1:.2%}, "
          f"完整往返={r_etf_t1.n_full_roundtrips}, 交易={len(r_etf_t1.trades)}")

    # ---------- 6) 网格效率明细 ----------
    print()
    print("=" * 60)
    print("示例 5:网格效率(每次完整往返,前 5 行)")
    print("=" * 60)
    eff = r1.grid_efficiency()
    if not eff.empty:
        print(eff.head().to_string(index=False))
        print(f"  ... 共 {len(eff)} 个完整往返,平均效率 {eff['efficiency'].mean():.2%}")

    print()
    print(f"全部报告已写入 {out_dir}/")


if __name__ == "__main__":
    main()
