"""11_fund_flow.py — 资金流 / 北向 / 龙虎榜示例

依赖: pip install akshare

展示:
    1) 个股资金流(主力 / 超大单 / 大单)
    2) 北向资金汇总
    3) 龙虎榜最近上榜
    4) 板块资金流排名 + 强流入筛选

运行:
    python examples/11_fund_flow.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baostock_tool import fund_flow as ff


def main():
    print("=== 1) 个股资金流(浦发银行 600000, 最近 10 个交易日) ===")
    df = ff.get_fund_flow("sh.600000")
    if df.empty:
        print("无数据(检查网络 / akshare 是否装好)")
    else:
        print(df.head(10).to_string(index=False))

    print("\n=== 2) 北向资金汇总(最近交易日) ===")
    df = ff.get_northbound()
    if not df.empty:
        print(df.to_string(index=False))

    print("\n=== 3) 龙虎榜(本月至今) ===")
    from datetime import datetime
    end = datetime.now().strftime("%Y%m%d")
    start = "20240101"
    df = ff.get_longhubang(start, end)
    if not df.empty:
        print(f"  共 {len(df)} 条,展示前 10 条:")
        print(df.head(10).to_string(index=False))
    else:
        print("无数据")

    print("\n=== 4) 板块资金流(行业) ===")
    df = ff.get_sector_fund_flow(indicator="今日", sector_type="行业资金流")
    if not df.empty:
        print(df.head(15).to_string(index=False))

    print("\n=== 5) 强资金流入板块(主力净流入 > 0) ===")
    df = ff.strong_capital_inflow(threshold=0)
    if not df.empty:
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
