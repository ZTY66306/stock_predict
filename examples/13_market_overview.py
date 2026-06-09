"""13_market_overview.py — 涨跌停统计 / 题材热度示例

依赖: pip install akshare

展示:
    1) 当日涨停股池 + 跌停股池
    2) 连板 / 炸板 股
    3) 行业 / 概念 涨停排行
    4) 综合市场情绪 + 强弱打分
    5) 近期涨停趋势

运行:
    python examples/13_market_overview.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baostock_tool import market_overview as mo


def main():
    from datetime import datetime, timedelta
    date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"=== 复盘日期: {date} ===")

    print("\n--- 1) 当日涨停股池 ---")
    df = mo.daily_limit_up(date)
    print(f"  共 {len(df)} 只涨停")
    if not df.empty:
        print(df.head(10).to_string(index=False))

    print("\n--- 2) 当日跌停股池 ---")
    df = mo.daily_limit_down(date)
    print(f"  共 {len(df)} 只跌停")

    print("\n--- 3) 2 连板以上 ---")
    df = mo.consecutive_limit_up(date, n=2)
    print(f"  共 {len(df)} 只")
    if not df.empty:
        print(df.head(5).to_string(index=False))

    print("\n--- 4) 炸板股 ---")
    df = mo.failed_limit_up(date)
    print(f"  共 {len(df)} 只")

    print("\n--- 5) 行业涨停排行 ---")
    df = mo.sector_limit_up_count(date)
    if not df.empty:
        print(df.head(10).to_string(index=False))

    print("\n--- 6) 概念涨停排行 ---")
    df = mo.concept_limit_up_count(date)
    if not df.empty:
        print(df.head(10).to_string(index=False))

    print("\n--- 7) 市场情绪 ---")
    import json
    s = mo.market_sentiment(date)
    print(json.dumps(s, ensure_ascii=False, indent=2, default=str))

    print("\n--- 8) 近 30 日涨停趋势 ---")
    df = mo.limit_up_count_series(start, date)
    if not df.empty:
        print(df.tail(10).to_string())


if __name__ == "__main__":
    main()
