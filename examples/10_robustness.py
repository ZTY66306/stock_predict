"""10_robustness.py — 滚动稳健性测试示例

展示:
    1) 同一组参数在多个滑动窗口上的表现
    2) 与另一组参数对比稳健性
    3) 输出稳健性评分

运行:
    python examples/10_robustness.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baostock_tool import data, optimizer as opt


def main():
    out_dir = "./output/robustness"
    os.makedirs(out_dir, exist_ok=True)

    code = "sh.600000"
    print(f"=== 标的: {code} ===")
    print("拉取 3 年 K 线...")
    df = data.get_kline(code, "2022-01-01", "2024-12-31")
    if df.empty:
        print("无数据")
        return

    # 1) MA(5,20) 的稳健性
    print("\n=== 参数 1: MA(5, 20) ===")
    r1 = opt.RollingRobustness(
        "ma_cross", params={"short": 5, "long": 20},
        window=120, step=30,
    ).run(df)
    print(r1.summary().to_string())
    print(f"\n稳健性评分: {r1.robustness_score()}")

    # 2) MA(3, 60) 的稳健性
    print("\n=== 参数 2: MA(3, 60) ===")
    r2 = opt.RollingRobustness(
        "ma_cross", params={"short": 3, "long": 60},
        window=120, step=30,
    ).run(df)
    print(r2.summary().to_string())
    print(f"\n稳健性评分: {r2.robustness_score()}")

    # 3) 保存 folds 表
    r1.folds_df().to_csv(os.path.join(out_dir, "ma_5_20_folds.csv"), index=False)
    r2.folds_df().to_csv(os.path.join(out_dir, "ma_3_60_folds.csv"), index=False)
    print(f"\nfold 明细已写入 {out_dir}/")

    # 4) 哪组参数更稳?直接比 overall
    s1 = r1.robustness_score()["overall"]
    s2 = r2.robustness_score()["overall"]
    if s1 > s2:
        print(f"\n推荐: MA(5, 20) 更稳健(评分 {s1:.3f} > {s2:.3f})")
    elif s2 > s1:
        print(f"\n推荐: MA(3, 60) 更稳健(评分 {s2:.3f} > {s1:.3f})")
    else:
        print(f"\n两组参数稳健性相当(评分均为 {s1:.3f})")


if __name__ == "__main__":
    main()
