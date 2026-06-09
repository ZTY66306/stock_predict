"""01 快速开始:拉取一只股票的 K 线,加技术指标,看几行数据。"""
import sys
sys.path.insert(0, "/home/ubuntu/work/stock")

from baostock_tool import data, indicators


def main():
    code = "sh.600000"  # 浦发银行
    print(f"=== 拉取 {code} 最近 1 年 K 线 ===")
    df = data.get_kline(code, "2024-01-01", "2025-12-31")
    print(f"共 {len(df)} 条,首日 {df.index[0].date()}, 末日 {df.index[-1].date()}")
    print(df.tail().to_string())

    print("\n=== 加全套技术指标 ===")
    feat = indicators.add_all(df)
    cols = ["close", "MA5", "MA20", "MACD_DIF", "MACD_DEA", "KDJ_K", "KDJ_J",
            "RSI14", "BOLL_UP", "BOLL_LOW"]
    print(feat[cols].tail().to_string())


if __name__ == "__main__":
    main()
