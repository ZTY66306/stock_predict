"""06 组合回测 + 风险平价 + 多空轮动示例。

使用 5 只合成资产演示:
- 等权组合
- 风险平价组合
- 多空轮动组合(按动量信号)
"""
import sys
sys.path.insert(0, "/home/ubuntu/work/stock")

import numpy as np
import pandas as pd

from baostock_tool import portfolio, report
from baostock_tool.utils import ensure_dir


def make_prices(n: int = 300, n_assets: int = 5) -> pd.DataFrame:
    np.random.seed(7)
    idx = pd.bdate_range("2022-01-01", periods=n)
    # 让 5 只资产有不同波动率 / 相关性
    sds = np.array([0.01, 0.02, 0.015, 0.025, 0.012])
    means = np.array([0.0005, 0.0003, 0.0008, 0.0002, 0.0006])
    rets = np.random.randn(n, n_assets) * sds + means
    prices = 10 * (1 + pd.DataFrame(rets, index=idx, columns=[f"A{i+1}" for i in range(n_assets)])).cumprod()
    return prices


def main():
    prices = make_prices()
    print(f"=== 合成 5 资产 × {len(prices)} 日 ===")

    # 1) 等权组合
    p_eq = portfolio.Portfolio(prices, rebalance_freq="W-MON").run()
    print("\n等权组合:")
    print(p_eq.summary().to_string())

    # 2) 风险平价组合
    rets = prices.pct_change().dropna()
    cov = rets.cov() * 252
    rp_weights = portfolio.risk_parity_weights(cov)
    print("\n风险平价权重:")
    print(rp_weights.to_string(float_format=lambda x: f"{x*100:.1f}%"))

    def weight_fn(dt, pr, rw=rp_weights):
        return rw

    p_rp = portfolio.Portfolio(prices, rebalance_freq="ME", weight_fn=weight_fn).run()
    print("\n风险平价组合:")
    print(p_rp.summary().to_string())

    # 3) 多空动量轮动
    mom_signal = prices.pct_change(20)  # 20 日动量作为信号
    p_ls = portfolio.long_short_backtest(prices, mom_signal, top_k=4, rebalance_freq="W-MON")
    print("\n多空动量组合(总做 2 多 + 2 空):")
    print(p_ls.summary().to_string())

    # 4) 出图
    out = ensure_dir("/home/ubuntu/work/stock/output/06_portfolio")
    # 把三种 equity 合并对比
    fig, ax = report.plt.subplots(figsize=(12, 6))
    ax.plot(p_eq.equity.index, p_eq.equity.values, label="等权", linewidth=2)
    ax.plot(p_rp.equity.index, p_rp.equity.values, label="风险平价", linewidth=2)
    ax.plot(p_ls.equity.index, p_ls.equity.values, label="多空动量", linewidth=2)
    ax.set_title("组合对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/compare.png", dpi=120)
    report.plt.close(fig)

    report.plot_equity(p_eq, save_path=f"{out}/equal_weight.png", title="等权组合")
    report.plot_equity(p_rp, save_path=f"{out}/risk_parity.png", title="风险平价组合")
    report.plot_equity(p_ls, save_path=f"{out}/long_short.png", title="多空动量组合")
    report.plot_drawdown(p_eq, save_path=f"{out}/equal_dd.png")
    print(f"\n图表已写入 {out}")


if __name__ == "__main__":
    main()
