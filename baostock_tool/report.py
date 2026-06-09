"""报告与可视化:回测报告、收益曲线、IC 分布、买卖信号图。"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import BacktestResult

# 中文字体检测:有则用,无则退化到 DejaVu Sans(部分中文会变成方框,但不影响逻辑)
def _setup_cn_font():
    candidates = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei",
                  "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    try:
        from matplotlib.font_manager import findSystemFonts
        available = set(os.path.basename(f) for f in findSystemFonts())
    except Exception:
        available = set()
    for name in candidates:
        if name in available or any(name.lower() in f.lower() for f in available):
            plt.rcParams["font.sans-serif"] = [name]
            break
    else:
        plt.rcParams["font.sans-serif"] = candidates
    plt.rcParams["axes.unicode_minus"] = False

_setup_cn_font()
matplotlib.rcParams["figure.max_open_warning"] = 20


def plot_equity(result: BacktestResult, benchmark: Optional[pd.Series] = None,
                title: str = "权益曲线", save_path: Optional[str] = None) -> None:
    """绘制回测权益曲线,可叠加基准"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result.equity.index, result.equity.values, label="策略", linewidth=2)
    if benchmark is not None and not benchmark.empty:
        # 把 benchmark 归一化到相同初始资金
        bench_norm = benchmark / benchmark.iloc[0] * result.cfg.initial_cash
        bench_norm = bench_norm.reindex(result.equity.index).ffill()
        ax.plot(bench_norm.index, bench_norm.values, label="基准", linewidth=1.5, alpha=0.7)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("权益(元)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"图表已保存到 {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_drawdown(result: BacktestResult, title: str = "回撤曲线",
                  save_path: Optional[str] = None) -> None:
    peak = result.equity.cummax()
    dd = result.equity / peak - 1
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax.plot(dd.index, dd.values, color="red", linewidth=1)
    ax.set_title(f"{title} (最大回撤: {result.max_drawdown*100:.2f}%)", fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("回撤")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    else:
        plt.show()
    plt.close(fig)


def plot_kline_with_signals(df: pd.DataFrame, signals: pd.DataFrame,
                            title: str = "K线与信号", save_path: Optional[str] = None) -> None:
    """画 K 线 + 买卖点"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["close"].values, color="black", linewidth=1, label="收盘价")

    # 买入点
    if "signal" in signals.columns:
        buy = signals[signals["signal"] == 1]
        sell = signals[signals["signal"] == -1]
        if not buy.empty:
            ax.scatter(buy.index, df.loc[buy.index, "close"], marker="^",
                       color="red", s=100, zorder=5, label="买入")
        if not sell.empty:
            ax.scatter(sell.index, df.loc[sell.index, "close"], marker="v",
                       color="green", s=100, zorder=5, label="卖出")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("价格")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    else:
        plt.show()
    plt.close(fig)


def plot_ic(ic_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """画 IC 序列 + 分布直方图"""
    if ic_df.empty:
        print("IC 数据为空,跳过绘图")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(ic_df.index, ic_df["IC"].values, linewidth=1)
    axes[0].axhline(0, color="red", linewidth=0.5, linestyle="--")
    axes[0].set_title("IC 时序")
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(ic_df["IC"].values, bins=30, color="steelblue", alpha=0.7)
    axes[1].axvline(0, color="red", linewidth=0.5, linestyle="--")
    axes[1].set_title("IC 分布")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    else:
        plt.show()
    plt.close(fig)


def plot_layered_returns(layered_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """画因子分层柱状图"""
    if layered_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(layered_df.index.astype(str), layered_df["期均收益"].values * 100)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("因子分层:期均收益率 (%)")
    ax.set_ylabel("期均收益(%)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    else:
        plt.show()
    plt.close(fig)


def render_text_report(result: BacktestResult, code: str = "", name: str = "回测") -> str:
    """生成纯文本回测报告"""
    summary = result.summary()
    lines = [
        "=" * 60,
        f"  策略回测报告: {name}  ({code})",
        "=" * 60,
        "",
        "## 绩效指标",
        summary.to_string(),
        "",
        f"## 交易记录(共 {len(result.trades)} 笔)",
    ]
    trades_df = result.trades_df()
    if not trades_df.empty:
        lines.append(trades_df.to_string(index=False))
    else:
        lines.append("无交易记录")
    return "\n".join(lines)


def write_text_report(result: BacktestResult, path: str, code: str = "", name: str = "回测") -> None:
    """写入文本报告"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_text_report(result, code, name))
    print(f"报告已写入 {path}")


def write_html_report(result: BacktestResult, path: str, code: str = "", name: str = "回测",
                      charts: Optional[list[str]] = None) -> None:
    """把报告写成 HTML(含相对路径图片)。尽量使用 jinja2 模板,缺失时回退到内联字符串。"""
    summary = result.summary()
    if hasattr(result, "risk_metrics"):
        risk = result.risk_metrics()
    else:
        risk = summary
    charts = charts or []
    trades_html = result.trades_df().to_html(index=False) if not result.trades_df().empty else "<p>无交易</p>"
    cards_html = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in summary.items()
    )
    monthly = result.monthly_returns() if hasattr(result, "monthly_returns") else pd.DataFrame()
    monthly_html = monthly.to_html(float_format=lambda x: f"{x*100:.2f}%") if not monthly.empty else "<p>无月度数据</p>"
    chart_html = "".join(f'<div class="chart"><img src="{c}" /></div>' for c in charts)

    # 尝试用 jinja2 模板
    try:
        from jinja2 import Template
        template_str = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{{name}} - {{code}}</title>
<style>
body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:30px;background:#f8f9fa;}
h1{color:#222;} h2{color:#444;border-bottom:2px solid #4a90e2;padding-bottom:6px;margin-top:30px;}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:20px 0;}
.card{background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:12px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.05);}
.card .k{color:#777;font-size:12px;margin-bottom:4px;}
.card .v{color:#222;font-size:18px;font-weight:bold;}
table{border-collapse:collapse;margin:12px 0;background:#fff;}
th,td{border:1px solid #e0e0e0;padding:6px 10px;font-size:13px;}
th{background:#f0f0f0;}
.chart{margin:20px 0;background:#fff;padding:10px;border-radius:6px;}
.chart img{max-width:100%;}
</style></head><body>
<h1>{{name}} - {{code}}</h1>
<div class="cards">{{cards|safe}}</div>
<h2>月度收益</h2>
{{monthly|safe}}
<h2>交易记录</h2>
{{trades|safe}}
<h2>图表</h2>
{{charts|safe}}
<hr><p style="color:#999;font-size:12px;">由 baostock_tool 生成</p>
</body></html>"""
        html = Template(template_str).render(
            name=name, code=code, cards=cards_html, monthly=monthly_html,
            trades=trades_html, charts=chart_html,
        )
    except Exception:
        # 回退到内联字符串
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{name} - {code}</title>
<style>body{{font-family:Arial,sans-serif;margin:40px;}}
table{{border-collapse:collapse;margin:20px 0;}}
th,td{{border:1px solid #ddd;padding:6px 12px;}}
th{{background:#f0f0f0;}}
h1{{color:#333;}}h2{{color:#555;border-bottom:1px solid #eee;padding-bottom:4px;}}
</style></head><body>
<h1>{name} - {code}</h1>
<h2>绩效指标</h2>
{summary.to_frame().to_html()}
<h2>交易记录</h2>
{result.trades_df().to_html(index=False) if not result.trades_df().empty else "<p>无交易</p>"}
<h2>图表</h2>
{''.join(f'<div><img src="{c}" style="max-width:100%"/></div>' for c in charts)}
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML 报告已写入 {path}")


# ============ 新增可视化 ============

def plot_monthly_heatmap(returns: pd.Series, title: str = "月度收益热力图",
                         save_path: Optional[str] = None) -> None:
    """月度收益透视表 + 颜色编码。"""
    if returns.empty:
        return
    df = returns.to_frame("ret").fillna(0)
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(index="year", columns="month", values="ret", aggfunc=lambda x: (1 + x).prod() - 1)
    fig, ax = plt.subplots(figsize=(10, max(3, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.2, vmax=0.2)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title, fontsize=14)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", fontsize=9,
                        color="black" if abs(val) < 0.1 else "white")
    fig.colorbar(im, ax=ax, label="月收益")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    else:
        plt.show()
    plt.close(fig)


def plot_rolling_sharpe(returns: pd.Series, window: int = 60, rf: float = 0.0,
                        title: str = "滚动夏普", save_path: Optional[str] = None) -> None:
    """滚动年化夏普。"""
    if returns.empty:
        return
    excess = returns - rf / 252
    roll_mean = excess.rolling(window, min_periods=window // 2).mean()
    roll_std = returns.rolling(window, min_periods=window // 2).std()
    sharpe = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sharpe.index, sharpe.values, linewidth=1.5)
    ax.axhline(0, color="red", linewidth=0.5, linestyle="--")
    ax.set_title(f"{title} (window={window})", fontsize=14)
    ax.set_ylabel("年化 Sharpe")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    else:
        plt.show()
    plt.close(fig)


def plot_position_timeline(trades: list, save_path: Optional[str] = None) -> None:
    """持仓时间线:横轴日期,纵轴累计持仓状态。"""
    if not trades:
        return
    from .backtest import Trade
    events = []
    for t in trades:
        sign = 1 if t.side == "buy" else -1
        events.append((t.date, sign, t.price))
    events.sort()
    fig, ax = plt.subplots(figsize=(12, 4))
    cum = 0
    xs, ys = [], []
    for d, s, p in events:
        cum = max(0, cum + s)
        xs.append(d)
        ys.append(cum)
    ax.fill_between(xs, ys, 0, step="post", alpha=0.3, color="steelblue")
    ax.step(xs, ys, where="post", linewidth=1.5, color="steelblue")
    ax.set_title("持仓时间线", fontsize=14)
    ax.set_ylabel("持仓状态(1=持有,0=空仓)")
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    else:
        plt.show()
    plt.close(fig)


def plot_kline_candlestick(df: pd.DataFrame, signals: Optional[pd.DataFrame] = None,
                            title: str = "K线(蜡烛图)", save_path: Optional[str] = None) -> None:
    """OHLC 蜡烛图。"""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df))
    width = 0.6
    up = df["close"] >= df["open"]
    # 影线
    ax.vlines(x, df["low"], df["high"], color="black", linewidth=0.8)
    # 实体
    ax.bar(x[up], (df["close"] - df["open"])[up], bottom=df["open"][up], width=width,
           color="red", edgecolor="red")
    ax.bar(x[~up], (df["open"] - df["close"])[~up], bottom=df["close"][~up], width=width,
           color="green", edgecolor="green")
    ax.set_xticks(x[::max(1, len(x) // 10)])
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in df.index[::max(1, len(x) // 10)]], rotation=30)
    if signals is not None and "signal" in signals.columns:
        buy = signals[signals["signal"] == 1]
        sell = signals[signals["signal"] == -1]
        if not buy.empty:
            ax.scatter([df.index.get_loc(d) for d in buy.index],
                       df.loc[buy.index, "close"] * 0.98, marker="^", color="red", s=80, zorder=5, label="买入")
        if not sell.empty:
            ax.scatter([df.index.get_loc(d) for d in sell.index],
                       df.loc[sell.index, "close"] * 1.02, marker="v", color="green", s=80, zorder=5, label="卖出")
        ax.legend()
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    else:
        plt.show()
    plt.close(fig)


def plot_drawdown_underwater(equity: pd.Series, save_path: Optional[str] = None) -> None:
    """水下曲线(回撤深度,可叠加回撤起止标记)。"""
    if equity.empty:
        return
    peak = equity.cummax()
    dd = (equity / peak - 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="steelblue", alpha=0.4)
    ax.plot(dd.index, dd.values, color="steelblue", linewidth=1)
    ax.set_title("水下回撤曲线", fontsize=14)
    ax.set_ylabel("回撤")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    else:
        plt.show()
    plt.close(fig)
