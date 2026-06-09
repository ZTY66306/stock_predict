"""命令行入口:提供一组子命令,不开 GUI。

用法:
    python -m baostock_tool.cli login
    python -m baostock_tool.cli kline sh.600000 --start 2024-01-01 --end 2024-12-31
    python -m baostock_tool.cli screen macd_golden --date 2024-12-01 --limit 20
    python -m baostock_tool.cli backtest sh.600000 --strategy ma_cross --start 2023-01-01
    python -m baostock_tool.cli predict sh.600000 --start 2023-01-01
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import pandas as pd

from . import client, data, indicators, screener, strategy, backtest, report, predict
from . import data_cache
from .utils import default_start, today_str
from .patterns import list_patterns


def cmd_login(args):
    ok = client.login()
    print("登录成功" if ok else "登录失败")
    sys.exit(0 if ok else 1)


def cmd_logout(args):
    client.logout()
    print("已登出")


def cmd_kline(args):
    df = data.get_kline(args.code, args.start, args.end,
                        frequency=args.frequency, adjustflag=args.adjust)
    if df.empty:
        print("无数据")
        return
    print(f"共 {len(df)} 条,首日 {df.index[0].date()},末日 {df.index[-1].date()}")
    print(df.tail(args.n).to_string())
    if args.save:
        df.to_csv(args.save)
        print(f"已保存到 {args.save}")


def cmd_industry(args):
    df = data.get_industry("sw")
    print(df.head(50).to_string())


def cmd_constituents(args):
    df = data.get_index_constituents(args.index, args.date)
    print(df.to_string())


def cmd_screen(args):
    df = screener.screen(args.template, date=args.date, limit=args.limit)
    print(df.to_string())
    if args.save:
        df.to_csv(args.save, index=False)


def cmd_backtest(args):
    df = data.get_kline(args.code, args.start, args.end)
    if df.empty:
        print("无数据")
        return
    sig = strategy.run_strategy(args.strategy, df, json.loads(args.params) if args.params else None)
    cfg = backtest.BacktestConfig(
        initial_cash=args.cash,
        commission=args.commission,
        slippage=args.slippage,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
    )
    engine = backtest.BacktestEngine(cfg)
    result = engine.run(df, sig)
    print("=" * 50)
    print(f"代码: {args.code}    策略: {args.strategy}")
    print("=" * 50)
    print(result.summary().to_string())
    if args.report:
        os.makedirs(args.report, exist_ok=True)
        report.write_text_report(result, os.path.join(args.report, "report.txt"),
                                  args.code, args.strategy)
        report.plot_equity(result, title=f"{args.code} {args.strategy}",
                           save_path=os.path.join(args.report, "equity.png"))
        report.plot_drawdown(result, save_path=os.path.join(args.report, "drawdown.png"))
        report.plot_kline_with_signals(df, sig,
                                       save_path=os.path.join(args.report, "kline.png"))


def cmd_predict(args):
    df = data.get_kline(args.code, args.start, args.end)
    if df.empty:
        print("无数据")
        return
    if args.model == "clf":
        m = predict.DirectionClassifier(kind=args.kind)
    else:
        m = predict.PriceRegressor(kind=args.kind)
    m.fit_walk_forward(df, horizon=args.horizon)
    pred = predict.predict_next(df, m, horizon=args.horizon)
    print("=" * 50)
    print(f"代码: {args.code}  模型: {args.kind}  horizon={args.horizon}")
    print("=" * 50)
    print(pd.Series(pred).to_string())
    # 特征重要性
    try:
        feat = predict.build_features(df)
        X, _ = predict.make_supervised(feat, horizon=args.horizon, target="direction")
        if not X.empty:
            imp = predict.feature_importance(m, X.columns.tolist(), top=10)
            print("\nTop-10 重要特征:")
            print(imp.to_string())
    except Exception as e:
        pass


def cmd_quant(args):
    from . import quant
    codes = data.get_index_codes(args.index, args.date)
    if not codes:
        print("无法获取成分股")
        return
    print(f"拉取 {args.index} {len(codes)} 只成分股 K 线...")
    start = (pd.Timestamp(args.date) - pd.Timedelta(days=args.lookback)).strftime("%Y-%m-%d")
    end = args.date
    klines = {}
    for c in codes:
        df = data.get_kline(c, start, end, show_progress=False)
        if not df.empty:
            klines[c] = df["close"]
    prices = pd.DataFrame(klines).sort_index().ffill()
    rets = prices.pct_change()
    fwd = prices.pct_change(5).shift(-5)
    factor = prices.pct_change(20)
    ic = quant.factor_ic(factor, fwd, method="spearman")
    print("IC 摘要:")
    print(quant.ic_summary(ic).to_string())
    layered = quant.layered_backtest(factor, fwd, q=5)
    print("\n分层回测:")
    print(layered.to_string())
    if args.report:
        os.makedirs(args.report, exist_ok=True)
        report.plot_ic(ic, save_path=os.path.join(args.report, "ic.png"))
        report.plot_layered_returns(layered, save_path=os.path.join(args.report, "layered.png"))


# ============ 新增子命令 ============

def cmd_risk(args):
    """单标的回测 + 风险指标全输出。"""
    df = data.get_kline(args.code, args.start, args.end)
    if df.empty:
        print("无数据")
        return
    sig = strategy.run_strategy(args.strategy, df, json.loads(args.params) if args.params else None)
    cfg = backtest.BacktestConfig(
        initial_cash=args.cash,
        commission=args.commission,
        slippage=args.slippage,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        risk_free_rate=args.rf,
    )
    engine = backtest.BacktestEngine(cfg)
    result = engine.run(df, sig)
    metrics = result.risk_metrics()
    print("=" * 60)
    print(f"风险与绩效全指标  {args.code}  {args.strategy}")
    print("=" * 60)
    print(metrics.to_string())
    if args.benchmark:
        bench_df = data.get_kline(args.benchmark, args.start, args.end)
        if not bench_df.empty:
            bench_eq = (1 + bench_df["close"].pct_change().fillna(0)).cumprod() * args.cash
            metrics_b = result.risk_metrics(bench_eq)
            print("\n相对基准(年化 alpha/beta/IR):")
            print(metrics_b.to_string())


def cmd_optimize(args):
    """网格搜索 + 可选 walk-forward。"""
    from . import optimizer
    df = data.get_kline(args.code, args.start, args.end)
    if df.empty:
        print("无数据")
        return
    grid = json.loads(args.grid)
    if args.walk_forward:
        opt = optimizer.WalkForwardOptimizer(
            strategy_name=args.strategy, param_grid=grid,
            n_splits=args.splits, metric=args.metric,
        )
        res = opt.run(df)
        print("=" * 60)
        print(f"Walk-Forward  {args.code}  {args.strategy}  metric={args.metric}")
        print("=" * 60)
        print(res.summary().to_string(index=False))
        if res.aggregate_metrics is not None and not res.aggregate_metrics.empty:
            print("\n聚合 OOS 指标:")
            print(res.aggregate_metrics.to_string())
        if args.report:
            os.makedirs(args.report, exist_ok=True)
            if res.oos_equity is not None and not res.oos_equity.empty:
                # 用一个 BacktestResult 包装后画图
                tmp = backtest.BacktestResult(equity=res.oos_equity, cfg=backtest.BacktestConfig(initial_cash=res.oos_equity.iloc[0]))
                report.plot_equity(tmp, title=f"OOS 权益 {args.code}",
                                   save_path=os.path.join(args.report, "oos_equity.png"))
                report.plot_drawdown(tmp, save_path=os.path.join(args.report, "oos_dd.png"))
    else:
        gs = optimizer.grid_search(df, args.strategy, grid, metric=args.metric)
        print(gs.head(args.top).to_string())


def cmd_pattern(args):
    """列出最近 K 线形态命中。"""
    from . import patterns as ptn
    df = data.get_kline(args.code, args.start, args.end)
    if df.empty:
        print("无数据")
        return
    patterns_to_check = args.patterns.split(",") if args.patterns else list_patterns()
    rows = []
    for name in patterns_to_check:
        try:
            mask = ptn.detect(df, name)
            hits = df.index[mask].tolist()
            for d in hits:
                rows.append({"date": d.date(), "pattern": name,
                             "close": float(df.loc[d, "close"])})
        except Exception as e:
            print(f"  [跳过] {name}: {e}")
    if not rows:
        print("最近区间无形态命中")
        return
    out = pd.DataFrame(rows).sort_values("date", ascending=False).head(args.limit)
    print(out.to_string(index=False))


def cmd_cache(args):
    info = data.kline_cache_info()
    if not info["exists"]:
        print("缓存目录不存在或为空")
        return
    print(f"缓存目录: {info['dir']}")
    print(f"文件数  : {info['files']}")
    print(f"占用    : {info['size_mb']} MB")
    if args.action == "clear":
        n = data.clear_kline_cache()
        print(f"已清空 {n} 个文件")


def cmd_info(args):
    """个股元信息 + 最近 5 日 K 线。"""
    info = data.get_securities_info(args.code)
    if info:
        print("=" * 50)
        print(f"个股信息  {args.code}")
        print("=" * 50)
        for k, v in info.items():
            print(f"  {k}: {v}")
    df = data.get_kline(args.code, default_start(30), today_str())
    if not df.empty:
        print("\n最近 5 日 K 线:")
        print(df.tail().to_string())
    if args.industry:
        idf = data.get_industry_detail(args.code, today_str())
        if not idf.empty:
            print("\n所属行业:")
            print(idf.to_string(index=False))


def cmd_macro(args):
    """宏观/利率/货币供应。"""
    print("=" * 50)
    print("宏观经济数据(最近 10 条):")
    print("=" * 50)
    macro = data.get_macro()
    if not macro.empty:
        print(macro.tail(10).to_string())
    print("\n存款准备金率(最近 5 条):")
    rr = data.get_required_reserve()
    if not rr.empty:
        print(rr.tail(5).to_string())
    print("\n存款利率(最近 5 条):")
    dr = data.get_deposit_rate()
    if not dr.empty:
        print(dr.tail(5).to_string())


def cmd_ranking(args):
    """横截面打分排序。"""
    codes = data.get_index_codes(args.index, args.date)
    if not codes:
        print("无法获取成分股")
        return
    print(f"打分 {args.index} {len(codes)} 只...")

    def factory():
        if args.task == "clf":
            return predict.DirectionClassifier(kind=args.kind)
        return predict.PriceRegressor(kind=args.kind)

    scores = predict.cross_sectional_score(
        codes, args.start, args.date, model_factory=factory,
        horizon=args.horizon, target="direction" if args.task == "clf" else "return",
    )
    if scores.empty:
        print("无打分结果")
        return
    # 取最近一行作为当前截面
    last = scores.iloc[-1].dropna().sort_values(ascending=False)
    print(f"\nTop {args.top} 看涨(分类)或高收益预测(回归):")
    print(last.head(args.top).to_string())
    print(f"\nBottom {args.top}:")
    print(last.tail(args.top).to_string())
    if args.save:
        last.to_csv(args.save)
        print(f"已保存到 {args.save}")


# 解析器
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baostock_tool", description="baostock 综合工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("login", help="登录 baostock").set_defaults(func=cmd_login)
    sub.add_parser("logout", help="登出").set_defaults(func=cmd_logout)
    sub.add_parser("industry", help="查看行业").set_defaults(func=cmd_industry)

    sp = sub.add_parser("kline", help="获取 K 线")
    sp.add_argument("code")
    sp.add_argument("--start", default=default_start(365))
    sp.add_argument("--end", default=today_str())
    sp.add_argument("--frequency", default="d", choices=["d", "w", "m", "5", "15", "30", "60"])
    sp.add_argument("--adjust", default="1")
    sp.add_argument("--n", type=int, default=20, help="尾部打印行数")
    sp.add_argument("--save", help="保存为 CSV")
    sp.set_defaults(func=cmd_kline)

    sp = sub.add_parser("constituents", help="指数成分股")
    sp.add_argument("index", choices=["hs300", "sz50", "zz500"])
    sp.add_argument("--date", default=today_str())
    sp.set_defaults(func=cmd_constituents)

    sp = sub.add_parser("screen", help="选股")
    sp.add_argument("template",
                    choices=["low_pe", "high_pb", "macd_golden", "kdj_oversold",
                             "rsi_oversold", "volume_breakout", "new_high_20",
                             "low_pe_high_turnover", "bullish_trend"])
    sp.add_argument("--date", default=today_str())
    sp.add_argument("--limit", type=int, default=30)
    sp.add_argument("--save", help="保存为 CSV")
    sp.set_defaults(func=cmd_screen)

    sp = sub.add_parser("backtest", help="回测")
    sp.add_argument("code")
    sp.add_argument("--strategy", default="ma_cross", choices=list(strategy.STRATEGIES))
    sp.add_argument("--start", default=default_start(365 * 3))
    sp.add_argument("--end", default=today_str())
    sp.add_argument("--cash", type=float, default=100000)
    sp.add_argument("--commission", type=float, default=0.0003)
    sp.add_argument("--slippage", type=float, default=0.001)
    sp.add_argument("--stop-loss", type=float, default=None)
    sp.add_argument("--take-profit", type=float, default=None)
    sp.add_argument("--params", help="策略参数 JSON,如 '{\"short\":5,\"long\":20}'")
    sp.add_argument("--report", help="报告输出目录")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("predict", help="ML 预测")
    sp.add_argument("code")
    sp.add_argument("--start", default=default_start(365 * 3))
    sp.add_argument("--end", default=today_str())
    sp.add_argument("--model", choices=["clf", "reg"], default="clf")
    sp.add_argument("--kind", default="gbdt")
    sp.add_argument("--horizon", type=int, default=1)
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("quant", help="因子分析")
    sp.add_argument("--index", default="hs300", choices=["hs300", "sz50", "zz500"])
    sp.add_argument("--date", default=today_str())
    sp.add_argument("--lookback", type=int, default=180)
    sp.add_argument("--report", help="报告输出目录")
    sp.set_defaults(func=cmd_quant)

    # ----- 新增 -----
    sp = sub.add_parser("risk", help="单标的回测 + 风险指标")
    sp.add_argument("code")
    sp.add_argument("--strategy", default="ma_cross", choices=list(strategy.STRATEGIES))
    sp.add_argument("--start", default=default_start(365 * 3))
    sp.add_argument("--end", default=today_str())
    sp.add_argument("--cash", type=float, default=100000)
    sp.add_argument("--commission", type=float, default=0.0003)
    sp.add_argument("--slippage", type=float, default=0.001)
    sp.add_argument("--stop-loss", type=float, default=None)
    sp.add_argument("--take-profit", type=float, default=None)
    sp.add_argument("--params", help="策略参数 JSON")
    sp.add_argument("--rf", type=float, default=0.025, help="年化无风险利率")
    sp.add_argument("--benchmark", help="基准代码,如 sh.000300")
    sp.set_defaults(func=cmd_risk)

    sp = sub.add_parser("optimize", help="网格搜索 / Walk-Forward 优化")
    sp.add_argument("code")
    sp.add_argument("--strategy", default="ma_cross", choices=list(strategy.STRATEGIES))
    sp.add_argument("--grid", required=True, help="参数网格 JSON,例 '{\"short\":[3,5,8],\"long\":[10,20,30]}'")
    sp.add_argument("--start", default=default_start(365 * 3))
    sp.add_argument("--end", default=today_str())
    sp.add_argument("--walk-forward", action="store_true")
    sp.add_argument("--splits", type=int, default=5)
    sp.add_argument("--metric", default="calmar_ratio",
                    choices=["calmar_ratio", "sharpe", "total_return", "sortino", "return_over_dd"])
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument("--report", help="报告输出目录(仅 walk-forward 模式)")
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser("pattern", help="K 线形态识别")
    sp.add_argument("code")
    sp.add_argument("--start", default=default_start(365))
    sp.add_argument("--end", default=today_str())
    sp.add_argument("--patterns", help="形态名,逗号分隔;默认全部")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_pattern)

    sp = sub.add_parser("cache", help="本地缓存管理")
    sp.add_argument("action", choices=["info", "clear"], default="info", nargs="?")
    sp.set_defaults(func=cmd_cache)

    sp = sub.add_parser("info", help="个股元信息 + 最近 K 线")
    sp.add_argument("code")
    sp.add_argument("--industry", action="store_true", help="附加显示所属行业")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("macro", help="宏观/利率/货币供应(最近数据)")
    sp.set_defaults(func=cmd_macro)

    sp = sub.add_parser("ranking", help="横截面 ML 打分排序")
    sp.add_argument("--index", default="hs300", choices=["hs300", "sz50", "zz500"])
    sp.add_argument("--date", default=today_str())
    sp.add_argument("--start", default=default_start(365))
    sp.add_argument("--task", choices=["clf", "reg"], default="clf")
    sp.add_argument("--kind", default="gbdt", choices=predict.AVAILABLE_MODELS)
    sp.add_argument("--horizon", type=int, default=5)
    sp.add_argument("--top", type=int, default=20)
    sp.add_argument("--save", help="保存为 CSV")
    sp.set_defaults(func=cmd_ranking)

    return p


import json  # noqa: E402


def main(argv: Optional[list[str]] = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:
        print(f"[错误] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
