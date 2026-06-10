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

from baostock_tool import client, data, indicators, screener, strategy, backtest, report, predict
from baostock_tool import data_cache
from baostock_tool import grid_backtest as gb
from baostock_tool import pairs_trading as pt
from baostock_tool import fund_flow as ff
from baostock_tool import dca
from baostock_tool import market_overview as mo
from baostock_tool import paper_trader as ptr
from baostock_tool.utils import default_start, today_str
from baostock_tool.patterns import list_patterns


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
            print("\n前 10 重要特征:")
            print(imp.to_string())
    except Exception as e:
        pass


def cmd_quant(args):
    from baostock_tool import quant
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
            print("\n相对基准(年化 α/β/信息比):")
            print(metrics_b.to_string())


def cmd_optimize(args):
    """网格搜索 + 可选 walk-forward。"""
    from baostock_tool import optimizer
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
    from baostock_tool import patterns as ptn
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


# ============ 网格回测 ============

def cmd_grid(args):
    """网格交易回测。"""
    t0 = None if args.t0 == "auto" else (args.t0 == "true")
    kwargs = dict(
        start=args.start, end=args.end, capital=args.cash,
        grid_mode=args.mode, n_grids=args.grids,
        lower=args.lower, upper=args.upper,
        lower_pct=args.lower_pct, upper_pct=args.upper_pct,
        lookback=args.lookback, shares_per_grid=args.shares,
        base_position=args.base, commission=args.commission,
        slippage=args.slippage, stop_loss=args.stop_loss,
        take_profit=args.take_profit, lot_size=args.lot,
        price_limit_override=args.price_limit, t0=t0, verbose=True,
    )
    result = gb.run(args.code, **kwargs)
    print("=" * 60)
    print(f"网格回测  {result.code} {result.name}")
    print("=" * 60)
    print(result.summary().to_string())
    if args.report:
        os.makedirs(args.report, exist_ok=True)
        gb.write_text_report(result, os.path.join(args.report, "grid_report.txt"))
        gb.plot(result, save_dir=args.report)
        tdf = result.trades_df()
        if not tdf.empty:
            tdf.to_csv(os.path.join(args.report, "grid_trades.csv"), index=False)
        eff = result.grid_efficiency()
        if not eff.empty:
            eff.to_csv(os.path.join(args.report, "grid_efficiency.csv"), index=False)


# ============ DCA 智能定投 ============

def cmd_dca(args):
    """智能定投回测。"""
    if args.compare:
        cmp = dca.compare_strategies(args.code, args.start, args.end,
                                      amount_per_period=args.amount,
                                      frequency=args.frequency)
        print("--- 4 种策略对比 ---")
        print(cmp.to_string(index=False))
    else:
        r = dca.dca_backtest(args.code, args.start, args.end,
                              amount_per_period=args.amount,
                              frequency=args.frequency,
                              strategy=args.strategy)
        print("=" * 60)
        print(f"DCA  {r.code} {r.name}  strategy={r.strategy}  freq={r.frequency}")
        print("=" * 60)
        print(r.summary().to_string())
        if args.report:
            import os
            os.makedirs(args.report, exist_ok=True)
            dca.write_text_report(r, os.path.join(args.report, "dca_report.txt"))
            dca.plot(r, save_path=os.path.join(args.report, "dca.png"))


# ============ market overview ============

def cmd_market(args):
    """涨跌停统计 / 题材热度。"""
    if args.action == "limit_up":
        df = mo.daily_limit_up(args.date)
        print(f"--- 当日涨停股池 {args.date} ({len(df)} 只) ---")
        if not df.empty:
            print(df.head(args.limit).to_string(index=False))
    elif args.action == "limit_down":
        df = mo.daily_limit_down(args.date)
        print(f"--- 当日跌停股池 {args.date} ({len(df)} 只) ---")
        if not df.empty:
            print(df.head(args.limit).to_string(index=False))
    elif args.action == "consecutive":
        df = mo.consecutive_limit_up(args.date, n=args.n)
        print(f"--- {args.n} 连板以上 {args.date} ({len(df)} 只) ---")
        if not df.empty:
            print(df.head(args.limit).to_string(index=False))
    elif args.action == "failed":
        df = mo.failed_limit_up(args.date)
        print(f"--- 当日炸板股 {args.date} ({len(df)} 只) ---")
        if not df.empty:
            print(df.head(args.limit).to_string(index=False))
    elif args.action == "sector":
        df = mo.sector_limit_up_count(args.date)
        print(f"--- 行业涨停排行 {args.date} ---")
        if not df.empty:
            print(df.head(args.limit).to_string(index=False))
    elif args.action == "concept":
        df = mo.concept_limit_up_count(args.date)
        print(f"--- 概念涨停排行 {args.date} ---")
        if not df.empty:
            print(df.head(args.limit).to_string(index=False))
    elif args.action == "sentiment":
        import json
        s = mo.market_sentiment(args.date)
        print(f"--- 市场情绪 {args.date} ---")
        print(json.dumps(s, ensure_ascii=False, indent=2, default=str))
    elif args.action == "trend":
        df = mo.limit_up_count_series(args.start, args.end)
        print(f"--- 涨停/跌停/炸板 趋势 {args.start} ~ {args.end} ---")
        if not df.empty:
            print(df.to_string())


# ============ paper trader ============

def cmd_paper(args):
    """实盘模拟器。"""
    import json
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    strategies = {c: args.strategy for c in codes}
    params = None
    if args.params:
        params = {c: json.loads(args.params) for c in codes}
    trader = ptr.PaperTrader(
        strategies=strategies, params=params,
        initial_cash=args.cash,
        state_path=args.state,
        webhook_url=args.webhook,
        log_path=args.log,
    )
    report = trader.run_once(date=args.date)
    print("=" * 60)
    print(f"Paper Trader  {report.date.date()}")
    print("=" * 60)
    print(report.summary().to_string())
    if report.signals:
        print("\n--- 推荐信号 ---")
        for s in report.signals:
            if s.signal != 0:
                print(f"  {s.code} {s.name}  signal={s.signal:+d}  "
                      f"shares={s.shares_to_trade}  reason={s.reason}")
    trader.save_state()
    if args.report:
        ptr.write_text_report(report, args.report)


# ============ 配对交易 ============

def cmd_pairs(args):
    """配对交易回测:选对 → 价差回测。"""
    from baostock_tool import data
    print(f"拉取 {len(args.codes.split(','))} 只股票 K 线...")
    prices = pd.DataFrame()
    for c in args.codes.split(","):
        c = c.strip()
        if not c:
            continue
        df = data.get_kline(c, args.start, args.end)
        if df.empty:
            print(f"  [跳过] {c} 无数据")
            continue
        prices[c] = df["close"]
    if prices.shape[1] < 2:
        print("可用股票少于 2 只,无法配对")
        return
    pairs = pt.select_pairs(prices, method=args.method, top_n=args.top_n)
    print(f"\n选出的配对(共 {len(pairs)} 对):")
    print(pairs.to_string(index=False))
    if args.backtest and len(pairs) > 0:
        row = pairs.iloc[0]
        a, b = row["code_a"], row["code_b"]
        print(f"\n回测配对: {a} vs {b}  hedge={row['hedge_ratio']:.4f}")
        result = pt.pairs_backtest(
            prices[a], prices[b],
            entry_z=args.entry_z, exit_z=args.exit_z, lookback=args.lookback,
            capital=args.cash, t0=args.t0,
        )
        print(result.summary().to_string())
        if args.report:
            os.makedirs(args.report, exist_ok=True)
            pt.write_text_report(result, os.path.join(args.report, "pairs_report.txt"))
            pt.plot(result, save_path=os.path.join(args.report, "pairs.png"))


# ============ 多策略融合 ============

def cmd_ensemble(args):
    """跑多策略融合并回测。"""
    from baostock_tool import data, backtest
    df = data.get_kline(args.code, args.start, args.end)
    if df.empty:
        print("无数据")
        return
    import json
    weights = json.loads(args.weights) if args.weights else None
    ens = strategy.EnsembleStrategy(
        args.strategies.split(","), weights=weights,
        voting=args.voting, entry_threshold=args.threshold,
        majority_min=args.majority_min,
    )
    sig = ens.run(df)
    cfg = backtest.BacktestConfig(initial_cash=args.cash)
    result = backtest.BacktestEngine(cfg).run(df, sig)
    print("=" * 50)
    print(f"Ensemble  {args.code}  voting={args.voting}")
    print("=" * 50)
    print(result.summary().to_string())
    if args.report:
        os.makedirs(args.report, exist_ok=True)
        report.plot_equity(result, save_path=os.path.join(args.report, "ensemble_equity.png"))


# ============ 滚动稳健性 ============

def cmd_robustness(args):
    """滚动稳健性测试。"""
    from baostock_tool import data, optimizer
    df = data.get_kline(args.code, args.start, args.end)
    if df.empty:
        print("无数据")
        return
    import json
    params = json.loads(args.params) if args.params else {}
    rr = optimizer.RollingRobustness(
        args.strategy, params=params,
        window=args.window, step=args.step,
    )
    result = rr.run(df)
    print("=" * 60)
    print(f"Rolling Robustness  {args.code}  strategy={args.strategy}")
    print("=" * 60)
    print(result.summary().to_string())
    print("\n稳健性评分(0~1,越大越稳健):")
    print(result.robustness_score())
    if args.report:
        os.makedirs(args.report, exist_ok=True)
        fdf = result.folds_df()
        if not fdf.empty:
            fdf.to_csv(os.path.join(args.report, "robustness_folds.csv"), index=False)


# ============ 资金流 / 北向 / 龙虎榜 ============

def cmd_fundflow(args):
    """资金流 / 北向 / 龙虎榜查询。"""
    if args.kind == "individual":
        df = ff.get_fund_flow(args.code, args.start, args.end)
        if df.empty:
            print("无资金流数据(可能 akshare 不可用或网络问题)")
            return
        print(f"--- 个股资金流 {args.code} ---")
        print(df.head(args.limit).to_string(index=False))
    elif args.kind == "northbound":
        df = ff.get_northbound()
        print("--- 北向资金汇总 ---")
        print(df.to_string(index=False))
    elif args.kind == "longhubang":
        df = ff.get_longhubang(args.start.replace("-", ""), args.end.replace("-", ""))
        print(f"--- 龙虎榜 {args.start} ~ {args.end} ({len(df)} 条) ---")
        if not df.empty:
            print(df.head(args.limit).to_string(index=False))
    elif args.kind == "sector":
        df = ff.get_sector_fund_flow(indicator=args.indicator, sector_type=args.sector_type)
        print(f"--- 板块资金流(行业,今日,前 {args.limit}) ---")
        if not df.empty:
            print(df.head(args.limit).to_string(index=False))


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

    sp = sub.add_parser("grid", help="网格交易回测(代码或中文名)")
    sp.add_argument("code", help="股票代码(sh.600000)或中文名(浦发银行)")
    sp.add_argument("--start", default=default_start(365 * 2))
    sp.add_argument("--end", default=today_str())
    sp.add_argument("--cash", type=float, default=100_000)
    sp.add_argument("--mode", choices=["fixed", "geometric"], default="geometric")
    sp.add_argument("--grids", type=int, default=10)
    sp.add_argument("--lower", type=float, default=None, help="网格下沿(价)")
    sp.add_argument("--upper", type=float, default=None, help="网格上沿(价)")
    sp.add_argument("--lower-pct", type=float, default=None, dest="lower_pct",
                    help="自动下沿=lookback 期最低*lower_pct")
    sp.add_argument("--upper-pct", type=float, default=None, dest="upper_pct",
                    help="自动上沿=lookback 期最高*upper_pct")
    sp.add_argument("--lookback", type=int, default=60)
    sp.add_argument("--shares", type=int, default=100, help="每格股数")
    sp.add_argument("--base", type=int, default=0, help="初始底仓(股)")
    sp.add_argument("--commission", type=float, default=0.0003)
    sp.add_argument("--slippage", type=float, default=0.001)
    sp.add_argument("--stop-loss", type=float, default=None, dest="stop_loss")
    sp.add_argument("--take-profit", type=float, default=None, dest="take_profit")
    sp.add_argument("--lot", type=int, default=100, help="整手股数")
    sp.add_argument("--price-limit", type=float, default=None, dest="price_limit",
                    help="手动指定涨跌幅,跳过代码识别")
    sp.add_argument("--t0", choices=["auto", "true", "false"], default="auto",
                    help="T+0 模式:auto=自动识别(ETF/可转债/国债) / "
                         "true=强制 T+0 / false=强制 T+1")
    sp.add_argument("--report", help="报告输出目录(同时输出图表与 CSV)")
    sp.set_defaults(func=cmd_grid)

    sp = sub.add_parser("pairs", help="配对交易(协整/相关选对 + 价差回测)")
    sp.add_argument("codes", help="逗号分隔的股票代码列表,至少 2 只")
    sp.add_argument("--start", default=default_start(365 * 2))
    sp.add_argument("--end", default=today_str())
    sp.add_argument("--method", choices=["correlation", "cointest", "distance"],
                    default="cointest")
    sp.add_argument("--top-n", type=int, default=5)
    sp.add_argument("--backtest", action="store_true", help="对第一对跑回测")
    sp.add_argument("--entry-z", type=float, default=2.0)
    sp.add_argument("--exit-z", type=float, default=0.5)
    sp.add_argument("--lookback", type=int, default=60)
    sp.add_argument("--cash", type=float, default=100_000)
    sp.add_argument("--t0", action="store_true",
                    help="T+0 模式(用 ETF/可转债配对时可开)")
    sp.add_argument("--report", help="报告输出目录")
    sp.set_defaults(func=cmd_pairs)

    sp = sub.add_parser("ensemble", help="多策略融合(投票/加权)")
    sp.add_argument("code", help="股票代码")
    sp.add_argument("--strategies", default="ma_cross,macd,kdj,rsi_oversold",
                    help="逗号分隔的策略名")
    sp.add_argument("--weights", help="权重 JSON,如 '[0.4,0.3,0.2,0.1]'")
    sp.add_argument("--voting", choices=["weighted", "majority", "veto"],
                    default="weighted")
    sp.add_argument("--threshold", type=float, default=0.3,
                    help="weighted 模式的入场阈值")
    sp.add_argument("--majority-min", type=int, default=2,
                    help="majority/veto 模式的最少同意数")
    sp.add_argument("--start", default=default_start(365 * 2))
    sp.add_argument("--end", default=today_str())
    sp.add_argument("--cash", type=float, default=100_000)
    sp.add_argument("--report", help="报告输出目录")
    sp.set_defaults(func=cmd_ensemble)

    sp = sub.add_parser("robustness", help="滚动稳健性测试")
    sp.add_argument("code", help="股票代码")
    sp.add_argument("--strategy", default="ma_cross")
    sp.add_argument("--params", help="策略参数 JSON,如 '{\"short\":5,\"long\":20}'")
    sp.add_argument("--window", type=int, default=252)
    sp.add_argument("--step", type=int, default=63)
    sp.add_argument("--start", default=default_start(365 * 3))
    sp.add_argument("--end", default=today_str())
    sp.add_argument("--report", help="报告输出目录")
    sp.set_defaults(func=cmd_robustness)

    sp = sub.add_parser("fundflow", help="资金流 / 北向 / 龙虎榜(需 akshare)")
    sp.add_argument("kind", choices=["individual", "northbound", "longhubang", "sector"])
    sp.add_argument("--code", default="", help="个股资金流时必填,形如 sh.600000")
    sp.add_argument("--start", default=default_start(30))
    sp.add_argument("--end", default=today_str())
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--indicator", default="今日",
                    choices=["今日", "3日", "5日", "10日"])
    sp.add_argument("--sector-type", default="行业资金流",
                    dest="sector_type",
                    choices=["行业资金流", "概念资金流", "地域资金流"])
    sp.set_defaults(func=cmd_fundflow)

    # ----- DCA 智能定投 -----
    sp = sub.add_parser("dca", help="智能定投回测(pure / smart / dip_buy / lump_sum)")
    sp.add_argument("code", help="股票代码")
    sp.add_argument("--start", default=default_start(365 * 3))
    sp.add_argument("--end", default=today_str())
    sp.add_argument("--amount", type=float, default=2000,
                    help="每期投入金额(默认 2000)")
    sp.add_argument("--frequency", choices=["weekly", "biweekly", "monthly"],
                    default="monthly")
    sp.add_argument("--strategy", choices=["lump_sum", "pure", "dip_buy", "smart"],
                    default="pure")
    sp.add_argument("--compare", action="store_true",
                    help="横向对比 4 种策略")
    sp.add_argument("--report", help="报告输出目录")
    sp.set_defaults(func=cmd_dca)

    # ----- market overview -----
    sp = sub.add_parser("market", help="涨跌停统计 / 题材热度(需 akshare)")
    sp.add_argument("action", choices=["limit_up", "limit_down", "consecutive",
                                          "failed", "sector", "concept", "sentiment",
                                          "trend"])
    sp.add_argument("--date", default=today_str())
    sp.add_argument("--start", default=default_start(30))
    sp.add_argument("--end", default=today_str())
    sp.add_argument("--n", type=int, default=2, help="连板数(>= N)")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_market)

    # ----- paper trader -----
    sp = sub.add_parser("paper", help="实盘模拟器(把策略接到每日信号)")
    sp.add_argument("codes", help="股票代码,逗号分隔")
    sp.add_argument("--strategy", default="ma_cross",
                    help="所有股票共用此策略")
    sp.add_argument("--params", help="策略参数 JSON")
    sp.add_argument("--cash", type=float, default=100_000)
    sp.add_argument("--date", default=today_str())
    sp.add_argument("--state", default="./paper_state.json",
                    help="状态持久化文件")
    sp.add_argument("--webhook", help="信号 webhook URL(可选)")
    sp.add_argument("--log", help="日志文件路径(可选)")
    sp.add_argument("--report", help="日报输出文件(可选)")
    sp.set_defaults(func=cmd_paper)

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
