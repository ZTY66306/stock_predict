"""paper_trader 离线单元测试(全部 mock)。"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from baostock_tool import paper_trader as ptt


# ============ mock 工具 ============

def _mock_kline_factory(seed_map=None):
    """返回一个根据 code 末尾数字生成不同走势的 mock。"""
    def _mock(code, start, end, **k):
        np.random.seed(seed_map.get(code, hash(code) % 1000) if seed_map else hash(code) % 1000)
        n = 250
        # 一段上涨趋势
        close = 10 + np.cumsum(np.random.randn(n) * 0.1) + np.linspace(0, 5, n)
        return pd.DataFrame({
            "open": close, "high": close + 0.2, "low": close - 0.2,
            "close": close, "volume": np.random.randint(1_000_000, 10_000_000, n),
        }, index=pd.bdate_range("2024-01-01", periods=n))
    return _mock


@pytest.fixture
def patched(monkeypatch):
    """把 get_kline / client / resolve_code 全 mock 掉。"""
    import baostock_tool.data as _data
    import baostock_tool.client as _client
    import baostock_tool.grid_backtest as _gb
    monkeypatch.setattr(_data, "get_kline", _mock_kline_factory())
    monkeypatch.setattr(_client, "login", lambda: True)
    monkeypatch.setattr(_client, "ensure_login", lambda: None)
    monkeypatch.setattr(_gb, "resolve_code", lambda c: (c, f"mock_{c[-3:]}"))
    yield


# ============ 数据结构 ============

def test_paper_position_market_value():
    p = ptt.PaperPosition(code="x", shares=100, avg_cost=10.0, last_price=11.0)
    assert p.market_value() == 1100
    assert p.unrealized_pnl() == 100


def test_paper_position_empty():
    p = ptt.PaperPosition(code="x")
    assert p.market_value() == 0
    assert p.unrealized_pnl() == 0


# ============ PaperTrader 基本 ============

def test_paper_trader_init(patched):
    trader = ptt.PaperTrader(
        strategies={"sh.600000": "ma_cross"},
        initial_cash=50_000,
    )
    assert trader.cash == 50_000
    assert "sh.600000" in trader.positions
    assert trader.positions["sh.600000"].shares == 0


def test_paper_trader_run_once_no_signal(patched):
    trader = ptt.PaperTrader(
        strategies={"sh.600000": "ma_cross"},
        params={"sh.600000": {"short": 5, "long": 20}},
        initial_cash=100_000,
    )
    report = trader.run_once(date="2024-12-15")
    assert report.date.strftime("%Y-%m-%d") == "2024-12-15"
    assert report.cash == 100_000
    assert report.total_equity == 100_000
    assert len(report.signals) == 1


def test_paper_trader_signal_dataframe(patched):
    trader = ptt.PaperTrader(
        strategies={"sh.600000": "ma_cross"},
        initial_cash=100_000,
    )
    report = trader.run_once(date="2024-12-15")
    df = report.signals_df()
    assert not df.empty
    assert "code" in df.columns
    assert "signal" in df.columns


def test_paper_trader_position_dataframe(patched):
    trader = ptt.PaperTrader(
        strategies={"sh.600000": "ma_cross"},
        initial_cash=100_000,
    )
    report = trader.run_once(date="2024-12-15")
    df = report.positions_df()
    assert not df.empty
    assert "code" in df.columns
    assert "shares" in df.columns


# ============ 状态持久化 ============

def test_save_and_load_state(patched, tmp_path):
    state_path = str(tmp_path / "state.json")
    trader = ptt.PaperTrader(
        strategies={"sh.600000": "ma_cross"},
        initial_cash=100_000, state_path=state_path,
    )
    trader.run_once(date="2024-12-15")
    # 修改 cash 后保存
    trader.cash = 80000
    trader.save_state()
    # 新建一个 trader 加载
    trader2 = ptt.PaperTrader(
        strategies={"sh.600000": "ma_cross"},
        initial_cash=100_000, state_path=state_path,
    )
    assert trader2.cash == 80000


def test_load_state_no_file(patched, tmp_path):
    state_path = str(tmp_path / "no_such.json")
    trader = ptt.PaperTrader(
        strategies={"sh.600000": "ma_cross"},
        initial_cash=100_000, state_path=state_path,
    )
    # 不应抛异常
    assert trader.cash == 100_000


def test_save_state_no_path(patched):
    trader = ptt.PaperTrader(
        strategies={"sh.600000": "ma_cross"},
        initial_cash=100_000, state_path=None,
    )
    # state_path 为 None 时 save_state 应直接返回 ""
    assert trader.save_state() == ""


# ============ 多次运行 ============

def test_paper_trader_runs_multiple_days(patched):
    trader = ptt.PaperTrader(
        strategies={"sh.600000": "ma_cross"},
        initial_cash=100_000,
    )
    r1 = trader.run_once(date="2024-12-15")
    r2 = trader.run_once(date="2024-12-16")
    r3 = trader.run_once(date="2024-12-17")
    assert len(trader.history) == 3
    assert r1.date != r2.date


def test_paper_trader_handles_failed_code(patched, monkeypatch):
    """某只股票拉数据失败时,不应影响其它股票。"""
    import baostock_tool.data as _data
    def selective_mock(code, start, end, **k):
        if code == "sh.000000":     # 不存在
            return pd.DataFrame()
        return _mock_kline_factory()(code, start, end, **k)
    monkeypatch.setattr(_data, "get_kline", selective_mock)
    trader = ptt.PaperTrader(
        strategies={"sh.000000": "ma_cross", "sh.600000": "ma_cross"},
        initial_cash=100_000,
    )
    report = trader.run_once(date="2024-12-15")
    # sh.000000 应该被跳过,只剩 sh.600000 一个 signal
    assert len(report.signals) == 1
    assert any("无数据" in line for line in report.log)


# ============ 日志 / webhook ============

def test_log_path_written(patched, tmp_path):
    log_path = str(tmp_path / "trader.log")
    trader = ptt.PaperTrader(
        strategies={"sh.600000": "ma_cross"},
        initial_cash=100_000, log_path=log_path,
    )
    trader.run_once(date="2024-12-15")
    assert os.path.exists(log_path)
    with open(log_path) as f:
        content = f.read()
    assert "2024-12-15" in content


def test_webhook_called_when_signal(patched, monkeypatch):
    """信号触发时应 POST 到 webhook。"""
    import baostock_tool.paper_trader as ptmod
    captured = {}
    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        class R: status_code = 200
        return R()
    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    # 用一个会产生信号的策略(固定 signal=+1)
    monkeypatch.setattr(ptmod.strategy, "run_strategy",
                        lambda *a, **k: pd.DataFrame({
                            "signal": [1], "position": [1],
                            "note": [""],
                        }, index=pd.bdate_range("2024-12-15", periods=1)))
    trader = ptt.PaperTrader(
        strategies={"sh.600000": "ma_cross"},
        initial_cash=100_000, webhook_url="http://x",
    )
    trader.run_once(date="2024-12-15")
    assert captured.get("url") == "http://x"
    assert "signals" in captured.get("payload", {})


def test_webhook_failure_does_not_crash(patched, monkeypatch):
    """webhook 失败不应让主流程崩。"""
    import requests
    def bad_post(*a, **k):
        raise ConnectionError("webhook down")
    monkeypatch.setattr(requests, "post", bad_post)
    import baostock_tool.paper_trader as ptmod
    monkeypatch.setattr(ptmod.strategy, "run_strategy",
                        lambda *a, **k: pd.DataFrame({
                            "signal": [1], "position": [1], "note": [""],
                        }, index=pd.bdate_range("2024-12-15", periods=1)))
    trader = ptt.PaperTrader(
        strategies={"sh.600000": "ma_cross"},
        initial_cash=100_000, webhook_url="http://x",
    )
    # 不应抛
    trader.run_once(date="2024-12-15")


# ============ write_text_report ============

def test_write_text_report(patched, tmp_path):
    out = str(tmp_path / "report.txt")
    trader = ptt.PaperTrader(
        strategies={"sh.600000": "ma_cross"},
        initial_cash=100_000,
    )
    report = trader.run_once(date="2024-12-15")
    ptt.write_text_report(report, out)
    assert os.path.exists(out)
    with open(out) as f:
        text = f.read()
    assert "Paper Trader 日报" in text
    assert "2024-12-15" in text
