"""market_overview 离线测试(全部 mock akshare)。"""
from __future__ import annotations

import pandas as pd
import pytest

from baostock_tool import market_overview as mo


# ============ mock 工具 ============

class FakeAK:
    """模拟 akshare 涨停 / 跌停 / 炸板 接口。"""

    def __init__(self, zt=None, zd=None, zb=None, zsdt=None):
        self._zt = zt if zt is not None else pd.DataFrame()
        self._zd = zd if zd is not None else pd.DataFrame()
        self._zb = zb if zb is not None else pd.DataFrame()
        self._zsdt = zsdt if zsdt is not None else pd.DataFrame()

    def stock_zt_pool_em(self, date):
        return self._zt

    def stock_zt_pool_dtgc_em(self, date):
        return self._zd

    def stock_zt_pool_zbgc_em(self, date):
        return self._zb

    def stock_zt_pool_zsdt_em(self, start_date, end_date):
        return self._zsdt


@pytest.fixture
def enable_akshare(monkeypatch):
    """启用 akshare + 注入 fake。"""
    monkeypatch.setattr(mo, "_HAS_AKSHARE", True)
    monkeypatch.setattr(mo, "ak", FakeAK())


def _sample_zt():
    return pd.DataFrame({
        "代码": ["000001", "600000", "002142", "601318"],
        "名称": ["平安银行", "浦发银行", "宁波银行", "中国平安"],
        "所属行业": ["银行", "银行", "银行", "保险"],
        "涨停原因": ["银行;业绩预增", "银行;增持", "银行", "保险;业绩"],
        "封板资金": [1e8, 2e8, 5e7, 1.5e8],
        "连板数": [1, 2, 3, 2],
    })


# ============ akshare 缺失时行为 ============

def test_require_akshare_raises_when_missing(monkeypatch):
    monkeypatch.setattr(mo, "_HAS_AKSHARE", False)
    with pytest.raises(ImportError):
        mo.daily_limit_up("2024-01-15")
    with pytest.raises(ImportError):
        mo.daily_limit_down("2024-01-15")
    with pytest.raises(ImportError):
        mo.failed_limit_up("2024-01-15")
    with pytest.raises(ImportError):
        mo.market_sentiment("2024-01-15")


# ============ daily_limit_up / down / failed ============

def test_daily_limit_up_returns_df(monkeypatch, enable_akshare):
    monkeypatch.setattr(mo, "ak", FakeAK(zt=_sample_zt()))
    df = mo.daily_limit_up("2024-01-15")
    assert len(df) == 4
    assert "代码" in df.columns


def test_daily_limit_up_empty(monkeypatch, enable_akshare):
    monkeypatch.setattr(mo, "ak", FakeAK(zt=pd.DataFrame()))
    df = mo.daily_limit_up("2024-01-15")
    assert df.empty


def test_daily_limit_down(monkeypatch, enable_akshare):
    monkeypatch.setattr(mo, "ak", FakeAK(zd=pd.DataFrame({
        "代码": ["000002", "600519"],
        "名称": ["万科A", "贵州茅台"],
    })))
    df = mo.daily_limit_down("2024-01-15")
    assert len(df) == 2


def test_failed_limit_up(monkeypatch, enable_akshare):
    monkeypatch.setattr(mo, "ak", FakeAK(zb=pd.DataFrame({
        "代码": ["000003"], "名称": ["炸板股"],
    })))
    df = mo.failed_limit_up("2024-01-15")
    assert len(df) == 1


# ============ consecutive_limit_up ============

def test_consecutive_limit_up_filters_by_n(monkeypatch, enable_akshare):
    monkeypatch.setattr(mo, "ak", FakeAK(zt=_sample_zt()))
    out = mo.consecutive_limit_up("2024-01-15", n=2)
    # 2 连板及以上:浦发(2)、宁波(3)、平安(2) — 3 只
    assert len(out) == 3
    assert "平安银行" not in out["名称"].tolist()  # 1 连板被滤


def test_consecutive_limit_up_n3(monkeypatch, enable_akshare):
    monkeypatch.setattr(mo, "ak", FakeAK(zt=_sample_zt()))
    out = mo.consecutive_limit_up("2024-01-15", n=3)
    assert len(out) == 1
    assert out.iloc[0]["名称"] == "宁波银行"


def test_consecutive_limit_up_empty_zt(monkeypatch, enable_akshare):
    monkeypatch.setattr(mo, "ak", FakeAK(zt=pd.DataFrame()))
    out = mo.consecutive_limit_up("2024-01-15", n=2)
    assert out.empty


# ============ sector / concept ============

def test_sector_limit_up_count(monkeypatch, enable_akshare):
    monkeypatch.setattr(mo, "ak", FakeAK(zt=_sample_zt()))
    out = mo.sector_limit_up_count("2024-01-15")
    assert out.iloc[0]["所属行业"] == "银行"
    assert out.iloc[0]["limit_up_count"] == 3


def test_concept_limit_up_count(monkeypatch, enable_akshare):
    monkeypatch.setattr(mo, "ak", FakeAK(zt=_sample_zt()))
    out = mo.concept_limit_up_count("2024-01-15")
    # 银行:3, 业绩:2, 业绩预增:1, 增持:1
    assert out.iloc[0]["concept"] == "银行"
    assert int(out.iloc[0]["code_count"]) == 3


def test_sector_count_empty(monkeypatch, enable_akshare):
    monkeypatch.setattr(mo, "ak", FakeAK(zt=pd.DataFrame()))
    assert mo.sector_limit_up_count("2024-01-15").empty


# ============ market_sentiment ============

def test_market_sentiment_basic(monkeypatch, enable_akshare):
    monkeypatch.setattr(mo, "ak", FakeAK(
        zt=_sample_zt(),                # 4 涨停
        zd=pd.DataFrame({"代码": ["000002"]}),  # 1 跌停
        zb=pd.DataFrame({"代码": ["000003"]}),  # 1 炸板
    ))
    s = mo.market_sentiment("2024-01-15")
    assert s["limit_up_count"] == 4
    assert s["limit_down_count"] == 1
    assert s["failed_limit_up_count"] == 1
    assert s["zr_ratio"] == 4.0
    assert 0 < s["failed_rate"] < 1
    assert s["top_sector"] == "银行"
    assert 0 <= s["strength"] <= 1


def test_market_sentiment_no_dl(monkeypatch, enable_akshare):
    """跌停为 0 时 zr_ratio 应为 inf。"""
    monkeypatch.setattr(mo, "ak", FakeAK(zt=_sample_zt(), zd=pd.DataFrame()))
    s = mo.market_sentiment("2024-01-15")
    assert s["zr_ratio"] == float("inf")


def test_market_sentiment_all_empty(monkeypatch, enable_akshare):
    """全空时给个默认情绪。"""
    monkeypatch.setattr(mo, "ak", FakeAK())
    s = mo.market_sentiment("2024-01-15")
    assert s["limit_up_count"] == 0
    assert s["limit_down_count"] == 0
    assert 0 <= s["strength"] <= 1


# ============ limit_up_count_series ============

def test_limit_up_count_series(monkeypatch, enable_akshare):
    df_mock = pd.DataFrame({
        "日期": ["2024-01-15", "2024-01-16", "2024-01-17"],
        "涨停数量": [50, 60, 45],
        "跌停数量": [5, 8, 10],
        "炸板数量": [10, 15, 12],
    })
    monkeypatch.setattr(mo, "ak", FakeAK(zsdt=df_mock))
    out = mo.limit_up_count_series("2024-01-15", "2024-01-17")
    assert len(out) == 3
    assert "limit_up" in out.columns
    assert "limit_down" in out.columns
    assert "failed_limit_up" in out.columns
    assert "fail_rate" in out.columns


def test_limit_up_count_series_empty(monkeypatch, enable_akshare):
    monkeypatch.setattr(mo, "ak", FakeAK(zsdt=pd.DataFrame()))
    out = mo.limit_up_count_series("2024-01-15", "2024-01-17")
    assert out.empty


# ============ first_limit_up(一字板) ============

def test_first_limit_up(monkeypatch, enable_akshare):
    zt = _sample_zt()
    zt["封板时间"] = ["09:30", "10:00", "09:25", "13:30"]
    monkeypatch.setattr(mo, "ak", FakeAK(zt=zt))
    out = mo.first_limit_up("2024-01-15")
    # 09:25 或 09:30 的 2 只
    assert len(out) == 2
