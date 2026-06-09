"""新增技术指标的离线单元测试。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from baostock_tool import indicators as ind


@pytest.fixture
def sample_df():
    np.random.seed(7)
    n = 80
    close = 10 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    open_ = close + np.random.randn(n) * 0.05
    volume = np.random.randint(1_000_000, 5_000_000, n)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume, "amount": close * volume,
    }, index=idx)


def test_dmi_shapes(sample_df):
    pdi, mdi, adx, adxr = ind.DMI(sample_df, 14)
    assert len(pdi) == len(sample_df)
    assert (pdi.dropna() >= 0).all() and (pdi.dropna() <= 100).all()
    assert (mdi.dropna() >= 0).all() and (mdi.dropna() <= 100).all()
    assert (adx.dropna() >= 0).all() and (adx.dropna() <= 100).all()


def test_willr_range(sample_df):
    wr = ind.WILLR(sample_df, 14)
    assert (wr.dropna() >= -100).all() and (wr.dropna() <= 0).all()


def test_roc_definition(sample_df):
    roc = ind.ROC(sample_df["close"], 12)
    expected = sample_df["close"].pct_change(12) * 100
    pd.testing.assert_series_equal(roc, expected, check_names=False)


def test_mfi_range(sample_df):
    mfi = ind.MFI(sample_df, 14)
    vals = mfi.dropna()
    assert (vals >= 0).all() and (vals <= 100).all()


def test_trix_finite(sample_df):
    trix = ind.TRIX(sample_df["close"], 12)
    assert len(trix) == len(sample_df)
    assert trix.iloc[-10:].notna().all()


def test_sar_marks_reversals(sample_df):
    sar = ind.SAR(sample_df, af_start=0.02, af_step=0.02, af_max=0.2)
    assert len(sar) == len(sample_df)
    # SAR 至少有一个值
    assert sar.notna().sum() > 0


def test_bias_definition(sample_df):
    bias = ind.BIAS(sample_df["close"], 6)
    ma6 = ind.MA(sample_df["close"], 6)
    expected = (sample_df["close"] - ma6) / ma6 * 100
    pd.testing.assert_series_equal(bias, expected, check_names=False)


def test_add_all_subset(sample_df):
    out = ind.add_all(sample_df, include=("ma", "macd", "rsi"))
    assert "MA5" in out.columns
    assert "MACD_DIF" in out.columns
    assert "RSI14" in out.columns
    # 没加 KDJ/BOLL
    assert "KDJ_K" not in out.columns
    assert "BOLL_MID" not in out.columns
    # 没加新增指标
    assert "PDI" not in out.columns


def test_add_all_full(sample_df):
    out = ind.add_all(sample_df)
    for col in ["MA5", "MACD_DIF", "KDJ_K", "RSI14", "BOLL_MID", "ATR14",
                "CCI", "OBV", "PDI", "ADX", "WR14", "ROC12", "MFI14", "TRIX12",
                "SAR", "BIAS6", "ret_1"]:
        assert col in out.columns, f"缺少列: {col}"
