"""机器学习预测模块。

提供:
    - build_features(df): 从 K 线构建技术指标 + 收益率特征
    - make_supervised(features, target_horizon, target_col='close')
    - PriceRegressor / DirectionClassifier: 包装 sklearn 模型
    - StackingEnsemble / WalkForwardML / cross_sectional_score
    - cross_validate_model: 时序交叉验证
    - predict_next(df, model): 给定最新数据预测下一期

可选加速:安装了 xgboost / lightgbm 后,可在 DirectionClassifier / PriceRegressor
里传 kind='xgb' / kind='lgbm'。缺失时按 sklearn GBDT 回退。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    StackingClassifier, StackingRegressor,
)
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from baostock_tool import indicators as ind

logger = logging.getLogger(__name__)

# 可选依赖探测
try:
    import xgboost  # noqa: F401
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
try:
    import lightgbm  # noqa: F401
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

# 暴露给用户的可用 kind(用于 DirectionClassifier / PriceRegressor)
AVAILABLE_MODELS = ("gbdt", "rf", "logistic", "ridge")
if _HAS_XGB:
    AVAILABLE_MODELS = AVAILABLE_MODELS + ("xgb",)
if _HAS_LGBM:
    AVAILABLE_MODELS = AVAILABLE_MODELS + ("lgbm",)


# ============ 特征工程 ============

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """从单只股票的 K 线 DataFrame 生成技术指标特征矩阵。

    输出: 索引为日期, 列全部为数值,末尾若干行会有 NaN
    """
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    vol = out["volume"]

    # 趋势
    for p in (5, 10, 20, 60):
        out[f"ma_{p}"] = ind.MA(close, p)
        out[f"close_ma{p}_ratio"] = close / out[f"ma_{p}"] - 1
    # 收益
    for p in (1, 3, 5, 10, 20):
        out[f"ret_{p}"] = close.pct_change(p)
    # 波动
    out["vol_5"] = out["ret_1"].rolling(5).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()
    out["vol_ratio"] = out["vol_5"] / out["vol_20"].replace(0, np.nan)
    # MACD
    dif, dea, hist = ind.MACD(close)
    out["macd_dif"] = dif
    out["macd_dea"] = dea
    out["macd_hist"] = hist
    out["macd_hist_diff"] = out["macd_hist"].diff()
    # KDJ
    k, d, j = ind.KDJ(out)
    out["kdj_k"] = k
    out["kdj_d"] = d
    out["kdj_j"] = j
    out["kdj_j_diff"] = out["kdj_j"].diff()
    # RSI
    out["rsi_6"] = ind.RSI(close, 6)
    out["rsi_14"] = ind.RSI(close, 14)
    # BOLL
    mid, up, lo = ind.BOLL(close)
    out["boll_pct"] = (close - lo) / (up - lo).replace(0, np.nan)
    out["boll_width"] = (up - lo) / mid
    # 量能
    out["vol_ma5"] = ind.MA(vol, 5)
    out["vol_ma20"] = ind.MA(vol, 20)
    out["vol_ratio_v"] = vol / out["vol_ma20"].replace(0, np.nan)
    # 价格位置
    for p in (20, 60, 120):
        hi = high.rolling(p).max()
        lo = low.rolling(p).min()
        out[f"pos_{p}"] = (close - lo) / (hi - lo).replace(0, np.nan)
    # 形态
    out["hl_ratio"] = (high - low) / close
    out["oc_ratio"] = (close - out["open"]) / out["open"].replace(0, np.nan)
    out["upper_shadow"] = (high - pd.concat([out["open"], close], axis=1).max(axis=1)) / close
    out["lower_shadow"] = (pd.concat([out["open"], close], axis=1).min(axis=1) - low) / close
    return out


def make_supervised(features: pd.DataFrame, horizon: int = 1,
                    target: str = "direction") -> tuple[pd.DataFrame, pd.Series]:
    """把特征转成 (X, y) 监督学习数据。

    target:
        'direction' -> 二分类(1=涨, 0=跌),用 horizon 日后的收益率符号
        'return'    -> 回归,horizon 日后的收益率
        'close'     -> 回归,horizon 日后的收盘价
    """
    f = features.copy()
    drop_cols = {"open", "high", "low", "close", "volume", "amount",
                 "preclose", "adjustflag", "turn", "tradestatus", "pctChg",
                 "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "isST"}
    X = f.drop(columns=[c for c in drop_cols if c in f.columns], errors="ignore")
    future_close = f["close"].shift(-horizon)
    if target == "direction":
        future_ret = future_close / f["close"] - 1
        y = (future_ret > 0).astype(int)
    elif target == "return":
        y = (future_close / f["close"] - 1)
    elif target == "close":
        y = future_close
    else:
        raise ValueError(f"未知 target: {target}")
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    return X, y


# ============ 模型 ============

@dataclass
class ModelMetrics:
    accuracy: float = 0.0
    mse: float = 0.0
    r2: float = 0.0
    n: int = 0

    def summary(self) -> pd.Series:
        return pd.Series({
            "样本数": self.n,
            "准确率(分类)": f"{self.accuracy * 100:.2f}%" if self.accuracy else "-",
            "MSE": f"{self.mse:.6f}" if self.mse else "-",
            "R²": f"{self.r2:.4f}" if self.r2 else "-",
        })


class _BaseModel:
    def __init__(self, model):
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X.fillna(0), y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X.fillna(0))

    def predict_proba(self, X: pd.DataFrame):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X.fillna(0))
        return None


class DirectionClassifier(_BaseModel):
    """涨跌二分类器。"""

    def __init__(self, kind: str = "gbdt", **kwargs):
        if kind == "gbdt":
            m = GradientBoostingClassifier(random_state=42, **kwargs)
        elif kind == "rf":
            m = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, **kwargs)
        elif kind == "logistic":
            m = LogisticRegression(max_iter=1000, **kwargs)
        elif kind == "xgb":
            if not _HAS_XGB:
                raise ImportError("需要安装 xgboost: pip install xgboost")
            from xgboost import XGBClassifier
            m = XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss", **kwargs)
        elif kind == "lgbm":
            if not _HAS_LGBM:
                raise ImportError("需要安装 lightgbm: pip install lightgbm")
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **kwargs)
        else:
            raise ValueError(f"未知 kind: {kind}; 可用: {AVAILABLE_MODELS}")
        super().__init__(m)
        self.kind = kind

    def fit_walk_forward(self, df: pd.DataFrame, horizon: int = 1,
                         n_splits: int = 5, target: str = "direction") -> ModelMetrics:
        feat = build_features(df)
        X, y = make_supervised(feat, horizon=horizon, target=target)
        if len(X) < 50:
            raise ValueError("样本不足(需 ≥ 50)")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        accs = []
        for train_idx, test_idx in tscv.split(X):
            self.model.fit(X.iloc[train_idx].fillna(0), y.iloc[train_idx])
            preds = self.model.predict(X.iloc[test_idx].fillna(0))
            accs.append(accuracy_score(y.iloc[test_idx], preds))
        # 训练全集
        self.model.fit(X.fillna(0), y)
        return ModelMetrics(accuracy=float(np.mean(accs)), n=len(X))


class PriceRegressor(_BaseModel):
    """价格/收益回归器。"""

    def __init__(self, kind: str = "gbdt", **kwargs):
        if kind == "gbdt":
            m = GradientBoostingRegressor(random_state=42, **kwargs)
        elif kind == "rf":
            m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, **kwargs)
        elif kind == "ridge":
            m = Ridge(**kwargs)
        elif kind == "xgb":
            if not _HAS_XGB:
                raise ImportError("需要安装 xgboost: pip install xgboost")
            from xgboost import XGBRegressor
            m = XGBRegressor(random_state=42, n_jobs=-1, **kwargs)
        elif kind == "lgbm":
            if not _HAS_LGBM:
                raise ImportError("需要安装 lightgbm: pip install lightgbm")
            from lightgbm import LGBMRegressor
            m = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **kwargs)
        else:
            raise ValueError(f"未知 kind: {kind}; 可用: {AVAILABLE_MODELS}")
        super().__init__(m)
        self.kind = kind

    def fit_walk_forward(self, df: pd.DataFrame, horizon: int = 1,
                         n_splits: int = 5, target: str = "return") -> ModelMetrics:
        feat = build_features(df)
        X, y = make_supervised(feat, horizon=horizon, target=target)
        if len(X) < 50:
            raise ValueError("样本不足(需 ≥ 50)")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mses, r2s = [], []
        for train_idx, test_idx in tscv.split(X):
            self.model.fit(X.iloc[train_idx].fillna(0), y.iloc[train_idx])
            preds = self.model.predict(X.iloc[test_idx].fillna(0))
            mses.append(mean_squared_error(y.iloc[test_idx], preds))
            r2s.append(r2_score(y.iloc[test_idx], preds))
        self.model.fit(X.fillna(0), y)
        return ModelMetrics(mse=float(np.mean(mses)), r2=float(np.mean(r2s)), n=len(X))


# ============ 顶层便捷 API ============

def predict_next(df: pd.DataFrame, model: _BaseModel, horizon: int = 1) -> dict:
    """用最近一行数据预测下一期。

    返回: {'direction': 0/1, 'prob_up': float, 'pred_return': float}
    """
    feat = build_features(df)
    X, _ = make_supervised(feat, horizon=horizon, target="direction")
    if X.empty:
        raise ValueError("特征数据为空")
    last = X.iloc[[-1]]
    result = {}
    proba = model.predict_proba(last) if hasattr(model, "predict_proba") else None
    if proba is not None:
        result["prob_up"] = float(proba[0, 1])
        result["direction"] = int(proba[0, 1] >= 0.5)
    else:
        result["direction"] = int(model.predict(last)[0])
    # 若同时是回归器
    if isinstance(model, PriceRegressor):
        feat_r, _ = make_supervised(feat, horizon=horizon, target="return")
        if not feat_r.empty:
            result["pred_return"] = float(model.predict(feat_r.iloc[[-1]])[0])
    return result


def feature_importance(model: _BaseModel, feature_names: list[str], top: int = 20) -> pd.Series:
    if not hasattr(model.model, "feature_importances_"):
        raise ValueError("当前模型不支持 feature_importances_")
    imp = pd.Series(model.model.feature_importances_, index=feature_names)
    return imp.sort_values(ascending=False).head(top)


# ============ 特征选择 ============

def select_features(X: pd.DataFrame, y: pd.Series, k: int = 20,
                    method: str = "mutual_info") -> tuple[pd.DataFrame, list[str]]:
    """基于 mutual_info / f_regression 选 Top-k 特征。返回 (X_selected, selected_names)。"""
    Xf = X.fillna(0)
    if method == "mutual_info":
        if y.dtype.kind in "biu":
            score_fn = mutual_info_classif
        else:
            score_fn = mutual_info_regression
        selector = SelectKBest(score_func=score_fn, k=min(k, Xf.shape[1]))
    elif method == "f_regression":
        selector = SelectKBest(score_func=f_regression, k=min(k, Xf.shape[1]))
    else:
        raise ValueError(f"未知 method: {method}")
    selector.fit(Xf, y)
    mask = selector.get_support()
    return Xf.loc[:, mask], list(Xf.columns[mask])


# ============ 集成模型 ============

class StackingEnsemble:
    """简单 Stacking 集成(分类 / 回归都行)。"""
    def __init__(self, task: str = "clf", base_kinds: Sequence[str] = ("gbdt", "rf"),
                 meta_kind: Optional[str] = None, **kwargs):
        if task not in ("clf", "reg"):
            raise ValueError("task 必须为 'clf' 或 'reg'")
        self.task = task
        self.kwargs = kwargs
        if meta_kind is None:
            meta_kind = "logistic" if task == "clf" else "ridge"
        estimators = []
        for k_ in base_kinds:
            if task == "clf":
                if k_ == "gbdt":
                    e = ("gbdt", GradientBoostingClassifier(random_state=42, n_estimators=100))
                elif k_ == "rf":
                    e = ("rf", RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1))
                elif k_ == "logistic":
                    e = ("logistic", LogisticRegression(max_iter=1000))
                else:
                    raise ValueError(k_)
            else:
                if k_ == "gbdt":
                    e = ("gbdt", GradientBoostingRegressor(random_state=42, n_estimators=100))
                elif k_ == "rf":
                    e = ("rf", RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1))
                elif k_ == "ridge":
                    e = ("ridge", Ridge())
                else:
                    raise ValueError(k_)
            estimators.append(e)
        meta = LogisticRegression(max_iter=1000) if (task == "clf" and meta_kind == "logistic") else (
            Ridge() if (task == "reg" and meta_kind == "ridge") else None
        )
        if meta is None:
            raise ValueError(f"meta_kind 不支持: {meta_kind}")
        cls = StackingClassifier if task == "clf" else StackingRegressor
        self.model = cls(estimators=estimators, final_estimator=meta, n_jobs=-1)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X.fillna(0), y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X.fillna(0))

    def predict_proba(self, X: pd.DataFrame):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X.fillna(0))
        return None


# ============ 横截面打分 / Walk-Forward ML ============

ModelFactory = Callable[[], _BaseModel]


def cross_sectional_score(codes: Sequence[str], start: str, end: str,
                          model_factory: ModelFactory, horizon: int = 5,
                          target: str = "return",
                          show_progress: bool = True) -> pd.DataFrame:
    """横截面打分:每只股票各自训练,输出未来 horizon 日的预测。

    返回: DataFrame(index=date, columns=code),单元是预测的收益/方向(根据 target)。
    """
    from baostock_tool import data
    from tqdm import tqdm
    iterator = tqdm(list(codes), desc="ML打分") if show_progress else codes
    out: dict[str, pd.Series] = {}
    for code in iterator:
        try:
            df = data.get_kline(code, start, end)
            if df.empty or len(df) < 120:
                continue
            feat = build_features(df)
            X, y = make_supervised(feat, horizon=horizon, target=target)
            if len(X) < 60:
                continue
            # walk-forward: 用前 80% 训练,后 20% 预测
            split = int(len(X) * 0.8)
            model = model_factory()
            model.fit(X.iloc[:split], y.iloc[:split])
            if target == "direction":
                # 输出概率
                proba = model.predict_proba(X.iloc[split:])
                preds = pd.Series(proba[:, 1], index=X.index[split:])
            else:
                preds = pd.Series(model.predict(X.iloc[split:]), index=X.index[split:])
            out[code] = preds
        except Exception as e:
            logger.debug("cross_sectional_score %s 失败: %s", code, e)
            continue
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).sort_index()
    return df


class WalkForwardML:
    """走步训练-打分:每隔 retrain_freq 根 bar 重新训练,给出未来 horizon 日预测。"""

    def __init__(self, model_factory: ModelFactory, horizon: int = 5,
                 retrain_freq: int = 20, min_train_size: int = 120,
                 target: str = "return"):
        self.model_factory = model_factory
        self.horizon = horizon
        self.retrain_freq = retrain_freq
        self.min_train_size = min_train_size
        self.target = target
        self.history_: list[dict] = []

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        feat = build_features(df)
        X, y = make_supervised(feat, horizon=self.horizon, target=self.target)
        if len(X) < self.min_train_size + 10:
            raise ValueError("样本不足")
        preds = pd.Series(index=X.index, dtype=float)
        last_train_end = self.min_train_size
        for end in range(self.min_train_size + self.retrain_freq, len(X) + 1, self.retrain_freq):
            tr_X = X.iloc[:end]
            tr_y = y.iloc[:end]
            te_idx = X.index[end - self.retrain_freq:end]
            if len(te_idx) == 0:
                break
            try:
                model = self.model_factory()
                model.fit(tr_X, tr_y)
                if self.target == "direction":
                    proba = model.predict_proba(X.iloc[end - self.retrain_freq:end])
                    preds.loc[te_idx] = proba[:, 1]
                else:
                    preds.loc[te_idx] = model.predict(X.iloc[end - self.retrain_freq:end])
                self.history_.append({"end_idx": end, "n_train": len(tr_X)})
            except Exception as e:
                logger.debug("WalkForwardML 训练失败 @ %s: %s", end, e)
            last_train_end = end
        return preds.dropna()
