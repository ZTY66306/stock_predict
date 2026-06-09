"""04 机器学习预测:涨跌方向 + 特征重要性。"""
import sys
sys.path.insert(0, "/home/ubuntu/work/stock")

from baostock_tool import data, predict
from baostock_tool.utils import default_start, today_str


def main():
    code = "sh.600000"
    print(f"=== 拉取 {code} 数据 ===")
    df = data.get_kline(code, "2022-01-01", today_str())
    print(f"共 {len(df)} 条")

    # 构造特征
    feat = predict.build_features(df)
    print(f"特征数: {feat.shape[1] - 5}")  # 减去原始 OHLCV

    # 涨跌分类
    print("\n=== 训练 GBDT 分类器(预测次日涨跌) ===")
    clf = predict.DirectionClassifier(kind="gbdt", n_estimators=200, max_depth=3)
    metrics = clf.fit_walk_forward(df, horizon=1, n_splits=5)
    print(metrics.summary().to_string())

    # 预测下一日
    pred = predict.predict_next(df, clf, horizon=1)
    print("\n=== 下一日预测 ===")
    for k, v in pred.items():
        print(f"  {k}: {v}")

    # 特征重要性
    X, _ = predict.make_supervised(feat, horizon=1, target="direction")
    imp = predict.feature_importance(clf, X.columns.tolist(), top=10)
    print("\nTop-10 重要特征:")
    print(imp.to_string())

    # 收益回归
    print("\n=== 训练 GBDT 回归器(预测 5 日收益) ===")
    reg = predict.PriceRegressor(kind="gbdt", n_estimators=200, max_depth=3)
    reg_metrics = reg.fit_walk_forward(df, horizon=5, n_splits=5, target="return")
    print(reg_metrics.summary().to_string())


if __name__ == "__main__":
    main()
