"""Huấn luyện và so sánh các model regression cho coffee price.

Dữ liệu được chia theo thời gian, tuyệt đối không shuffle. Model chỉ được học
trên 80% quan sát đầu; 20% cuối là test mô phỏng dữ liệu tương lai.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "features" / "training_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"


def metrics(actual, predicted):
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / np.sum((actual - np.mean(actual)) ** 2)),
    }


def main():
    dataset = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    target_column = "next_day_price"
    excluded_columns = {target_column, "date", "province", "product", "unit", "source", "source_url"}
    feature_columns = [column for column in dataset.columns if column not in excluded_columns]
    split_index = int(len(dataset) * 0.8)

    train = dataset.iloc[:split_index]
    test = dataset.iloc[split_index:]
    X_train, y_train = train[feature_columns], train[target_column]
    X_test, y_test = test[feature_columns], test[target_column]

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
        ),
    }

    comparison = []
    naive_metrics_path = MODEL_DIR / "naive" / "metrics.json"
    if naive_metrics_path.exists():
        naive = json.loads(naive_metrics_path.read_text(encoding="utf-8"))
        comparison.append({key: naive[key] for key in ("model", "mae", "rmse", "r2")})

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        result = {"model": name, **metrics(y_test.to_numpy(), prediction)}
        comparison.append(result)

        output_dir = MODEL_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "model.pkl").open("wb") as model_file:
            pickle.dump(model, model_file)
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    comparison_frame = pd.DataFrame(comparison).sort_values("mae")
    comparison_frame.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
