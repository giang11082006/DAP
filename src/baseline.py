"""Đánh giá baseline đơn giản trước khi huấn luyện Machine Learning.

Naive forecast giả định giá ở lần quan sát kế tiếp bằng giá hiện tại.
Không shuffle dữ liệu: 80% đầu dùng train lịch sử, 20% cuối dùng test.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "features" / "training_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "naive"


def calculate_metrics(actual, predicted):
    """Tính các metric phổ biến bằng công thức minh bạch, không làm thay đổi dữ liệu."""
    errors = actual - predicted
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": float(1 - np.sum(errors**2) / np.sum((actual - np.mean(actual)) ** 2)),
    }


def main():
    dataset = pd.read_csv(DATASET_PATH, parse_dates=["date"]).sort_values("date")
    split_index = int(len(dataset) * 0.8)
    test = dataset.iloc[split_index:].copy()

    # price_lag_1 là giá quan sát trước đó, nên naive không nhìn thấy target tương lai.
    actual = test["next_day_price"].to_numpy(dtype=float)
    predicted = test["price_lag_1"].to_numpy(dtype=float)
    metrics = calculate_metrics(actual, predicted)
    metrics.update(
        {
            "model": "naive",
            "train_rows": split_index,
            "test_rows": len(test),
            "test_start": test["date"].min().date().isoformat(),
            "test_end": test["date"].max().date().isoformat(),
            "target": "next_day_price (next calendar day coffee price)",
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
