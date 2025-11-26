#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Đánh giá nhanh các file CSV chứa cột y_true,y_pred trong thư mục outputs/.
Ví dụ:
    python eval_outputs.py outputs/france_tcn_dev.csv outputs/france_informer_dev.csv
"""

import argparse
import math
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if not {"y_true", "y_pred"}.issubset(df.columns):
        raise ValueError("CSV phải chứa cột y_true và y_pred.")

    y_true = df["y_true"].astype(float).values
    y_pred = df["y_pred"].astype(float).values
    if len(y_true) != len(y_pred):
        raise ValueError("Độ dài y_true và y_pred không khớp.")

    epsilon = 1e-6
    cap = max(abs(y_true).max(), epsilon)
    mape = (abs(y_true - y_pred) / (cap + epsilon)).mean() * 100
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"MAPE": mape, "RMSE": rmse, "MAE": mae}


def main():
    parser = argparse.ArgumentParser(description="Đánh giá các file dự báo.")
    parser.add_argument("files", nargs="+", help="Đường dẫn tới các file CSV.")
    args = parser.parse_args()

    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"[SKIP] {path} không tồn tại.")
            continue
        try:
            df = pd.read_csv(path)
            metrics = compute_metrics(df)
            print(f"\nFile: {path}")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}")


if __name__ == "__main__":
    main()

