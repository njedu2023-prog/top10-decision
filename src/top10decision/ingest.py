# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


def load_latest_pred(pred_dir="data/pred") -> pd.DataFrame:
    pred_path = Path(pred_dir) / "pred_top10_latest.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"缺少 {pred_path}，请先运行 sync_from_a_top10.py")

    df = pd.read_csv(pred_path)
    # 常见字段校验
    if "ts_code" not in df.columns:
        raise ValueError("预测文件必须包含列：ts_code")
    return df
