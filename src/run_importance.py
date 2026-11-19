import json
from pathlib import Path
import joblib

from src.config import MODEL_PATH, ROOT, FEATURE_SET
from src.data import load_data
from src.schema import TARGET, get_feature_columns
from src.evaluate import compute_feature_importance


def main():
    print("Loading model and data for feature importance...")
    model = joblib.load(MODEL_PATH)
    split_indices = json.loads((ROOT / "models" / "split_indices.json").read_text("utf-8"))
    df = load_data()
    columns = get_feature_columns(FEATURE_SET)
    X_test = df.loc[split_indices["test"], columns]
    y_test = df.loc[split_indices["test"], TARGET]

    imp_df = compute_feature_importance(model, X_test, y_test, ROOT / "reports")
    print("Feature importance calculation complete!")
    print(imp_df.head(10)[["feature", "mdi_importance", "perm_importance_mean"]])


if __name__ == "__main__":
    main()
