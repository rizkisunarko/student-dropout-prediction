import pandas as pd

from .config import CLASSES, DATA_PATH, FEATURE_SET
from .labels import FEATURE_DISPLAY, get_option_label
from .schema import FEATURES, TARGET, get_feature_columns


def load_data(path=DATA_PATH):
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    frame.columns = frame.columns.str.strip()
    expected = get_feature_columns("B_full") + [TARGET]
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if frame.shape != (4424, 37):
        raise ValueError(f"Unexpected data shape: {frame.shape}")
    if frame.isna().sum().sum():
        raise ValueError("Dataset contains missing values")
    if sorted(frame[TARGET].unique().tolist()) != sorted(CLASSES):
        raise ValueError("Target classes do not match the project contract")
    return frame


def feature_metadata(frame, feature_set=FEATURE_SET):
    selected = set(get_feature_columns(feature_set))
    result = []
    
    # Boundary constraints that make logical sense for academic data
    domain_bounds = {
        "age_at_enrollment": (16, 75),
        "application_order": (0, 9),
        "previous_qualification_grade": (0.0, 200.0),
        "admission_grade": (0.0, 200.0),
        "cu_1st_credited": (0, 30),
        "cu_1st_enrolled": (0, 30),
        "cu_1st_evaluations": (0, 40),
        "cu_1st_approved": (0, 30),
        "cu_1st_grade": (0.0, 20.0),
        "cu_1st_without_evaluations": (0, 10),
        "cu_2nd_credited": (0, 30),
        "cu_2nd_enrolled": (0, 30),
        "cu_2nd_evaluations": (0, 40),
        "cu_2nd_approved": (0, 30),
        "cu_2nd_grade": (0.0, 20.0),
        "cu_2nd_without_evaluations": (0, 10),
        "unemployment_rate": (0.0, 100.0),
        "inflation_rate": (-10.0, 50.0),
        "gdp": (-20.0, 20.0)
    }

    for csv_name, item_id, kind, group in FEATURES:
        if csv_name not in selected:
            continue
        values = sorted(frame[csv_name].unique().tolist())
        disp = FEATURE_DISPLAY.get(item_id, {})
        entry = {
            "id": item_id,
            "csv_name": csv_name,
            "label": disp.get("label", csv_name),
            "group": group,
            "type": "select" if kind == "nominal" else kind if kind == "binary" else "number",
            "unit": disp.get("unit", ""),
            "placeholder": disp.get("placeholder", ""),
            "observed_min": float(frame[csv_name].min()),
            "observed_max": float(frame[csv_name].max())
        }

        if kind in ("binary", "nominal"):
            entry["options"] = [
                {"value": int(v), "label": get_option_label(item_id, v)}
                for v in values
            ]
        else:
            if group == "macro":
                entry["median"] = float(frame[csv_name].median())
        
        if kind == "numeric":
            if item_id in domain_bounds:
                entry["valid_min"], entry["valid_max"] = domain_bounds[item_id]
            else:
                entry["valid_min"] = entry["observed_min"]
                entry["valid_max"] = entry["observed_max"]

        result.append(entry)
    return result
