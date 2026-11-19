import json

import pandas as pd

from src.config import FEATURES_PATH, MODEL_INFO_PATH, MODEL_PATH


class Predictor:
    def __init__(self):
        self.pipeline = None
        self.features = self._read_json(FEATURES_PATH, [])
        self.info = self._read_json(MODEL_INFO_PATH, {})
        if MODEL_PATH.exists():
            import joblib
            self.pipeline = joblib.load(MODEL_PATH)

    @staticmethod
    def _read_json(path, default):
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def schema(self):
        return {"features": self.features, "classes": self.info.get("classes", []),
                "model": {k: self.info[k] for k in ("feature_set", "n_features") if k in self.info}}

    def validate(self, values):
        expected = {item["id"]: item for item in self.features}
        errors = {}
        warnings = {}
        extras = sorted(set(values) - set(expected))
        if extras:
            errors["_extra"] = f"Field tidak dikenal: {', '.join(extras)}"
        for item_id, item in expected.items():
            if item_id not in values:
                errors[item_id] = "Field wajib diisi"
                continue
            value = values[item_id]
            try:
                if item["type"] == "number":
                    number = float(value)
                    if isinstance(value, bool):
                        raise ValueError
                    if "valid_min" in item and number < item["valid_min"]:
                        errors[item_id] = f"harus >= {item['valid_min']}"
                    if "valid_max" in item and number > item["valid_max"]:
                        errors[item_id] = f"harus <= {item['valid_max']}"
                    if (
                        item.get("observed_min") is not None
                        and (
                            number < item["observed_min"]
                            or number > item["observed_max"]
                        )
                    ):
                        warnings[item_id] = "nilai di luar rentang data latih"
                else:
                    allowed = {str(option["value"]) for option in item.get("options", [])}
                    val_str = str(value)
                    try:
                        num_val = float(value)
                        if num_val.is_integer():
                            val_str = str(int(num_val))
                    except (ValueError, TypeError):
                        pass
                    if val_str not in allowed:
                        errors[item_id] = "kode kategori tidak dikenal"
            except (TypeError, ValueError):
                errors[item_id] = "tipe nilai tidak valid"
        return errors, warnings

    def predict(self, values):
        if self.pipeline is None:
            raise RuntimeError("Model belum tersedia. Jalankan training terlebih dahulu.")
        errors, warnings = self.validate(values)
        if errors:
            raise ValueError(errors)
        by_id = {item["id"]: item for item in self.features}
        row = {}
        for item_id, value in values.items():
            item = by_id[item_id]
            row[item["csv_name"]] = float(value)
        frame = pd.DataFrame([row], columns=[item["csv_name"] for item in self.features])
        probabilities = self.pipeline.predict_proba(frame)[0]
        classes = list(self.pipeline.classes_)
        probability_map = {label: round(float(prob), 6) for label, prob in zip(classes, probabilities)}
        result = {
            "label": classes[int(probabilities.argmax())],
            "probabilities": probability_map,
            "model": self.schema()["model"],
        }
        if warnings:
            result["warnings"] = warnings
        return result
