import json
import platform
from datetime import datetime, timezone

import joblib
import sklearn

from .config import CLASSES, FEATURES_PATH, MODEL_INFO_PATH, MODEL_PATH
from .data import feature_metadata
from .schema import get_feature_columns


def export_model(
    model, training_frame, feature_set, metrics=None, strategy=None, best_params=None
):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    fitted_steps = [("prep", model.named_steps["prep"]), ("rf", model.named_steps["rf"])]
    from sklearn.pipeline import Pipeline

    inference_pipeline = Pipeline(fitted_steps)
    joblib.dump(inference_pipeline, MODEL_PATH, compress=3)
    FEATURES_PATH.write_text(
        json.dumps(
            feature_metadata(training_frame, feature_set), indent=2
        ),
        encoding="utf-8",
    )
    info = {
        "classes": list(inference_pipeline.classes_),
        "feature_set": feature_set,
        "n_features": len(get_feature_columns(feature_set)),
        "strategy": strategy,
        "seed": 42,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }
    if metrics:
        info["test_metrics"] = metrics
    if best_params:
        info["best_params"] = best_params
    MODEL_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return inference_pipeline
