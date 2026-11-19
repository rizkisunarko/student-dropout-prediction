from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder


def build_preprocessor(nominal, binary, numeric):
    return ColumnTransformer(
        [
            (
                "nom",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                nominal,
            ),
            ("bin", "passthrough", binary),
            ("num", "passthrough", numeric),
        ],
        verbose_feature_names_out=False,
    )


def build_model(
    strategy, nominal, binary, numeric, rf_params=None, seed=42
):
    if strategy not in {"baseline", "class_weight", "smotenc"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    rf_options = dict(rf_params or {})
    rf_options.setdefault("random_state", seed)
    if strategy == "class_weight":
        rf_options["class_weight"] = "balanced"
    else:
        rf_options.setdefault("class_weight", None)
    prep = build_preprocessor(nominal, binary, numeric)
    steps = [("prep", prep)]
    if strategy == "smotenc":
        steps.append(
            (
                "smote",
                SMOTENC(
                    categorical_features=list(range(len(nominal) + len(binary))),
                    random_state=seed,
                    sampling_strategy="auto",
                    k_neighbors=5,
                ),
            )
        )
    steps.append(("rf", RandomForestClassifier(**rf_options)))
    return ImbPipeline(steps)
