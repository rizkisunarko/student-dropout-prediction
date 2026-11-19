import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

if __package__:
    from .config import (
        CLASSES,
        FEATURE_SET,
        N_SPLITS,
        ROOT,
        SEED,
        TEST_SIZE,
    )
    from .data import load_data
    from .evaluate import evaluate_model
    from .export import export_model
    from .pipeline import build_model
    from .schema import FEATURES, TARGET, get_feature_columns
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import (
        CLASSES,
        FEATURE_SET,
        N_SPLITS,
        ROOT,
        SEED,
        TEST_SIZE,
    )
    from src.data import load_data
    from src.evaluate import evaluate_model
    from src.export import export_model
    from src.pipeline import build_model
    from src.schema import FEATURES, TARGET, get_feature_columns


def recall_for(label):
    return make_scorer(
        recall_score, labels=[label], average="macro", zero_division=0
    )


SCORING = {
    "macro_f1": "f1_macro",
    "balanced_acc": "balanced_accuracy",
    "accuracy": "accuracy",
    "recall_enrolled": recall_for("Enrolled"),
    "recall_dropout": recall_for("Dropout"),
}


def _columns(columns):
    nominal = [n for n, _, kind, _ in FEATURES if n in columns and kind == "nominal"]
    binary = [n for n, _, kind, _ in FEATURES if n in columns and kind == "binary"]
    numeric = [n for n in columns if n not in nominal + binary]
    return nominal, binary, numeric


def train():
    frame = load_data()
    columns = get_feature_columns(FEATURE_SET)
    X, y = frame[columns], frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "split_indices.json").write_text(
        json.dumps({"train": X_train.index.tolist(), "test": X_test.index.tolist()}),
        encoding="utf-8",
    )
    nominal, binary, numeric = _columns(columns)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    cv_rows = []
    fitted = {}
    for name, strategy in (("E1", "baseline"), ("E2", "class_weight"), ("E3", "smotenc")):
        model = build_model(strategy, nominal, binary, numeric, seed=SEED)
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        for metric in SCORING:
            cv_rows.append({
                "experiment": name, "strategy": strategy, "metric": metric,
                "mean": scores[f"test_{metric}"].mean(),
                "std": scores[f"test_{metric}"].std(),
            })
        model.fit(X_train, y_train)
        fitted[name] = model
    cv_frame = pd.DataFrame(cv_rows)
    metrics_dir = ROOT / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cv_frame.to_csv(metrics_dir / "cv_E1_E3.csv", index=False)
    best_strategy = (
        cv_frame[cv_frame["metric"] == "macro_f1"]
        .sort_values("mean", ascending=False)
        .iloc[0]["strategy"]
    )
    tuning_base = build_model(best_strategy, nominal, binary, numeric, seed=SEED)
    param_grid = {
        "rf__n_estimators": [100, 200, 300],
        "rf__max_depth": [None, 10, 20, 30],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        tuning_base, param_grid, n_iter=40, cv=cv, scoring="f1_macro",
        random_state=SEED, n_jobs=-1, refit=True, return_train_score=True
    )
    search.fit(X_train, y_train)
    fitted["E4"] = search.best_estimator_
    pd.DataFrame(search.cv_results_).to_csv(metrics_dir / "e4_tuning.csv", index=False)
    all_test_rows = []
    for name, model in fitted.items():
        all_test_rows.extend(evaluate_model(model, X_test, y_test, name, ROOT / "reports"))
    test_frame = pd.DataFrame(all_test_rows)
    test_frame.to_csv(metrics_dir / "test_E1_E4.csv", index=False)
    final = export_model(
        fitted["E4"], X_train, FEATURE_SET,
        metrics={"macro_f1": float(test_frame[
            (test_frame.experiment == "E4") & (test_frame["class"] == "overall")
        ].iloc[0]["macro_f1"])},
        strategy=best_strategy,
        best_params=search.best_params_,
    )
    return final


if __name__ == "__main__":
    train()
