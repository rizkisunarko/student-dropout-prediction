from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

if __package__:
    from .config import CLASSES
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import CLASSES


def evaluate_model(model, X_test, y_test, name, output_dir):
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=CLASSES, zero_division=0
    )
    rows = [
        {
            "experiment": name,
            "class": label,
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i],
        }
        for i, label in enumerate(CLASSES)
    ]
    rows.append(
        {
            "experiment": name,
            "class": "overall",
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "macro_f1": f1_score(
                y_test, predictions, labels=CLASSES, average="macro"
            ),
        }
    )
    pd.DataFrame(rows).to_csv(metrics_dir / f"test_{name}.csv", index=False)
    matrix = confusion_matrix(y_test, predictions, labels=CLASSES)
    for normalized, suffix in ((False, "counts"), (True, "normalized")):
        values = matrix.astype(float)
        if normalized:
            values = values / values.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(values, cmap="Blues")
        fig.colorbar(image, ax=ax)
        ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)),
               xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Predicted", ylabel="Actual",
               title=f"{name} Confusion Matrix ({suffix})")
        for row in range(len(CLASSES)):
            for col in range(len(CLASSES)):
                text = f"{values[row, col]:.2f}" if normalized else str(matrix[row, col])
                ax.text(col, row, text, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(figures_dir / f"confusion_{name}_{suffix}.png", dpi=150)
        plt.close(fig)
    return rows


def compute_feature_importance(model, X_test, y_test, output_dir, seed=42):
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rf = model.named_steps["rf"]
    prep = model.named_steps["prep"]
    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        feature_names = np.array(X_test.columns)

    mdi_importances = rf.feature_importances_

    # Permutation importance on test set (n_jobs=1 for safe Windows execution)
    perm = permutation_importance(
        model, X_test, y_test, scoring="f1_macro", n_repeats=5, random_state=seed, n_jobs=1
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mdi_importance": mdi_importances,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }).sort_values("mdi_importance", ascending=False)

    importance_df.to_csv(metrics_dir / "feature_importance.csv", index=False)

    # Plot Top 15 Features (MDI and Permutation)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    top_mdi = importance_df.head(15).iloc[::-1]
    ax1.barh(top_mdi["feature"], top_mdi["mdi_importance"], color="#3b82f6")
    ax1.set_title("Top 15 Features - MDI (Gini Importance)", fontsize=12)
    ax1.set_xlabel("Mean Decrease in Impurity")

    top_perm = importance_df.sort_values("perm_importance_mean", ascending=False).head(15).iloc[::-1]
    ax2.barh(top_perm["feature"], top_perm["perm_importance_mean"], xerr=top_perm["perm_importance_std"], color="#10b981", capsize=3)
    ax2.set_title("Top 15 Features - Permutation Importance (Test F1 Macro)", fontsize=12)
    ax2.set_xlabel("Mean F1-Macro Drop")

    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    return importance_df


def classification_report_frame(model, X_test, y_test):
    return pd.DataFrame(
        classification_report(
            y_test, model.predict(X_test), labels=CLASSES,
            output_dict=True, zero_division=0
        )
    ).T
