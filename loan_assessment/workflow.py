"""Training, evaluation, and reporting for the loan assessment workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _feature_preprocessor(features: pd.DataFrame, *, scale_numeric: bool = False):
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def _validate_dataset(
    data: pd.DataFrame,
    default_target: str,
    loan_amount_target: str,
    id_column: str | None,
) -> list[str]:
    if data.empty:
        raise ValueError("The input CSV has no rows.")
    if data.columns.duplicated().any():
        raise ValueError("The input CSV contains duplicate column names.")
    required = [default_target, loan_amount_target]
    if id_column:
        required.append(id_column)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Required column(s) not found: {', '.join(missing)}")

    excluded = {default_target, loan_amount_target}
    if id_column:
        excluded.add(id_column)
    feature_columns = [column for column in data.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No feature columns remain after excluding targets and identifier.")
    supported_types = ["number", "bool", "object", "category", "string"]
    if data[feature_columns].select_dtypes(include=supported_types).shape[1] != len(feature_columns):
        raise ValueError("All feature columns must be numeric, boolean, or categorical.")
    return feature_columns


def _plot_eda(data: pd.DataFrame, default_target: str, output_dir: Path) -> None:
    target_counts = data[default_target].fillna("missing").astype(str).value_counts()
    fig, axis = plt.subplots(figsize=(6, 4))
    target_counts.plot.bar(ax=axis, color="#4472C4")
    axis.set(title=f"{default_target} distribution", xlabel="Class", ylabel="Rows")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_overview.png", dpi=140)
    plt.close(fig)

    numeric = data.select_dtypes(include="number").drop(columns=[default_target], errors="ignore")
    if numeric.empty:
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.text(0.5, 0.5, "No numeric features", ha="center", va="center")
        axis.set_axis_off()
        fig.savefig(output_dir / "numeric_distributions.png", dpi=140)
        plt.close(fig)
    else:
        numeric.hist(bins=20, color="#70AD47", figsize=(10, 8))
        plt.tight_layout()
        plt.savefig(output_dir / "numeric_distributions.png", dpi=140)
        plt.close()


def _train_classifier(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    id_column: str | None,
    output_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], Pipeline, pd.DataFrame]:
    usable = data.loc[data[target_column].notna()].copy()
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(usable[target_column].astype(str))
    counts = np.bincount(target)
    if len(label_encoder.classes_) != 2:
        raise ValueError(f"Classification target '{target_column}' must contain exactly two classes.")
    if counts.min() < 2:
        raise ValueError("Each classification class needs at least two non-missing rows.")

    test_rows = max(2, int(np.ceil(len(usable) * 0.25)))
    if test_rows >= len(usable) - 1:
        raise ValueError("Classification needs enough labeled rows for both train and test sets.")
    x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
        usable[feature_columns],
        target,
        usable.index,
        test_size=test_rows,
        random_state=seed,
        stratify=target,
    )
    preprocessor = _feature_preprocessor(x_train, scale_numeric=True)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
    }

    fitted: dict[str, Pipeline] = {}
    scores: dict[str, float] = {}
    for name, estimator in candidates.items():
        model = Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])
        model.fit(x_train, y_train)
        scores[name] = float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))
        fitted[name] = model

    selected_name = max(scores, key=scores.get)
    selected = fitted[selected_name]
    predictions = selected.predict(x_test)
    probabilities = selected.predict_proba(x_test)[:, 1]
    metrics = {
        "target": target_column,
        "positive_class": str(label_encoder.classes_[1]),
        "selected_model": selected_name,
        "validation_roc_auc_by_model": scores,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": [[int(value) for value in row] for row in confusion_matrix(y_test, predictions)],
    }
    _write_json(output_dir / "classification_metrics.json", metrics)

    prediction_data = pd.DataFrame(
        {
            "row_index": test_idx,
            "actual": label_encoder.inverse_transform(y_test),
            "predicted": label_encoder.inverse_transform(predictions),
            "default_probability": probabilities,
        }
    )
    if id_column:
        prediction_data.insert(1, id_column, data.loc[test_idx, id_column].to_numpy())
    prediction_data.to_csv(output_dir / "classification_predictions.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=label_encoder.classes_,
        ax=axes[0],
        colorbar=False,
    )
    axes[0].set_title("Classification confusion matrix")
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    auc = metrics["validation_roc_auc_by_model"][selected_name]
    axes[1].plot(false_positive_rate, true_positive_rate, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "classification_evaluation.png", dpi=140)
    plt.close(fig)
    return metrics, selected, prediction_data


def _train_regressor(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    id_column: str | None,
    output_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], Pipeline]:
    numeric_target = pd.to_numeric(data[target_column], errors="coerce")
    usable = data.loc[numeric_target.notna() & np.isfinite(numeric_target)].copy()
    if len(usable) < 10:
        raise ValueError(f"Regression target '{target_column}' needs at least 10 numeric rows.")

    x_train, x_test, y_train, y_test, _, test_idx = train_test_split(
        usable[feature_columns],
        numeric_target.loc[usable.index],
        usable.index,
        test_size=0.25,
        random_state=seed,
    )
    model = Pipeline(
        [
            ("preprocessor", _feature_preprocessor(x_train)),
            (
                "model",
                RandomForestRegressor(n_estimators=250, random_state=seed, n_jobs=-1),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "target": target_column,
        "model": "random_forest",
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "evaluation_rows": int(len(y_test)),
    }
    _write_json(output_dir / "regression_metrics.json", metrics)

    prediction_data = pd.DataFrame(
        {"row_index": test_idx, "actual": y_test.to_numpy(), "predicted": predictions}
    )
    if id_column:
        prediction_data.insert(1, id_column, data.loc[test_idx, id_column].to_numpy())
    prediction_data.to_csv(output_dir / "regression_predictions.csv", index=False)
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(y_test, predictions, alpha=0.65)
    low = float(min(y_test.min(), predictions.min()))
    high = float(max(y_test.max(), predictions.max()))
    axis.plot([low, high], [low, high], linestyle="--", color="gray")
    axis.set(xlabel="Actual amount", ylabel="Predicted amount", title="Loan amount predictions")
    fig.tight_layout()
    fig.savefig(output_dir / "regression_evaluation.png", dpi=140)
    plt.close(fig)
    return metrics, model


def _segment_customers(
    data: pd.DataFrame,
    feature_columns: list[str],
    id_column: str | None,
    output_dir: Path,
    clusters: int,
    seed: int,
) -> tuple[dict[str, Any], Pipeline]:
    if len(data) < clusters:
        raise ValueError(f"Customer segmentation needs at least {clusters} rows.")
    features = data[feature_columns]
    preprocessor = _feature_preprocessor(features, scale_numeric=True)
    transformed = preprocessor.fit_transform(features)
    model = KMeans(n_clusters=clusters, n_init=10, random_state=seed)
    labels = model.fit_predict(transformed)

    segmentation = pd.DataFrame({"row_index": data.index, "cluster": labels})
    if id_column:
        segmentation.insert(1, id_column, data[id_column].to_numpy())
    segmentation.to_csv(output_dir / "customer_segments.csv", index=False)

    if transformed.shape[1] == 0:
        raise ValueError("No usable feature columns remain after preprocessing for segmentation.")
    if transformed.shape[1] == 1:
        coordinates = np.column_stack((transformed[:, 0], np.zeros(len(transformed))))
    else:
        coordinates = PCA(n_components=2, random_state=seed).fit_transform(transformed)
    fig, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="tab10", alpha=0.75)
    axis.set(title="Customer segments (PCA projection)", xlabel="Component 1", ylabel="Component 2")
    fig.colorbar(scatter, ax=axis, label="Cluster")
    fig.tight_layout()
    fig.savefig(output_dir / "customer_segments.png", dpi=140)
    plt.close(fig)

    summary = {
        "algorithm": "kmeans",
        "clusters": int(clusters),
        "rows": int(len(data)),
        "cluster_sizes": {
            str(cluster): int(count)
            for cluster, count in pd.Series(labels).value_counts().sort_index().items()
        },
    }
    _write_json(output_dir / "segmentation_metrics.json", summary)
    return summary, Pipeline([("preprocessor", preprocessor), ("model", model)])


def run_workflow(
    data: pd.DataFrame,
    output_dir: str | Path,
    *,
    default_target: str = "default",
    loan_amount_target: str = "requested_loan_amount",
    id_column: str | None = None,
    clusters: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Run EDA, two supervised tasks, and unsupervised customer segmentation."""
    if clusters < 2:
        raise ValueError("The number of clusters must be at least 2.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    feature_columns = _validate_dataset(data, default_target, loan_amount_target, id_column)

    overview = {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "feature_columns": feature_columns,
        "missing_values": {str(k): int(v) for k, v in data.isna().sum().items()},
        "classification_target_distribution": {
            str(k): int(v)
            for k, v in data[default_target].value_counts(dropna=False).items()
        },
    }
    _write_json(output_path / "data_overview.json", overview)
    _plot_eda(data, default_target, output_path)

    classification, classifier, _ = _train_classifier(
        data, feature_columns, default_target, id_column, output_path, seed
    )
    regression, regressor = _train_regressor(
        data, feature_columns, loan_amount_target, id_column, output_path, seed
    )
    segmentation, segmenter = _segment_customers(
        data, feature_columns, id_column, output_path, clusters, seed
    )
    joblib.dump(
        {
            "classification": classifier,
            "regression": regressor,
            "segmentation": segmenter,
            "feature_columns": feature_columns,
            "default_target": default_target,
            "loan_amount_target": loan_amount_target,
        },
        output_path / "models.joblib",
    )

    report = (
        "# Loan Assessment Workflow Report\n\n"
        "## Data overview\n\n"
        f"- Rows: {overview['rows']}\n"
        f"- Candidate features: {len(feature_columns)}\n"
        f"- Classification target: `{default_target}`\n"
        f"- Regression target: `{loan_amount_target}`\n\n"
        "## Default classification\n\n"
        f"- Selected model: `{classification['selected_model']}`\n"
        f"- ROC AUC: {classification['validation_roc_auc_by_model'][classification['selected_model']]:.3f}\n"
        f"- Accuracy / precision / recall / F1: {classification['accuracy']:.3f} / "
        f"{classification['precision']:.3f} / {classification['recall']:.3f} / "
        f"{classification['f1']:.3f}\n\n"
        "## Requested loan amount regression\n\n"
        f"- MAE: {regression['mae']:.2f}\n"
        f"- RMSE: {regression['rmse']:.2f}\n"
        f"- R-squared: {regression['r2']:.3f}\n\n"
        "## Customer segmentation\n\n"
        f"- Algorithm: K-means ({segmentation['clusters']} clusters)\n"
        f"- Cluster sizes: `{json.dumps(segmentation['cluster_sizes'], sort_keys=True)}`\n\n"
        "These results are an evaluation exercise, not a lending decision system. "
        "Validate data quality, fairness, calibration, and operational requirements "
        "before any real-world use.\n"
    )
    (output_path / "report.md").write_text(report, encoding="utf-8")
    return {
        "overview": overview,
        "classification": classification,
        "regression": regression,
        "segmentation": segmentation,
        "output_dir": str(output_path),
    }
