"""Compare supervised models and explore applicant segments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .synthetic_data import make_demo_data
    from .task2 import (
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
        ORDERED_CATEGORIES,
        ORDERED_COLUMN,
        LoanFeatureEngineer,
    )
else:
    from synthetic_data import make_demo_data
    from task2 import (
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
        ORDERED_CATEGORIES,
        ORDERED_COLUMN,
        LoanFeatureEngineer,
    )

CLASSIFIER_NAMES = ("Logistic Regression", "Decision Tree", "Random Forest", "XGBoost")
REGRESSION_NUMERIC_COLUMNS = [
    "age",
    "annual_income",
    "credit_score",
    "debt_to_income",
    "employment_years",
]
REGRESSION_CATEGORICAL_COLUMNS = ["home_ownership", "loan_purpose"]
CLUSTER_COLUMNS = ["age", "annual_income", "credit_score"]


def _classifier_preprocessor() -> ColumnTransformer:
    """Create the task2 preprocessing branches for a new classifier."""
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
            (
                "ordered",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                categories=ORDERED_CATEGORIES,
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                ORDERED_COLUMN,
            ),
        ],
        remainder="drop",
    )


def _classifier_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        [
            ("features", LoanFeatureEngineer()),
            ("preprocess", _classifier_preprocessor()),
            ("classifier", estimator),
        ]
    )


def _regression_pipeline(estimator: Any) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                REGRESSION_NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                REGRESSION_CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessor), ("regressor", estimator)])


def _classifier_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data.drop(columns=["applicant_id", "default"]).copy()
    features["application_date"] = pd.date_range(
        "2022-01-01", periods=len(features), freq="D"
    )
    return features


def _run_clustering(
    data: pd.DataFrame, output_dir: Path
) -> tuple[list[dict[str, float | int]], pd.DataFrame]:
    cluster_features = data[CLUSTER_COLUMNS]
    cluster_preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    scaled_features = cluster_preprocessor.fit_transform(cluster_features)

    cluster_metrics: list[dict[str, float | int]] = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_features)
        cluster_metrics.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(scaled_features, labels)),
            }
        )

    figure, (elbow_axis, silhouette_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
    ks = [int(metric["k"]) for metric in cluster_metrics]
    elbow_axis.plot(
        ks, [float(metric["inertia"]) for metric in cluster_metrics], marker="o"
    )
    elbow_axis.set(
        title="K-Means elbow curve",
        xlabel="Number of clusters (k)",
        ylabel="Inertia",
    )
    silhouette_axis.plot(
        ks,
        [float(metric["silhouette"]) for metric in cluster_metrics],
        marker="o",
        color="darkorange",
    )
    silhouette_axis.set(
        title="Silhouette scores",
        xlabel="Number of clusters (k)",
        ylabel="Silhouette score",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "task3_elbow_silhouette.png", dpi=150)
    plt.close(figure)

    best_k = int(max(cluster_metrics, key=lambda metric: float(metric["silhouette"]))["k"])
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    segments = data[["applicant_id", "default", *CLUSTER_COLUMNS]].copy()
    segments["cluster"] = final_model.fit_predict(scaled_features)
    segments.to_csv(output_dir / "task3_segments.csv", index=False)
    summary = (
        segments.groupby("cluster", observed=True)
        .agg(
            applicants=("applicant_id", "size"),
            age=("age", "mean"),
            annual_income=("annual_income", "mean"),
            credit_score=("credit_score", "mean"),
            default_rate=("default", "mean"),
        )
        .round(3)
    )
    return cluster_metrics, summary


def run_demo(
    output_dir: Path = Path("outputs"),
    rows: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    """Train and compare classifiers/regressors and write the clustering plot."""
    data = make_demo_data(rows=rows, seed=seed)
    X = _classifier_features(data)
    y = data["default"]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            oob_score=True,
            bootstrap=True,
            random_state=seed,
            n_jobs=1,
        ),
    }

    fitted_classifiers: dict[str, Pipeline] = {}
    rows_for_table: list[dict[str, str | float]] = []
    for name, estimator in classifiers.items():
        pipeline = _classifier_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        validation_accuracy = float(accuracy_score(y_valid, pipeline.predict(X_valid)))
        fitted_classifiers[name] = pipeline
        rows_for_table.append(
            {"Model": name, "Validation score": f"Accuracy: {validation_accuracy:.3f}"}
        )

    forest = fitted_classifiers["Random Forest"].named_steps["classifier"]
    print(f"Random Forest OOB score: {forest.oob_score_:.3f}")

    logistic = fitted_classifiers["Logistic Regression"]
    logistic_predictions = logistic.predict(X_valid.iloc[:5])
    logistic_probabilities = logistic.predict_proba(X_valid.iloc[:5])
    print("\nLogistic Regression predictions for 5 validation rows:")
    print(
        pd.DataFrame(
            {
                "predict": logistic_predictions,
                "predict_proba": [
                    probability.tolist() for probability in logistic_probabilities
                ],
            },
            index=X_valid.index[:5],
        ).to_string()
    )

    xgb_pipeline = _classifier_pipeline(
        XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=20,
            random_state=seed,
            n_jobs=1,
        )
    )
    fitted_preprocessor = Pipeline(
        [
            ("features", LoanFeatureEngineer()),
            ("preprocess", _classifier_preprocessor()),
        ]
    )
    validation_matrix = fitted_preprocessor.fit(X_train, y_train).transform(X_valid)
    xgb_pipeline.fit(
        X_train,
        y_train,
        classifier__eval_set=[(validation_matrix, y_valid)],
        classifier__verbose=False,
    )
    xgb_accuracy = float(accuracy_score(y_valid, xgb_pipeline.predict(X_valid)))
    fitted_classifiers["XGBoost"] = xgb_pipeline
    rows_for_table.append(
        {"Model": "XGBoost", "Validation score": f"Accuracy: {xgb_accuracy:.3f}"}
    )

    for name in ("Random Forest", "XGBoost"):
        pipeline = fitted_classifiers[name]
        model = pipeline.named_steps["classifier"]
        importances = model.feature_importances_
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        important = (
            pd.Series(importances, index=feature_names)
            .sort_values(ascending=False)
            .head(5)
        )
        print(f"\n{name} top 5 features:")
        print(important.to_string(float_format=lambda value: f"{value:.4f}"))

    X_regression = data.drop(
        columns=["applicant_id", "default", "requested_loan_amount"]
    )
    y_regression = data["requested_loan_amount"]
    X_reg_train, X_reg_valid, y_reg_train, y_reg_valid = train_test_split(
        X_regression,
        y_regression,
        test_size=0.25,
        random_state=seed,
    )
    regression_models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=100.0, max_iter=10_000),
    }
    fitted_regressors: dict[str, Pipeline] = {}
    for name, estimator in regression_models.items():
        pipeline = _regression_pipeline(estimator)
        pipeline.fit(X_reg_train, y_reg_train)
        score = float(r2_score(y_reg_valid, pipeline.predict(X_reg_valid)))
        fitted_regressors[name] = pipeline
        rows_for_table.append(
            {"Model": name, "Validation score": f"R-squared: {score:.3f}"}
        )

    coefficient_frames = []
    for name, pipeline in fitted_regressors.items():
        coefficient_frames.append(
            pd.Series(
                pipeline.named_steps["regressor"].coef_,
                index=pipeline.named_steps["preprocess"].get_feature_names_out(),
                name=name,
            )
        )
    coefficient_table = pd.concat(coefficient_frames, axis=1).fillna(0)
    top_coefficient_names = (
        coefficient_table.abs().max(axis=1).sort_values(ascending=False).head(10).index
    )
    print("\nTop regression coefficients (standardized input features):")
    print(coefficient_table.loc[top_coefficient_names].round(2).to_string())
    lasso_zero_count = int(
        np.count_nonzero(
            np.isclose(fitted_regressors["Lasso"].named_steps["regressor"].coef_, 0.0)
        )
    )
    print(
        f"Lasso zero coefficients: {lasso_zero_count} of "
        f"{len(fitted_regressors['Lasso'].named_steps['regressor'].coef_)}"
    )

    cluster_metrics, segment_summary = _run_clustering(data, output_dir)
    best_cluster = max(cluster_metrics, key=lambda metric: float(metric["silhouette"]))
    rows_for_table.append(
        {
            "Model": f"K-Means (k={best_cluster['k']})",
            "Validation score": f"Silhouette: {float(best_cluster['silhouette']):.3f}",
        }
    )

    results = pd.DataFrame(rows_for_table)
    results.to_csv(output_dir / "task3_results.csv", index=False)
    print("\nResults table:")
    print(results.to_string(index=False))
    print("\nK-Means elbow/silhouette values:")
    print(
        pd.DataFrame(cluster_metrics).to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )
    print(f"\nSegment profiles for best silhouette k={best_cluster['k']}:")
    print(segment_summary.to_string())
    print(f"\nSaved plot: {output_dir / 'task3_elbow_silhouette.png'}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_demo(output_dir=args.output_dir, rows=args.rows, seed=args.seed)


if __name__ == "__main__":
    main()
