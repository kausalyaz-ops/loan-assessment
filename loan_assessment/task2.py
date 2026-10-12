"""Demonstrate leakage-safe loan preprocessing and feature engineering."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

if __package__:
    from .synthetic_data import make_demo_data
else:
    from synthetic_data import make_demo_data

NUMERIC_COLUMNS = [
    "age",
    "annual_income",
    "credit_score",
    "debt_to_income",
    "employment_years",
    "requested_loan_amount",
    "application_year",
    "application_month",
    "application_day",
    "income_to_loan_ratio",
    "log_annual_income",
]
CATEGORICAL_COLUMNS = ["home_ownership", "loan_purpose"]
ORDERED_COLUMN = ["credit_band"]
ORDERED_CATEGORIES = [["low", "medium", "high"]]
REQUIRED_COLUMNS = [
    "age",
    "annual_income",
    "credit_score",
    "debt_to_income",
    "employment_years",
    "home_ownership",
    "loan_purpose",
    "requested_loan_amount",
    "application_date",
]


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add date, affordability, log-income, and ordered credit-band features."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> LoanFeatureEngineer:
        missing_columns = sorted(set(REQUIRED_COLUMNS).difference(X.columns))
        if missing_columns:
            raise ValueError(f"Missing required input columns: {missing_columns}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing_columns = sorted(set(REQUIRED_COLUMNS).difference(X.columns))
        if missing_columns:
            raise ValueError(f"Missing required input columns: {missing_columns}")

        features = X.copy()
        application_date = pd.to_datetime(features["application_date"], errors="raise")
        features["application_year"] = application_date.dt.year
        features["application_month"] = application_date.dt.month
        features["application_day"] = application_date.dt.day
        loan_amount = features["requested_loan_amount"]
        features["income_to_loan_ratio"] = (
            features["annual_income"] / loan_amount.where(loan_amount > 0)
        )
        features["log_annual_income"] = np.log1p(features["annual_income"])
        features["credit_band"] = pd.cut(
            features["credit_score"],
            bins=[-np.inf, 579, 669, np.inf],
            labels=["low", "medium", "high"],
        )
        return features


def make_loan_pipeline(numeric_imputer: str = "median") -> Pipeline:
    """Create the complete estimator so preprocessing is fit within each fold."""
    if numeric_imputer == "median":
        imputer = SimpleImputer(strategy="median")
    elif numeric_imputer == "knn":
        imputer = KNNImputer(n_neighbors=5)
    else:
        raise ValueError("numeric_imputer must be either 'median' or 'knn'")

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", imputer),
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
                            OrdinalEncoder(
                                categories=ORDERED_CATEGORIES,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                ORDERED_COLUMN,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", LoanFeatureEngineer()),
            ("preprocess", preprocess),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _new_raw_applicants() -> pd.DataFrame:
    """Return three raw records with the same schema as the training features."""
    return pd.DataFrame(
        [
            {
                "applicant_id": "NEW001",
                "age": 34,
                "annual_income": 72_000.0,
                "credit_score": 710,
                "debt_to_income": 0.22,
                "employment_years": 8.0,
                "home_ownership": "mortgage",
                "loan_purpose": "home_improvement",
                "requested_loan_amount": 18_000.0,
                "application_date": "2025-01-15",
            },
            {
                "applicant_id": "NEW002",
                "age": 27,
                "annual_income": 41_000.0,
                "credit_score": 590,
                "debt_to_income": 0.48,
                "employment_years": 2.0,
                "home_ownership": "rent",
                "loan_purpose": "debt_consolidation",
                "requested_loan_amount": 24_000.0,
                "application_date": "2025-02-03",
            },
            {
                "applicant_id": "NEW003",
                "age": 51,
                "annual_income": 105_000.0,
                "credit_score": 780,
                "debt_to_income": 0.16,
                "employment_years": 19.0,
                "home_ownership": "own",
                "loan_purpose": "auto",
                "requested_loan_amount": 12_000.0,
                "application_date": "2025-03-20",
            },
        ]
    )


def run_demo(
    output_path: Path = Path("outputs/loan_default_pipeline.joblib"),
    rows: int = 600,
    seed: int = 42,
) -> tuple[dict[str, float], np.ndarray]:
    """Compare imputation choices, save the best pipeline, and reload to predict."""
    data = make_demo_data(rows=rows, seed=seed)
    data["application_date"] = pd.date_range("2022-01-01", periods=rows, freq="D")
    y = data["default"]
    X = data.drop(columns=["default", "applicant_id"])

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores: dict[str, float] = {}
    for strategy in ("median", "knn"):
        pipeline = make_loan_pipeline(numeric_imputer=strategy)
        fold_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=folds,
            scoring="accuracy",
            n_jobs=1,
        )
        scores[strategy] = float(fold_scores.mean())

    best_strategy = max(scores, key=scores.get)
    fitted_pipeline = make_loan_pipeline(numeric_imputer=best_strategy)
    fitted_pipeline.fit(X, y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_pipeline, output_path)
    loaded_pipeline = joblib.load(output_path)
    new_features = _new_raw_applicants().drop(columns="applicant_id")
    predictions = loaded_pipeline.predict(new_features)

    print(f"Median-imputation CV accuracy: {scores['median']:.2f}")
    print(f"KNN-imputation CV accuracy: {scores['knn']:.2f}")
    print(f"Selected numeric imputer: {best_strategy}")
    print(f"Pipeline CV accuracy: {scores[best_strategy]:.2f}")
    print(f"Predictions for 3 new rows: {predictions.tolist()}")
    print(f"Saved and reloaded pipeline: {output_path}")
    return scores, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/loan_default_pipeline.joblib"),
    )
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_demo(output_path=args.output, rows=args.rows, seed=args.seed)


if __name__ == "__main__":
    main()
