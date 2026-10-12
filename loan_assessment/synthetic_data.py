"""Generate a small deterministic dataset for demonstrating the workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_data(rows: int = 600, seed: int = 42) -> pd.DataFrame:
    """Create plausible, explicitly synthetic loan-applicant records."""
    if rows < 40:
        raise ValueError("Demo data needs at least 40 rows for stratified evaluation.")

    rng = np.random.default_rng(seed)
    age = rng.integers(21, 71, size=rows)
    income = np.clip(rng.lognormal(mean=10.8, sigma=0.55, size=rows), 18_000, 240_000)
    credit_score = np.clip(rng.normal(675, 55, size=rows), 420, 850).round().astype(int)
    debt_to_income = np.clip(rng.beta(2.2, 5.0, size=rows), 0.01, 0.75)
    employment_years = np.clip(rng.gamma(2.2, 3.0, size=rows), 0, 40).round(1)
    home_ownership = rng.choice(
        ["rent", "mortgage", "own"], size=rows, p=[0.42, 0.43, 0.15]
    )
    loan_purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "medical", "auto", "other"],
        size=rows,
        p=[0.36, 0.2, 0.12, 0.18, 0.14],
    )

    amount = (
        2_500
        + 0.14 * income
        + 13_000 * debt_to_income
        + rng.normal(0, 5_000, size=rows)
    )
    amount *= np.where(loan_purpose == "home_improvement", 1.25, 1.0)
    requested_loan_amount = np.clip(amount, 1_000, 60_000).round(2)

    default_logit = (
        -1.8
        + 3.2 * debt_to_income
        - 0.008 * (credit_score - 650)
        - 0.025 * employment_years
        + np.where(home_ownership == "rent", 0.25, 0.0)
        + np.where(loan_purpose == "medical", 0.2, 0.0)
    )
    default_probability = 1 / (1 + np.exp(-default_logit))
    default = rng.binomial(1, default_probability)

    data = pd.DataFrame(
        {
            "applicant_id": [f"A{i:05d}" for i in range(1, rows + 1)],
            "age": age,
            "annual_income": income.round(2),
            "credit_score": credit_score,
            "debt_to_income": debt_to_income.round(3),
            "employment_years": employment_years,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "requested_loan_amount": requested_loan_amount,
            "default": default,
        }
    )

    for column in ("annual_income", "employment_years", "loan_purpose"):
        missing = rng.choice(rows, size=max(1, rows // 100), replace=False)
        data.loc[missing, column] = np.nan
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo_loan_applicants.csv"))
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_demo_data(rows=args.rows, seed=args.seed).to_csv(args.output, index=False)
    print(f"Wrote {args.rows} synthetic applicants to {args.output}")


if __name__ == "__main__":
    main()
