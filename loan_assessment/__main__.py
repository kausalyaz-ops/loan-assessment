"""Command line entry point for the loan assessment workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .synthetic_data import make_demo_data
from .workflow import run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run loan modeling and customer segmentation.")
    parser.add_argument("--data", type=Path, help="Input applicant CSV; omit to run synthetic demo data.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--default-target", default="default")
    parser.add_argument("--loan-amount-target", default="requested_loan_amount")
    parser.add_argument("--id-column")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.data:
        if not args.data.is_file():
            parser.error(f"Input CSV does not exist: {args.data}")
        data = pd.read_csv(args.data)
        id_column = args.id_column
    else:
        data = make_demo_data(seed=args.seed)
        id_column = args.id_column or "applicant_id"
        demo_path = args.output_dir / "demo_loan_applicants.csv"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(demo_path, index=False)
        print(f"Generated synthetic demo data: {demo_path}")

    results = run_workflow(
        data,
        args.output_dir,
        default_target=args.default_target,
        loan_amount_target=args.loan_amount_target,
        id_column=id_column,
        clusters=args.clusters,
        seed=args.seed,
    )
    print(f"Workflow complete. Report: {Path(results['output_dir']) / 'report.md'}")


if __name__ == "__main__":
    main()
