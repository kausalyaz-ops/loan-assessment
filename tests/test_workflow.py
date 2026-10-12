from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from loan_assessment.synthetic_data import make_demo_data
from loan_assessment.workflow import run_workflow


class WorkflowTests(unittest.TestCase):
    def test_demo_runs_all_three_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            results = run_workflow(
                make_demo_data(rows=160),
                temporary_directory,
                id_column="applicant_id",
                clusters=3,
            )
            output = Path(temporary_directory)

            self.assertEqual(results["classification"]["target"], "default")
            self.assertIn(
                results["classification"]["selected_model"],
                results["classification"]["validation_roc_auc_by_model"],
            )
            self.assertGreaterEqual(results["regression"]["mae"], 0)
            self.assertEqual(sum(results["segmentation"]["cluster_sizes"].values()), 160)
            for filename in (
                "report.md",
                "classification_metrics.json",
                "regression_metrics.json",
                "customer_segments.csv",
                "models.joblib",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            with (output / "classification_metrics.json").open(encoding="utf-8") as file:
                self.assertIn("confusion_matrix", json.load(file))

    def test_demo_data_is_reproducible(self) -> None:
        first = make_demo_data(rows=80, seed=7)
        second = make_demo_data(rows=80, seed=7)
        self.assertTrue(first.equals(second))


if __name__ == "__main__":
    unittest.main()
