# Loan Assessment ML Workflow

A reproducible workflow for loan-default classification, requested-loan-amount
regression, and customer segmentation. The project accepts a trainer-provided
CSV and keeps both target columns out of model features to avoid target leakage.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the included demo

The demo creates a deterministic sample dataset, excludes its generated
`applicant_id` identifier from the features, and runs all three tasks:

```powershell
python -m loan_assessment
```

## Preprocessing, features, and pipeline example

Run a focused example that compares median and KNN numeric imputation using
stratified cross-validation. Feature engineering, imputation, scaling, and
encoding are all fitted inside each fold. The best-scoring pipeline is saved
with joblib, loaded again, and used to predict three new raw applicant rows:

```powershell
python -m loan_assessment.task2
```

The example decomposes `application_date`, adds an income-to-loan ratio and a
log-income feature, one-hot encodes nominal categories, and ordinal-encodes an
ordered credit-score band. It writes
`outputs/loan_default_pipeline.joblib`. The reported accuracy is an example
metric on synthetic data, not a lending-performance guarantee.

## Run with a trainer-provided dataset

The CSV must contain a binary target and a numeric requested-loan-amount
target. By default their column names are `default` and `requested_loan_amount`.
All remaining columns are considered candidate customer features. Optionally
exclude an identifier column from the analysis:

```powershell
python -m loan_assessment --data path\to\applicants.csv --id-column applicant_id
```

Change target names, output location, cluster count, or random seed as needed:

```powershell
python -m loan_assessment `
  --data path\to\applicants.csv `
  --default-target loan_status `
  --loan-amount-target loan_amount `
  --id-column customer_id `
  --clusters 4 `
  --output-dir outputs `
  --seed 42
```

The default-classification target must have exactly two classes, and the
requested-amount target must be numeric. Rows with missing target values are
excluded from the corresponding supervised task. Missing feature values are
imputed within each training pipeline.

## Workflow and outputs

1. Inspect the data, target distributions, and missing values.
2. Build preprocessing pipelines for numeric and categorical features.
3. Compare logistic regression and random forest classifiers using ROC AUC;
   report accuracy, precision, recall, F1, and a confusion matrix.
4. Train a random forest regressor and report MAE, RMSE, and R-squared.
5. Scale/encode customer features, cluster customers with K-means, and save
   cluster assignments and a two-dimensional visualization.

Generated artifacts are written under `outputs/`:

- `data_overview.json` and `report.md`
- `classification_metrics.json`, `classification_predictions.csv`, and the
  confusion-matrix and ROC-curve plots
- `regression_metrics.json` and `regression_predictions.csv`
- `customer_segments.csv` and `customer_segments.png`
- `models.joblib`

The demo data is synthetic and is only for exercising the workflow; it is not
appropriate for lending decisions. Metrics from a trainer dataset should be
interpreted with the dataset's class balance, collection process, and evaluation
split in mind.

## Tests

```powershell
python -m unittest discover -s tests -v
```
