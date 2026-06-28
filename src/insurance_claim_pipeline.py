"""
Insurance Claim Risk Prediction Pipeline
----------------------------------------
A portfolio-ready data mining project for automobile insurance claim prediction.

Expected target column: OUTCOME
Expected classes: 0 = No Claim, 1 = Claim

Run:
  python src/insurance_claim_pipeline.py --data-path data/raw/car_insurance_claim.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TARGET_COL = "OUTCOME"
DROP_COLUMNS = ["ID"]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    for path in [TABLES_DIR, FIGURES_DIR, PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load CSV data and remove non-informative identifier columns."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Place the CSV file in data/raw/ or pass --data-path."
        )

    df = pd.read_csv(data_path)

    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=col)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' was not found in the dataset.")

    return df


def summarize_dataset(df: pd.DataFrame) -> None:
    """Save basic data quality summaries for documentation."""
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_percent": (df.isna().mean() * 100).round(2),
            "unique_values": df.nunique(dropna=True),
        }
    )
    summary.to_csv(TABLES_DIR / "data_quality_summary.csv", index=True)

    target_dist = df[TARGET_COL].value_counts(dropna=False).rename("count").to_frame()
    target_dist["percent"] = (target_dist["count"] / len(df) * 100).round(2)
    target_dist.to_csv(TABLES_DIR / "target_distribution.csv")


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target."""
    X = df.drop(columns=TARGET_COL)
    y = df[TARGET_COL]
    return X, y


def get_feature_types(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Identify numerical and categorical feature columns."""
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numerical_features, categorical_features


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Create preprocessing pipeline for numerical and categorical variables."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Evaluate a fitted model and save confusion matrix figure."""
    y_pred = pipeline.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    # ROC-AUC is available when the estimator exposes decision_function or predict_proba.
    try:
        if hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(X_test)
            metrics["roc_auc"] = roc_auc_score(y_test, scores)
        elif hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, scores)
        else:
            metrics["roc_auc"] = np.nan
    except Exception:
        metrics["roc_auc"] = np.nan

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(
        TABLES_DIR / f"classification_report_{model_name.lower().replace(' ', '_')}.csv"
    )

    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()

    return metrics


def save_tree_feature_importance(tree_pipeline: Pipeline) -> None:
    """Save top feature importances from the Decision Tree model."""
    preprocessor = tree_pipeline.named_steps["preprocess"]
    model = tree_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(TABLES_DIR / "feature_importance_decision_tree.csv", index=False)

    top_features = importance_df.head(10).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Feature Importance")
    plt.title("Top 10 Features - Decision Tree")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_features_decision_tree.png", dpi=150)
    plt.close()


def train_and_evaluate(df: pd.DataFrame) -> None:
    """Train Decision Tree and SVM models using leakage-safe pipelines."""
    summarize_dataset(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numerical_features, categorical_features = get_feature_types(X_train)

    # Decision Tree pipeline: no scaling required.
    tree_preprocessor = build_preprocessor(
        numerical_features, categorical_features, scale_numeric=False
    )
    tree_pipeline = Pipeline(
        steps=[
            ("preprocess", tree_preprocessor),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=5,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Linear SVM pipeline: scaling helps SVM performance.
    svm_preprocessor = build_preprocessor(
        numerical_features, categorical_features, scale_numeric=True
    )
    svm_pipeline = Pipeline(
        steps=[
            ("preprocess", svm_preprocessor),
            (
                "model",
                LinearSVC(
                    max_iter=20000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    tree_pipeline.fit(X_train, y_train)
    svm_pipeline.fit(X_train, y_train)

    metrics = [
        evaluate_model("Decision Tree", tree_pipeline, X_test, y_test),
        evaluate_model("Linear SVM", svm_pipeline, X_test, y_test),
    ]

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(TABLES_DIR / "model_metrics.csv", index=False)

    save_tree_feature_importance(tree_pipeline)

    # Hyperparameter tuning for SVM using only training data.
    svm_grid = GridSearchCV(
        estimator=svm_pipeline,
        param_grid={"model__C": [0.1, 1.0, 10.0]},
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    svm_grid.fit(X_train, y_train)

    grid_results = pd.DataFrame(svm_grid.cv_results_)
    grid_results.to_csv(TABLES_DIR / "svm_gridsearch_results.csv", index=False)

    best_params = pd.DataFrame([svm_grid.best_params_])
    best_params["best_cv_score_f1"] = svm_grid.best_score_
    best_params.to_csv(TABLES_DIR / "svm_best_params.csv", index=False)

    print("Modeling completed successfully.")
    print(metrics_df)
    print("SVM best params:", svm_grid.best_params_)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insurance claim risk prediction pipeline")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "car_insurance_claim.csv",
        help="Path to the input CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    df = load_dataset(args.data_path)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
