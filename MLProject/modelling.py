import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).parent / "bank_transactions_data_preprocessing.csv"
TARGET_COL = "TransactionType_Debit"
EXPERIMENT_NAME = "smsml-basic"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    drop_cols = [
        "TransactionID",
        "AccountID",
        "TransactionDate",
        "PreviousTransactionDate",
    ]

    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    return X, y


def main() -> None:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_models=True)

    df = load_data(DATA_PATH)
    X, y = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="rf_baseline"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]

        mlflow.log_metric("test_accuracy", float(accuracy_score(y_test, preds)))
        mlflow.log_metric("test_f1", float(f1_score(y_test, preds)))
        mlflow.log_metric("test_roc_auc", float(roc_auc_score(y_test, probas)))


if __name__ == "__main__":
    main()
