"""Trains an XGBoost classifier to predict churn probability from aggregated
session features, and reports real held-out accuracy/AUC.

Usage: .venv10projects/bin/python ml/train_model.py
"""
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = "data/training_data.csv"
MODEL_PATH = "models/churn_model.json"

FEATURE_COLS = [
    "session_duration",
    "movement_events",
    "deaths",
    "menu_opens",
    "menu_time",
    "time_since_last_death",
]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df["churn_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    print(f"[train_model] held-out accuracy={acc:.4f} auc={auc:.4f} "
          f"(train={len(X_train)}, test={len(X_test)})")

    model.save_model(MODEL_PATH)
    print(f"[train_model] saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
