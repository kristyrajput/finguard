from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed"

columns = [
    "type",
    "sender_balance_diff",
    "receiver_balance_diff",
    "sender_balance_error",
    "receiver_balance_error",
    "is_zero_sender_balance",
    "amount_log",
    "isFraud",
]
files = sorted(DATA_PATH.glob("features_part_*.csv"))
df = pd.concat(
    [pd.read_csv(file, usecols=columns) for file in files],
    ignore_index=True,
)

X = pd.get_dummies(
    df.drop(columns="isFraud"),
    columns=["type"],
    dtype="int8",
)
y = df["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training Logistic Regression...")

model = LogisticRegression(
    class_weight="balanced",
    max_iter=200,
    random_state=42,
)
model.fit(X_train, y_train)

# Default classification threshold: 0.5
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))
print("--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))
print("\nPR-AUC (Average Precision):",
      round(average_precision_score(y_test, y_prob), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

MODEL_PATH = PROJECT_ROOT / "logistic_regression_model.joblib"
joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
