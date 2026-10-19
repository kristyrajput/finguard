from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "random_forest_final.joblib"

THRESHOLD = 0.95

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

if not files:
    raise FileNotFoundError("Processed feature files not found.")

print("Loading processed data...")

df = pd.concat(
    [pd.read_csv(file, usecols=columns) for file in files],
    ignore_index=True,
)

# Recreate the same split used during training
development, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["isFraud"],
)

del df, development

X_test = test.drop(columns=["isFraud"])
y_test = test["isFraud"]

del test

# Encode transaction type
X_test = pd.get_dummies(
    X_test,
    columns=["type"],
    dtype="int8",
)

# Load the final trained model
model = joblib.load(MODEL_PATH)

# Ensure the test features match training features
X_test = X_test.reindex(
    columns=model.feature_names_in_,
    fill_value=0,
)

print("Calculating test probabilities...")

probabilities = model.predict_proba(X_test)[:, 1]

# Apply the threshold selected using validation data
predictions = (probabilities >= THRESHOLD).astype(int)

print(f"\n--- FINAL TEST RESULTS (Threshold = {THRESHOLD}) ---")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Non-Fraud", "Fraud"],
    digits=4,
    zero_division=0,
))

print("Precision:", round(
    precision_score(y_test, predictions, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, predictions, zero_division=0), 4
))
print("F1-score:", round(
    f1_score(y_test, predictions, zero_division=0), 4
))
print("PR-AUC (Average Precision):", round(
    average_precision_score(y_test, probabilities), 4
))
print("ROC-AUC:", round(
    roc_auc_score(y_test, probabilities), 4
))

print("\nTotal test transactions:", f"{len(y_test):,}")
print("Actual fraud cases:", f"{int(y_test.sum()):,}")
print("Transactions flagged as fraud:", f"{int(predictions.sum()):,}")
print("False alarms:", f"{int(((predictions == 1) & (y_test == 0)).sum()):,}")
print("Fraud cases missed:", f"{int(((predictions == 0) & (y_test == 1)).sum()):,}")
