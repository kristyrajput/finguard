from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "random_forest_model.joblib"

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

model = joblib.load(MODEL_PATH)
# Match the model's training feature columns and order.
X_test = X_test.reindex(
    columns=model.feature_names_in_,
    fill_value=0,
)
print("Original test rows:", len(X_test))
print("Actual fraud cases:", int(y_test.sum()))
print("Test fraud rate:", round(y_test.mean() * 100, 4), "%")

print("\nScoring the original test set...")
probabilities = model.predict_proba(X_test)[:, 1]

# Start with the default threshold of 0.5
predictions = (probabilities >= 0.5).astype(int)

print("\n--- RANDOM FOREST: ORIGINAL TEST SET ---")
print(classification_report(
    y_test, predictions, digits=4, zero_division=0
))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print("PR-AUC:", round(
    average_precision_score(y_test, probabilities), 4
))
print("ROC-AUC:", round(
    roc_auc_score(y_test, probabilities), 4
))