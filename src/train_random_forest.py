from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Load a reproducible sample from each processed file.
# Keep all fraud rows; sample non-fraud rows to control memory.
parts = []

for file in sorted(DATA_PATH.glob("features_part_*.csv")):
    chunk = pd.read_csv(file, usecols=columns)

    fraud = chunk[chunk["isFraud"] == 1]
    nonfraud = chunk[chunk["isFraud"] == 0]

    # approximate proportional sampling from each file
    nonfraud_sample = nonfraud.sample(
        n=min(len(nonfraud),20_000),
        random_state=42,
    )
    parts.append(pd.concat([fraud, nonfraud_sample]))

df = pd.concat(parts, ignore_index=True)
print("Sample shape:", df.shape)
print("Fraud cases:", int(df["isFraud"].sum()))
X = pd.get_dummies(
    df.drop(columns="isFraud"),
    columns=["type"],
    dtype="int8",
)
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=16,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print("\n--- CLASSIFICATION REPORT ---")
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

joblib.dump(model, MODEL_PATH)
print("\nModel saved:", MODEL_PATH)