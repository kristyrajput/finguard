from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
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
df = pd.concat(
    [pd.read_csv(f, usecols=columns) for f in files],
    ignore_index=True,
)
X = pd.get_dummies(
    df.drop(columns="isFraud"),
    columns=["type"],
    dtype="int8",
)
y = df["isFraud"]
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = joblib.load(MODEL_PATH)
X_test = X_test.reindex(
    columns=model.feature_names_in_,
    fill_value=0,
)
print("Scoring original test set...")

probabilities = model.predict_proba(X_test)[:, 1]
print("PR-AUC:", round(
    average_precision_score(y_test, probabilities), 4
))

print("\n--- RANDOM FOREST THRESHOLD COMPARISON ---")

print(
    f"{'Threshold':>9} {'Precision':>10} {'Recall':>9} "
    f"{'F1':>8} {'TP':>7} {'FP':>8} {'FN':>6}"
)

for threshold in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(y_test, predictions, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(
        f"{threshold:>9.2f} {precision:>10.4f} "
        f"{recall:>9.4f} {f1:>8.4f} "
        f"{tp:>7,} {fp:>8,} {fn:>6,}"
    )