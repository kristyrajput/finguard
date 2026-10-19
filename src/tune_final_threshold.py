from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "random_forest_final.joblib"

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

development, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["isFraud"],
)

train, validation = train_test_split(
    development,
    test_size=0.25,
    random_state=42,
    stratify=development["isFraud"],
)

del df, development, test, train

X_val = validation.drop(columns=["isFraud"])
y_val = validation["isFraud"]

X_val = pd.get_dummies(X_val, columns=["type"], dtype="int8")

model = joblib.load(MODEL_PATH)
X_val = X_val.reindex(
    columns=model.feature_names_in_,
    fill_value=0,
)

print("Calculating validation probabilities...")
probabilities = model.predict_proba(X_val)[:, 1]

print("\n--- VALIDATION THRESHOLD COMPARISON ---")
print("Threshold | Precision | Recall | F1 | TP | FP | FN")

for threshold in [0.50, 0.80, 0.90, 0.95, 0.99]:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions, zero_division=0)

    tp = int(((predictions == 1) & (y_val == 1)).sum())
    fp = int(((predictions == 1) & (y_val == 0)).sum())
    fn = int(((predictions == 0) & (y_val == 1)).sum())

    print(
        f"{threshold:>9.2f} | {precision:>9.4f} | "
        f"{recall:>6.4f} | {f1:>5.4f} | "
        f"{tp:>4} | {fp:>5} | {fn:>4}"
    )

'''
Precision: Of the transactions flagged as fraud, how many actually are fraud?
Recall: Of all actual fraud transactions, how many did the model catch?
False alarms (FP): Normal transactions incorrectly flagged as fraud.
Fraud missed (FN): Actual fraud transactions the model failed to catch.
'''
