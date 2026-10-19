
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

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

if not files:
    raise FileNotFoundError("Processed feature files not found.")

print("Loading processed data...")

df = pd.concat(
    [pd.read_csv(file, usecols=columns) for file in files],
    ignore_index=True,
)

# Split into train+validation and final test
development, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["isFraud"],
)

# Split development into 60% train and 20% validation
train, validation = train_test_split(
    development,
    test_size=0.25,
    random_state=42,
    stratify=development["isFraud"],
)

del df, development, test   # memory clean up

# Sample ONLY the training data.
# Keep all fraud cases and sample non-fraud cases.
fraud_train = train[train["isFraud"] == 1]
normal_train = train[train["isFraud"] == 0]

train_sample = train
del train

X_train = train_sample.drop(columns=["isFraud"])
y_train = train_sample["isFraud"]

X_val = validation.drop(columns=["isFraud"])
y_val = validation["isFraud"]

# Convert transaction type into numerical columns
X_train = pd.get_dummies(X_train, columns=["type"], dtype="int8")
X_val = pd.get_dummies(X_val, columns=["type"], dtype="int8")

# Ensure identical feature columns
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

print("Training rows:", f"{len(X_train):,}")
print("Training fraud cases:", f"{int(y_train.sum()):,}")
print("Validation rows:", f"{len(X_val):,}")
print("Validation fraud cases:", f"{int(y_val.sum()):,}")

print("\nTraining final Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=16,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Validation probabilities; do not use the test set here
val_probabilities = model.predict_proba(X_val)[:, 1]

print("\n--- VALIDATION RESULTS ---")
print(
    "PR-AUC:",
    round(average_precision_score(y_val, val_probabilities), 4),
)
print(
    "ROC-AUC:",
    round(roc_auc_score(y_val, val_probabilities), 4),
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
print("Next: tune the decision threshold using validation data.")

