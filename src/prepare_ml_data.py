from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed"

files = sorted(DATA_PATH.glob("features_part_*.csv"))

if not files:
    raise FileNotFoundError("processed feature files not found")

columns=[
    "type",
    "sender_balance_diff",
    "receiver_balance_diff",
    "sender_balance_error",
    "receiver_balance_error",
    "is_zero_sender_balance",
    "amount_log",
    "isFraud",
]

print("loading ml columns from processed files ....")

frames = [
    pd.read_csv(file,usecols=columns)
    for file in files
]

df = pd.concat(frames , ignore_index=True)
print(" the datframe",df.head())
print("\nDataset shape:", df.shape)
print("\nTarget distribution:")
print(df["isFraud"].value_counts())
print("\nFraud percentage:")
print(round(df["isFraud"].mean() * 100, 4), "%")


# Separate predictors and target
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# Encode transaction type as numeric indicator columns
X = pd.get_dummies(X, columns=["type"], dtype="int8")
# Stratified split preserves the fraud/non-fraud proportion
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\n--- TRAIN/TEST SPLIT ---")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Training fraud cases:", int(y_train.sum()))
print("Testing fraud cases:", int(y_test.sum()))

print("\nTrain fraud rate:", round(y_train.mean() * 100, 4), "%")
print("Test fraud rate:", round(y_test.mean() * 100, 4), "%")

print("\nML dataset preparation complete.")