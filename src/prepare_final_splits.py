from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed"

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

# First reserve 20% for final testing
development, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["isFraud"],
)

# Split the remaining 80% into 60% train and 20% validation
train, validation = train_test_split(
    development,
    test_size=0.25,
    random_state=42,
    stratify=development["isFraud"],
)

print("--- FINAL DATA SPLITS ---")

for name, part in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test),
]:
    print(f"\n{name} rows: {len(part):,}")
    print(f"{name} fraud cases: {int(part['isFraud'].sum()):,}")
    print(f"{name} fraud rate: {part['isFraud'].mean() * 100:.4f}%")

print("\nSplit preparation complete.")
