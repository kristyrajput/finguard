
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"

files = sorted(OUTPUT_PATH.glob("features_part_*.csv"))

if not files:
    raise FileNotFoundError(
        "No processed feature files found. Check data/processed/"
    )

expected_rows = 6_362_620
total_rows = 0

print("Feature files found:", len(files))

for file in files:
    # Count rows without loading the entire file into memory
    row_count = sum(
        len(chunk)
        for chunk in pd.read_csv(file, usecols=["step"], chunksize=100_000)
    )
    total_rows += row_count
    print(f"{file.name}: {row_count:,} rows")

print("\n--- VALIDATION SUMMARY ---")
print(f"Total processed rows: {total_rows:,}")
print(f"Expected original rows: {expected_rows:,}")

if total_rows == expected_rows:
    print("SUCCESS: All dataset rows are accounted for.")
else:
    print("WARNING: Row count does not match the original dataset.")

# Check the six engineered features in the first processed file
sample = pd.read_csv(files[0], nrows=1000)

feature_columns = [
    "sender_balance_diff",
    "receiver_balance_diff",
    "sender_balance_error",
    "receiver_balance_error",
    "is_zero_sender_balance",
    "amount_log",
]

missing_columns = [
    col for col in feature_columns if col not in sample.columns
]

if missing_columns:
    print("WARNING: Missing feature columns:", missing_columns)
else:
    print("\nAll six feature columns exist.")
    print("Missing values in sample:")
    print(sample[feature_columns].isna().sum())
