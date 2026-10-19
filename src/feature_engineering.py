
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"

CHUNK_SIZE = 250_000

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("Dataset found:", CSV_PATH.name)
print("Feature engineering setup ready.")


for chunk_number , chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE), start=1):
    # 1. Sender balance decrease
    chunk["sender_balance_diff"] = (
        chunk["oldbalanceOrg"] - chunk["newbalanceOrig"]
    )

    # 2. Receiver balance increase
    chunk["receiver_balance_diff"] = (
        chunk["newbalanceDest"] - chunk["oldbalanceDest"]
    )
    # 3. Difference between balance change and transaction amount
    chunk["sender_balance_error"] = (
        chunk["sender_balance_diff"] - chunk["amount"]
    )

    chunk["receiver_balance_error"] = (
        chunk["receiver_balance_diff"] - chunk["amount"]
    )
    # 4. Both sender balances are zero
    chunk["is_zero_sender_balance"] = (
        (chunk["amount"] > 0) &
        (chunk["oldbalanceOrg"] == 0)
        & (chunk["newbalanceOrig"] == 0)
    ).astype("int8")

    # 5. Log-transformed transaction amount
    chunk["amount_log"] = np.log1p(chunk["amount"])

    # Save the processed chunk
    output_file = OUTPUT_PATH / f"features_part_{chunk_number:02d}.csv"

    chunk.to_csv(
        output_file,
        index=False,
        mode="w",
        header=True
    )

    print(f"Processed chunk {chunk_number}: {len(chunk):,} rows")

print("\nFeature engineering completed!")


# Validate the first processed chunk
sample_file = OUTPUT_PATH / "features_part_01.csv"

df = pd.read_csv(sample_file, nrows=1000)

print("\n--- FEATURE VALIDATION ---")
print("Sample shape:", df.shape)

feature_columns = [
    "sender_balance_diff",
    "receiver_balance_diff",
    "sender_balance_error",
    "receiver_balance_error",
    "is_zero_sender_balance",
    "amount_log",
]

print("\nFeature columns:")
print(df[feature_columns].head())

print("\nMissing values in sample:")
print(df[feature_columns].isna().sum())

print("\nFeature statistics:")
print(df[feature_columns].describe())
