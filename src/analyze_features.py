
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed"

feature_columns = [
    "sender_balance_diff",
    "receiver_balance_diff",
    "sender_balance_error",
    "receiver_balance_error",
    "is_zero_sender_balance",
    "amount_log",
]

files = sorted(DATA_PATH.glob("features_part_*.csv"))

if not files:
    raise FileNotFoundError("No processed feature files found.")

# Store sums and counts for accurate full-dataset means
totals = {}

for file in files:
    for chunk in pd.read_csv(
        file,
        usecols=["isFraud"] + feature_columns,
        chunksize=100_000,
    ):
        grouped = chunk.groupby("isFraud")[feature_columns].agg(
            ["sum", "count"]
        )

        for label, row in grouped.iterrows():
            if label not in totals:
                totals[label] = row.copy()
            else:
                totals[label] += row

# Build a readable comparison table
results = {}

for label, values in totals.items():
    results[label] = {}

    for col in feature_columns:
        count = values[(col, "count")]
        total = values[(col, "sum")]

        results[label][col] = total / count if count else 0

summary = pd.DataFrame(results).T
summary.index = summary.index.map(
    {0: "Non-fraud", 1: "Fraud"}
)

print("\n--- FEATURE MEAN COMPARISON ---")
print(summary.T.round(3).to_string())

# Zero sender balance percentage
zero_balance_counts = {0: 0, 1: 0}
label_counts = {0: 0, 1: 0}

for file in files:
    for chunk in pd.read_csv(
        file,
        usecols=["isFraud", "is_zero_sender_balance"],
        chunksize=100_000,
    ):
        for label in [0, 1]:
            subset = chunk[chunk["isFraud"] == label]
            label_counts[label] += len(subset)
            zero_balance_counts[label] += int(
                subset["is_zero_sender_balance"].sum()
            )

print("\n--- ZERO SENDER BALANCE RATE ---")

for label, name in [(0, "Non-fraud"), (1, "Fraud")]:
    count = label_counts[label]
    rate = (
        zero_balance_counts[label] / count * 100
        if count else 0
    )
    print(f"{name}: {rate:.3f}%")

print("\nFeature comparison completed.")
