from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV found in {DATA_DIR}")

csv_path = csv_files[0]
chunksize = 250_000

total_rows=0
missing_counts=None
fraud_counts={}
flag_counts = {}
type_counts = {}
type_amounts = {}
min_step = None
max_step = None
amount_sum = 0.0
amount_min = float("inf")
amount_max = float("-inf")

print(f"Auditing: {csv_path.name}")
for chunk in pd.read_csv(csv_path, chunksize=chunksize):
    total_rows += len(chunk)
    # Update missing counts
    missing = chunk.isna().sum()
    if missing_counts is None:
        missing_counts = missing
    else:
        missing_counts = missing_counts.add(missing,fill_value=0)
    for label , count in chunk["isFraud"].value_counts(dropna=False).items():
        fraud_counts[int(label)] = fraud_counts.get(int(label),0) + int(count)
    for label, count in chunk["isFlaggedFraud"].value_counts().items():
        flag_counts[int(label)] = flag_counts.get(int(label), 0) + int(count)
    for txn_type, count in chunk["type"].value_counts().items():
        type_counts[txn_type] = type_counts.get(txn_type, 0) + int(count)
        
    amount_by_type = chunk.groupby("type")["amount"].agg(["sum", "count"])

    for txn_type, row in amount_by_type.iterrows():
        if txn_type not in type_amounts:
            type_amounts[txn_type] = {"sum": 0.0, "count": 0}
        type_amounts[txn_type]["sum"] += float(row["sum"])
        type_amounts[txn_type]["count"] += int(row["count"])

    min_step = int(chunk["step"].min()) if min_step is None else min(
        min_step, int(chunk["step"].min())
    )
    max_step = int(chunk["step"].max()) if max_step is None else max(
        max_step, int(chunk["step"].max())
    )

    amount_sum += float(chunk["amount"].sum())
    amount_min = min(amount_min, float(chunk["amount"].min()))
    amount_max = max(amount_max, float(chunk["amount"].max()))

    print(f"Processed {total_rows:,} rows...")

fraud_total = fraud_counts.get(1, 0)
fraud_rate = fraud_total / total_rows * 100 if total_rows else 0

print("\n========== FULL DATASET AUDIT ==========")
print(f"Total rows: {total_rows:,}")
print(f"Total columns: 11")
print(f"Step range: {min_step} to {max_step}")
print(f"Total transaction amount: {amount_sum:,.2f}")
print(f"Minimum transaction amount: {amount_min:,.2f}")
print(f"Maximum transaction amount: {amount_max:,.2f}")

print("\nFraud labels:")
for label, count in sorted(fraud_counts.items()):
    print(f"  {label}: {count:,} ({count / total_rows * 100:.6f}%)")

print(f"\nOverall fraud rate: {fraud_rate:.6f}%")

print("\nExisting flagged-fraud labels:")
for label, count in sorted(flag_counts.items()):
    print(f"  {label}: {count:,}")

print("\nTransaction types:")
for txn_type, count in sorted(type_counts.items()):
    print(f"  {txn_type}: {count:,}")

print("\nMissing values across the full dataset:")
print(missing_counts.astype(int))

print("\nAmount summary by transaction type:")
for txn_type, stats in sorted(type_amounts.items()):
    avg = stats["sum"] / stats["count"]
    print(
        f"  {txn_type}: count={stats['count']:,}, "
        f"total={stats['sum']:,.2f}, average={avg:,.2f}"
    )   

