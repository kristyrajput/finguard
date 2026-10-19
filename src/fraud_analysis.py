import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Dataset path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"

CHUNK_SIZE = 250_000


# --------------------------------------------------
# 2. Initialize analysis counters
# --------------------------------------------------

type_counts = {}
fraud_by_type = {}

fraud_count = 0
fraud_amount = 0.0

# Confusion matrix counts:
# isFlaggedFraud = existing flag
# isFraud = reference label

tp = 0  # Fraud correctly flagged
fn = 0  # Fraud not flagged
fp = 0  # Non-fraud incorrectly flagged
tn = 0  # Non-fraud correctly not flagged


# --------------------------------------------------
# 3. Process the dataset in chunks

# --------------------------------------------------

for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE):
    for txn_type, group in chunk.groupby("type"):
        total=len(group)
        fraud_in_group = int(group["isFraud"].sum())
        type_counts[txn_type] = (
            type_counts.get(txn_type, 0) + total
        )
        fraud_by_type[txn_type] = (
            fraud_by_type.get(txn_type, 0) + fraud_in_group
        )
     # Total number of fraud-labelled transactions
    fraud_mask = chunk["isFraud"] == 1

    fraud_count += int(fraud_mask.sum())

    # Total amount associated with fraud-labelled transactions
    fraud_amount += float(
        chunk.loc[fraud_mask, "amount"].sum()
    )
     # Compare existing flags against the fraud labels
    actual_fraud = chunk["isFraud"] == 1
    flagged_fraud = chunk["isFlaggedFraud"] == 1

    tp += int((actual_fraud & flagged_fraud).sum())
    fn += int((actual_fraud & ~flagged_fraud).sum())
    fp += int((~actual_fraud & flagged_fraud).sum())
    tn += int((~actual_fraud & ~flagged_fraud).sum())
# --------------------------------------------------
# 4. Print fraud analysis by transaction type
# --------------------------------------------------

print("\n" + "=" * 60)
print("FRAUD ANALYSIS BY TRANSACTION TYPE")
print("=" * 60)

for txn_type in sorted(type_counts):
    total = type_counts[txn_type]
    fraud = fraud_by_type.get(txn_type, 0)
    fraud_rate = (fraud / total) * 100

    print(
        f"{txn_type}: total={total:,}, "
        f"fraud={fraud:,}, "
        f"fraud_rate={fraud_rate:.4f}%"
    )


# --------------------------------------------------
# 5. Print overall fraud-associated amount
# --------------------------------------------------

print("\n" + "=" * 60)
print("OVERALL FRAUD SUMMARY")
print("=" * 60)

print(f"Total transactions: {sum(type_counts.values()):,}")
print(f"Fraud-labelled transactions: {fraud_count:,}")

overall_fraud_rate = (
    fraud_count / sum(type_counts.values())
) * 100

print(f"Overall fraud rate: {overall_fraud_rate:.6f}%")
print(f"Fraud-associated amount: {fraud_amount:,.2f}")

if fraud_count > 0:
    average_fraud_amount = fraud_amount / fraud_count
    print(f"Average amount per fraud-labelled transaction: "
          f"{average_fraud_amount:,.2f}")


# --------------------------------------------------
# 6. Evaluate the existing fraud flag
# --------------------------------------------------

print("\n" + "=" * 60)
print("EXISTING FRAUD FLAG EVALUATION")
print("=" * 60)

print(f"True Positives (TP):  {tp:,}")
print(f"False Negatives (FN): {fn:,}")
print(f"False Positives (FP): {fp:,}")
print(f"True Negatives (TN):  {tn:,}")

print("\nConfusion matrix:")
print("Rows = actual labels; Columns = existing flags")
print(f"{'':18} {'Flagged (1)':>15} {'Not flagged (0)':>18}")
print(f"{'Fraud (1)':18} {tp:>15,} {fn:>18,}")
print(f"{'Non-fraud (0)':18} {fp:>15,} {tn:>18,}")

if tp + fn > 0:
    recall = tp / (tp + fn) * 100
    print(f"\nFraud recall: {recall:.4f}%")

if tp + fp > 0:
    precision = tp / (tp + fp) * 100
    print(f"Flag precision: {precision:.4f}%")
else:
    print("Flag precision: undefined (no transactions flagged)")

print("\nAnalysis completed successfully.")
