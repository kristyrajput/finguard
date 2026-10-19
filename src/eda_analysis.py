import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the raw dataset
CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

CHUNK_SIZE = 250_000
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

print("Dataset found successfully!")
print(f"Dataset path: {CSV_PATH}")

fraud_count =0 
non_fraud_count = 0

fraud_amount_sum = 0.0
non_fraud_amount_sum=0.0

fraud_samples=[]
non_fraud_samples=[]

SAMPLE_SIZE_PER_CHUNK = 500

for chunk in pd.read_csv(CSV_PATH,chunksize=CHUNK_SIZE):
    fraud= chunk[chunk["isFraud"]==1]
    non_fraud = chunk[chunk["isFraud"]==0]

    # full dataset counts
    fraud_count+=len(fraud)
    non_fraud_count+=len(non_fraud)

    # full dataset amount totals
    fraud_amount_sum+=fraud["amount"].sum()
    non_fraud_amount_sum+=non_fraud["amount"].sum()

    #small samples for visualization
    fraud_samples.append(
        fraud["amount"].sample(
            n=min(SAMPLE_SIZE_PER_CHUNK, len(fraud)),
            random_state=42
        )
    )

    non_fraud_samples.append(
        non_fraud["amount"].sample(
            n=min(SAMPLE_SIZE_PER_CHUNK, len(non_fraud)),
            random_state=42
        )
    )

# calculate average using all transactions
avg_fraud_amount = fraud_amount_sum / fraud_count
avg_non_fraud_amount = non_fraud_amount_sum / non_fraud_count

print("\nTRANSACTION AMOUNT ANALYSIS")
print("-" * 40)

print(f"Fraud transactions: {fraud_count:,}")
print(f"Non-fraud transactions: {non_fraud_count:,}")

print(f"Fraud Amount: {fraud_amount_sum:,.2f}")
print(f"Non-fraud Amount: {non_fraud_amount_sum:,.2f}")

print(f"Average fraud amount: {avg_fraud_amount:,.2f}")
print(f"Average non-fraud amount: {avg_non_fraud_amount:,.2f}")

# combine the samples
fraud_plot_data = pd.concat(fraud_samples, ignore_index=True)
non_fraud_plot_data = pd.concat(non_fraud_samples, ignore_index=True)

# Remove zero amounts for logarithmic plotting
fraud_plot_data = fraud_plot_data[fraud_plot_data > 0]
non_fraud_plot_data = non_fraud_plot_data[non_fraud_plot_data > 0]

# Create logarithmically spaced bins
min_amount = min(
    fraud_plot_data.min(),
    non_fraud_plot_data.min()
)

max_amount = max(
    fraud_plot_data.max(),
    non_fraud_plot_data.max()
)

bins = np.logspace(
    np.log10(min_amount),
    np.log10(max_amount),
    50
)

plt.figure(figsize=(12, 6))

sns.histplot(
    non_fraud_plot_data,
    bins=bins,
    color="blue",
    alpha=0.4,
    stat="density",
    label="Non-fraud"
)

sns.histplot(
    fraud_plot_data,
    bins=bins,
    color="red",
    alpha=0.4,
    stat="density",
    label="Fraud"
)

plt.xscale("log")
plt.title("Transaction Amount Distribution: Fraud vs Non-fraud")
plt.xlabel("Transaction Amount (log scale)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "transaction_amount_distribution.png",
    dpi=150
)

plt.show()