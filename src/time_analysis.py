import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

CHUNK_SIZE = 250_000

# --------------------------------------------------
# 1. Initialize counters
# --------------------------------------------------

time_stats = {}
type_fraud_stats = {}

# --------------------------------------------------
# 2. Read data in chunks
# --------------------------------------------------

for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE):

    # Aggregate by simulation step
    for step, group in chunk.groupby("step"):

        if step not in time_stats:
            time_stats[step] = {
                "total": 0,
                "fraud": 0
            }

        time_stats[step]["total"] += len(group)
        time_stats[step]["fraud"] += int(
            group["isFraud"].sum()
        )

    # Count fraud-labelled transactions by type
    fraud_rows = chunk[chunk["isFraud"] == 1]

    fraud_by_type = fraud_rows.groupby("type").size()

    for txn_type, count in fraud_by_type.items():
        type_fraud_stats[txn_type] = (
            type_fraud_stats.get(txn_type, 0) + int(count)
        )


# --------------------------------------------------
# 3. Build a time summary
# --------------------------------------------------

time_df = pd.DataFrame.from_dict(
    time_stats,
    orient="index"
)

time_df.index.name = "step"
time_df = time_df.sort_index()

time_df["fraud_rate_pct"] = (
    time_df["fraud"] / time_df["total"] * 100
)

print("\nFRAUD BY SIMULATION STEP")
print("=" * 60)

print(time_df.sort_values(
    "fraud",
    ascending=False
).head(10).to_string())


# --------------------------------------------------
# 4. Fraud counts by transaction type
# --------------------------------------------------

print("\nFRAUD COUNTS BY TRANSACTION TYPE")
print("=" * 60)

for txn_type, count in sorted(
    type_fraud_stats.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{txn_type}: {count:,}")




# --------------------------------------------------
# Fraud by transaction type in selected intervals
# --------------------------------------------------

target_groups = {61, 621, 641, 721, 741}
interval_type_stats = {}

for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE):

    fraud_rows = chunk[chunk["isFraud"] == 1].copy()

    fraud_rows["step_group"] = (
        ((fraud_rows["step"] - 1) // 20) * 20 + 1
    )

    fraud_rows = fraud_rows[
        fraud_rows["step_group"].isin(target_groups)
    ]

    grouped = fraud_rows.groupby(
        ["step_group", "type"]
    ).size()

    for (step_group, txn_type), count in grouped.items():
        key = (int(step_group), txn_type)
        interval_type_stats[key] = (
            interval_type_stats.get(key, 0) + int(count)
        )

print("\nFRAUD COUNTS BY TYPE IN SELECTED INTERVALS")
print("=" * 60)

for (step_group, txn_type), count in sorted(
    interval_type_stats.items()
):
    print(
        f"Steps {step_group}-{min(step_group + 19, 743)} | "
        f"{txn_type}: {count:,}"
    )

# --------------------------------------------------
# 5. Plot fraud counts over simulation steps
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    time_df.index,
    time_df["fraud"],
    linewidth=1
)

plt.title("Fraud-labelled Transactions by Simulation Step")
plt.xlabel("Simulation Step")
plt.ylabel("Fraud-labelled Transaction Count")
plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "fraud_by_simulation_step.png",
    dpi=150
)

plt.show()


# Plot fraud rate over simulation steps
plt.figure(figsize=(12, 6))

plt.plot(
    time_df.index,
    time_df["fraud_rate_pct"],
    linewidth=1
)

plt.title("Fraud Rate by Simulation Step")
plt.xlabel("Simulation Step")
plt.ylabel("Fraud Rate (%)")
plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "fraud_rate_by_simulation_step.png",
    dpi=150
)

plt.show()

# Group simulation steps into intervals of 20
time_df["step_group"] = ((time_df.index - 1) // 20) * 20 + 1

grouped_time = time_df.groupby("step_group").agg(
    total_transactions=("total", "sum"),
    fraud_transactions=("fraud", "sum")
)

grouped_time["fraud_rate_pct"] = (
    grouped_time["fraud_transactions"]
    / grouped_time["total_transactions"]
    * 100
)
print("\nGroups with highest fraud rates:")
print(
    grouped_time.sort_values(
        "fraud_rate_pct",
        ascending=False
    ).head(10).to_string()
)

print("\nGroups with the fewest transactions:")
print(
    grouped_time.sort_values(
        "total_transactions"
    ).head(10).to_string()
)
plt.figure(figsize=(12, 6))

plt.plot(
    grouped_time.index,
    grouped_time["fraud_rate_pct"],
    marker="o"
)

plt.title("Fraud Rate per 20 Simulation Steps")
plt.xlabel("Starting Simulation Step")
plt.ylabel("Fraud Rate (%)")
plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "fraud_rate_grouped_steps.png",
    dpi=150
)

plt.show()