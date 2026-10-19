import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

CHUNK_SIZE = 250_000

balance_stats = {
    0: {
        "count": 0,
        "sender_decrease_sum": 0.0,
        "receiver_increase_sum": 0.0,
        "sender_match_count": 0,
        "receiver_match_count": 0,
    },
    1: {
        "count": 0,
        "sender_decrease_sum": 0.0,
        "receiver_increase_sum": 0.0,
        "sender_match_count": 0,
        "receiver_match_count": 0,
    },
}

# Read the csv in chunks
for chunk in pd.read_csv(CSV_PATH,chunksize=CHUNK_SIZE):
    # calculate the balance chnages
    chunk["sender_decrease"]= chunk["oldbalanceOrg"] - chunk["newbalanceOrig"]
    chunk["receiver_increase"]= chunk["newbalanceDest"] - chunk["oldbalanceDest"]
    # check whether balance changes match the transaction amount
    chunk["sender_matches"]=(
        (chunk["sender_decrease"]-chunk["amount"]).abs()<0.01
        )
    chunk["receiver_matches"]=(
        (chunk["receiver_increase"]-chunk["amount"]).abs()<0.01
        )
    print(chunk.head())
    # aggregate separately for non-fraud and fraud
    for label in [0,1]:

        group = chunk[chunk["isFraud"]==label]
        stats=balance_stats[label]
        stats["count"]+=len(group)
        stats["sender_decrease_sum"]+=group["sender_decrease"].sum()
        stats["receiver_increase_sum"]+=group["receiver_increase"].sum()
        stats["sender_match_count"]+=group["sender_matches"].sum()
        stats["receiver_match_count"]+=group["receiver_matches"].sum()

# display the results
print("\nBALANCE-CHANGE ANALYSIS")
print("=" * 60)

for label , name in[(0 , "Non-fraud") , (1, "fraud")]:
    stats=balance_stats[label]
    count=stats["count"]
    print(f"\n{name}")
    print("-" * 40)
    print(f"Transactions: {count:,}")
    print(
        "Average sender balance decrease: "
        f"{stats['sender_decrease_sum'] / count:,.2f}"
    )
    print(
        "Average receiver balance increase: "
        f"{stats['receiver_increase_sum'] / count:,.2f}"
    )
    print(
            "Sender decrease matches amount: "
            f"{stats['sender_match_count'] }"
        )
    
    print(
            "Receiver increase matches amount: "
            f"{stats['receiver_match_count'] }"
        )
    print(
        "Sender decrease matches amount: "
        f"{stats['sender_match_count'] / count * 100:.2f}%"
    )

    print(
        "Receiver increase matches amount: "
        f"{stats['receiver_match_count'] / count * 100:.2f}%"
    )






