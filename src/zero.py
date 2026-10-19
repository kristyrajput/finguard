
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
files = sorted((root / "data" / "processed").glob("features_part_*.csv"))

fraud_rows = 0
zero_balance_fraud = 0

for file in files:
    for chunk in pd.read_csv(
        file,
        usecols=["isFraud", "oldbalanceOrg", "newbalanceOrig",
                 "is_zero_sender_balance"],
        chunksize=100_000,
    ):
        fraud = chunk["isFraud"] == 1
        zero_balance = (
            (chunk["amount"] > 0)
            (chunk["oldbalanceOrg"] == 0)
            & (chunk["newbalanceOrig"] == 0)
        )

        fraud_rows += int(fraud.sum())
        zero_balance_fraud += int((fraud & zero_balance).sum())

print("Fraud rows:", fraud_rows)
print("Fraud with zero sender balance:", zero_balance_fraud)
print("Percentage:", round(zero_balance_fraud / fraud_rows * 100, 3))
