from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV found in {DATA_DIR}")

csv_path = csv_files[0]
df = pd.read_csv(csv_path,nrows=100) 
index , result = df["isFraud"].value_counts(dropna=False).items()
print(index , result)
for txn_type, group in df.groupby("type"):
        print(txn_type, group['isFraud'].sum())
fraud_mask = df["isFraud"] == 1
print(fraud_mask)
fraud=df[df["isFraud"]==1]
print(fraud.head())

fraud_by_type = fraud.groupby("type").size()
print(fraud_by_type)