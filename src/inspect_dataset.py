from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_dir=project_root / "data" / "raw"

# Find CSV files in the raw data folder
csv_files = list(data_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV file found in: {data_dir}\n"
        "Place the PaySim CSV inside data/raw/."
    )

csv_path = csv_files[0]
print("Dataset:", csv_path.name)
print(f"File size: {csv_path.stat().st_size / (1024**2):.2f} MB")

df = pd.read_csv(csv_path,nrows=100_000)  # Load a sample of the dataset for inspection
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("Column names:")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- MISSING VALUES IN SAMPLE ---")
print(df.isna().sum())

print("\n--- DUPLICATE ROWS IN SAMPLE ---")
print(df.duplicated().sum())

print("\n--- FRAUD LABEL DISTRIBUTION IN SAMPLE ---")
if "isFraud" in df.columns:
    print(df["isFraud"].value_counts(dropna=False))
    print("\nFraud percentage in sample: ")
    print(df["isFraud"].value_counts(normalize=True, dropna=False) * 100)
else :
    print("Column 'isFraud' not found in the dataset sample.")