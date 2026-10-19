from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models"

THRESHOLD = 0.95

columns = [
    "type",
    "sender_balance_diff",
    "receiver_balance_diff",
    "sender_balance_error",
    "receiver_balance_error",
    "is_zero_sender_balance",
    "amount_log",
    "isFraud",
]

files = sorted(DATA_PATH.glob("features_part_*.csv"))

if not files:
    raise FileNotFoundError("Processed feature files not found.")

print("Loading processed data...")

df = pd.concat(
    [pd.read_csv(file, usecols=columns) for file in files],
    ignore_index=True,
)

# Recreate the same 60/20/20 split used during final RF training
development, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["isFraud"],
)

_, validation = train_test_split(
    development,
    test_size=0.25,
    random_state=42,
    stratify=development["isFraud"],
)

del df, development, validation

X_test = test.drop(columns=["isFraud"])
y_test = test["isFraud"]
del test

X_test = pd.get_dummies(
    X_test,
    columns=["type"],
    dtype="int8",
)

models = {
    "Logistic Regression": ROOT / "logistic_regression_model.joblib",
    "Random Forest": MODEL_PATH / "random_forest_final.joblib",
}

results = []

for name, path in models.items():
    print(f"\nEvaluating {name}...")

    if not path.exists():
        print(f"Model not found: {path}")
        continue

    model = joblib.load(path)

    # Align test columns with the model's training features
    model_features = list(model.feature_names_in_)
    X_model = X_test.reindex(
        columns=model_features,
        fill_value=0,
    )

    probabilities = model.predict_proba(X_model)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    row = {
        "Model": name,
        "Threshold": THRESHOLD,
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "PR-AUC": average_precision_score(
            y_test, probabilities
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        ),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }

    results.append(row)

if not results:
    raise FileNotFoundError(
        "No models were evaluated. Check the model paths."
    )

comparison = pd.DataFrame(results)

print("\n--- MODEL COMPARISON ON THE SAME TEST SET ---")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

REPORT_PATH = ROOT / "reports" / "model_comparison.csv"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(REPORT_PATH, index=False)

print(f"\nComparison saved to: {REPORT_PATH}")
