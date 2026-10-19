from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "random_forest_model.joblib"
OUTPUT_PATH = ROOT / "reports"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
importance_df=pd.DataFrame({
    "feature": model.feature_names_in_,
    "importance": model.feature_importances_,
})
importance_df = importance_df.sort_values(
    "importance", ascending=False
)
print("\n--- RANDOM FOREST FEATURE IMPORTANCE ---")
print(importance_df.to_string(
    index=False,
    formatters={"importance": "{:.4f}".format},
))
# Save the results
csv_path = OUTPUT_PATH / "feature_importance.csv"
importance_df.to_csv(csv_path, index=False)

# Plot the importance scores
plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["feature"],
    importance_df["importance"],
)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plot_path = OUTPUT_PATH / "feature_importance.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print("\nSaved table:", csv_path)
print("Saved chart:", plot_path)