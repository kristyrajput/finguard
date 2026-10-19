
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -------------------- PATHS & CONFIGURATION --------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest_final.joblib"

# Use the threshold selected during model evaluation.
THRESHOLD = 0.95

REQUIRED_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

FEATURE_COLUMNS = [
    "sender_balance_diff",
    "receiver_balance_diff",
    "sender_balance_error",
    "receiver_balance_error",
    "is_zero_sender_balance",
    "amount_log",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]


# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="FinGuard Pro | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FinGuard Pro")
st.subheader("Financial Transaction & Fraud Intelligence")
st.write(
    "Upload a transaction CSV to identify potentially "
    "fraudulent transactions using a trained Random Forest model."
)

st.info(
    "This is a decision-support tool using a model trained on "
    "synthetic PaySim data. Predictions are risk estimates, "
    "not proof of fraud."
)


# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# -------------------- UPLOAD CSV --------------------

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
    help="Use a CSV containing the six required original columns.",
)

if uploaded_file is None:
    st.caption(
        "Required columns: type, amount, oldbalanceOrg, "
        "newbalanceOrig, oldbalanceDest, newbalanceDest."
    )
    st.stop()


# -------------------- READ & VALIDATE DATA --------------------

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV contains no transactions.")
    st.stop()

missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))

if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

if "isFraud" in df.columns:
    st.warning(
        "This file contains isFraud labels. The app will not "
        "use them to make predictions."
    )

st.write("### Uploaded Data")
st.write(f"Rows: {len(df):,}")
st.dataframe(df.head(10), use_container_width=True)


# -------------------- FEATURE ENGINEERING --------------------

try:
    X = pd.DataFrame(index=df.index)

    X["sender_balance_diff"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"]
    )
    X["receiver_balance_diff"] = (
        df["newbalanceDest"] - df["oldbalanceDest"]
    )

    X["sender_balance_error"] = (
        X["sender_balance_diff"] - df["amount"]
    )
    X["receiver_balance_error"] = (
        X["receiver_balance_diff"] - df["amount"]
    )

    X["is_zero_sender_balance"] = (
        (df["amount"] > 0)
        & (df["oldbalanceOrg"] == 0)
        & (df["newbalanceOrig"] == 0)
    ).astype("int8")

    X["amount_log"] = np.log1p(df["amount"])

    # Match the one-hot encoding used during training.
    type_dummies = pd.get_dummies(
        df["type"],
        prefix="type",
        dtype="int8",
    )

    X = pd.concat([X, type_dummies], axis=1)

    # Preserve the exact feature names and order expected by the model.
    X = X.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        st.error(
            "Some required values are missing or invalid. "
            "Please clean the CSV and upload it again."
        )
        st.stop()

    if (df["amount"] < 0).any():
        st.error("Transaction amounts cannot be negative.")
        st.stop()

except Exception as error:
    st.error(f"Feature preparation failed: {error}")
    st.stop()


# -------------------- PREDICT FRAUD RISK --------------------

if st.button("Analyze Transactions", type="primary"):
    try:
        expected_count = getattr(model, "n_features_in_", None)

        if expected_count != len(FEATURE_COLUMNS):
            st.error(
                "Model feature count does not match the dashboard. "
                "Check the saved model and training configuration."
            )
            st.stop()

        probabilities = model.predict_proba(X)[:, 1]

        results = df.copy()
        results["fraud_probability"] = probabilities
        results["predicted_fraud"] = (
            probabilities >= THRESHOLD
        ).astype(int)

        results["risk_level"] = np.select(
            [
                probabilities >= THRESHOLD,
                probabilities >= 0.50,
            ],
            [
                "High Risk",
                "Review",
            ],
            default="Low Risk",
        )

        # -------------------- SUMMARY --------------------

        high_risk = int(
            (results["predicted_fraud"] == 1).sum()
        )
        review_count = int(
            (results["risk_level"] == "Review").sum()
        )
        low_risk = int(
            (results["risk_level"] == "Low Risk").sum()
        )

        st.divider()
        st.header("Analysis Results")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Transactions", f"{len(results):,}")
        c2.metric("High-Risk Flags", f"{high_risk:,}")
        c3.metric("Review", f"{review_count:,}")
        c4.metric("Low Risk", f"{low_risk:,}")

        st.caption(
            f"High-risk classification threshold: "
            f"{THRESHOLD:.0%} predicted fraud probability. "
            "Review category: 50% to below the threshold."
        )

        # -------------------- RESULTS TABLE --------------------

        st.subheader("Transaction-Level Predictions")

        display_results = results.sort_values(
            "fraud_probability",
            ascending=False,
        )

        st.dataframe(
            display_results,
            use_container_width=True,
            height=420,
        )

        # -------------------- DOWNLOAD RESULTS --------------------

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions as CSV",
            data=csv_data,
            file_name="finguard_predictions.csv",
            mime="text/csv",
        )

        st.success("Analysis completed successfully.")

    except Exception as error:
        st.error(f"Prediction failed: {error}")
