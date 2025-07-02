import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Settings ----------
MODEL_FILE  = Path(__file__).parent / "diabetes_model.pkl"
SCALER_FILE = Path(__file__).parent / "scaler.pkl"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
# --------------------------------

@st.cache_resource
def load_artifacts():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def single_prediction_form():
    st.header("Single‑Patient Prediction")
    st.markdown("Enter values for a single patient to predict diabetes.")

    cols = st.columns(2)
    input_vals = []

    numeric_defaults = {
        "Pregnancies": 0, "Glucose": 100, "BloodPressure": 70, "SkinThickness": 20,
        "Insulin": 85, "BMI": 25.0, "DiabetesPedigreeFunction": 0.5, "Age": 30
    }

    for i, feat in enumerate(FEATURES):
        with cols[i % 2]:
            if feat in ["BMI", "DiabetesPedigreeFunction"]:
                val = st.number_input(
                    feat, min_value=0.0, value=numeric_defaults[feat], step=0.1
                )
            else:
                val = st.number_input(
                    feat, min_value=0, value=numeric_defaults[feat], step=1
                )
            input_vals.append(val)

    if st.button(" Predict", use_container_width=True):
        model, scaler = load_artifacts()
        X = scaler.transform([input_vals])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        st.success(
            f"**Prediction:** {' Diabetic (1)' if pred else ' Not Diabetic (0)'}\n\n"
            f"Estimated Probability of Diabetes: **{prob:.2%}**"
        )


def batch_prediction_uploader():
    st.header(" Batch Prediction")
    st.markdown(
        "Upload a **CSV** with the 8 feature columns in this order:\n"
        f"`{' | '.join(FEATURES)}`"
    )

    file = st.file_uploader("Upload CSV File", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("Preview of uploaded data:", df.head())

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return

        model, scaler = load_artifacts()
        X_scaled = scaler.transform(df[FEATURES])
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

        df["Prediction"] = preds
        df["Probability"] = probs.round(4)
        st.success(" Predictions complete")
        st.dataframe(df)

        csv_out = df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download CSV",
            csv_out,
            file_name="diabetes_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main():
    st.title(" Diabetes Prediction Web App")
    st.markdown(
        """
        Predict diabetes using a trained **RandomForest** model.

        - Use **Single‑Patient** for manual entry.
        - Use **Batch Prediction** to upload a CSV and get many predictions.
        """
    )

    tabs = st.tabs([" Single Prediction", "📁 Batch Prediction"])
    with tabs[0]:
        single_prediction_form()
    with tabs[1]:
        batch_prediction_uploader()


if __name__ == "__main__":
    main()
