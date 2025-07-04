import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model & scaler
model = joblib.load("model/heart_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("❤️ Heart Disease Prediction ")
st.markdown("""
Upload a CSV file with patient data to predict the risk of heart disease.

**Expected columns:**
- age, sex, chest pain type, resting bp s, cholesterol, fasting blood sugar, 
resting ecg, max heart rate, exercise angina, oldpeak, ST slope
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Automatically drop target column if it accidentally exists
    if 'target' in data.columns:
        data = data.drop('target', axis=1)

    st.write("📊 Uploaded Data Preview:", data.head())

    # Scale and predict
    scaled = scaler.transform(data)
    preds = model.predict(scaled)

    results = pd.DataFrame({
        "Patient": np.arange(1, len(preds)+1),
        "Prediction": ["Heart Disease" if p==1 else "Normal" for p in preds]
    })

    st.write("✅ Predictions:", results)

    # Download
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download predictions as CSV",
        csv,
        "heart_disease_predictions.csv",
        "text/csv",
        key='download-csv'
    )
