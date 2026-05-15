import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ---------------- LOAD FILES ----------------
with open("KNN_heart.pkl", "rb") as file:
    model = joblib.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = joblib.load(file)

with open("columns.pkl", "rb") as file:
    columns = joblib.load(file)

# ---------------- TITLE ----------------
st.title("❤️ Heart Disease Prediction System")
st.markdown("Predict whether a person has heart disease or not using Machine Learning.")

st.divider()

# ---------------- USER INPUTS ----------------
st.subheader("Enter Patient Details")

age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["Male", "Female"])
resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
oldpeak = st.slider("Oldpeak", 0.0, 10.0, 1.0)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    ["Y", "N"]
)

st_slope = st.selectbox(
    "ST Slope",
    ["Flat", "Up", "Down"]
)

# ---------------- PREPARE INPUT ----------------
input_data = {
    'Age': age,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'MaxHR': max_hr,
    'Oldpeak': oldpeak,
    'Sex_M': 1 if sex == 'Male' else 0,
    'ChestPainType_ATA': 1 if chest_pain == 'ATA' else 0,
    'ChestPainType_NAP': 1 if chest_pain == 'NAP' else 0,
    'ChestPainType_TA': 1 if chest_pain == 'TA' else 0,
    'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
    'RestingECG_ST': 1 if resting_ecg == 'ST' else 0,
    'ExerciseAngina_Y': 1 if exercise_angina == 'Y' else 0,
    'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
    'ST_Slope_Up': 1 if st_slope == 'Up' else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure correct column order
input_df = input_df.reindex(columns=columns, fill_value=0)

# Scale input
scaled_input = scaler.transform(input_df)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    prediction = model.predict(scaled_input)[0]

    st.divider()

    if prediction == 1:
        st.error("⚠️ High Chance of Heart Disease")
    else:
        st.success("✅ Low Chance of Heart Disease")

    # Probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled_input)[0]

        st.subheader("Prediction Probability")
        st.write(f"No Disease: {probability[0] * 100:.2f}%")
        st.write(f"Heart Disease: {probability[1] * 100:.2f}%")

st.divider()

st.caption("Built with Streamlit + Machine Learning")