import streamlit as st
import numpy as np
import joblib

# Load model, columns, scaler
model = joblib.load("KNN_heart.pkl")
columns = joblib.load("columns.pkl")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details:")

# Input fields
age = st.number_input("Age")
restingBP = st.number_input("Resting Blood Pressure")
cholesterol = st.number_input("Cholesterol")
fastingBS = st.number_input("Fasting Blood Sugar")
maxHR = st.number_input("Max Heart Rate")
oldpeak = st.number_input("Oldpeak")

sex = st.selectbox("Sex", ["Male", "Female"])
chestPain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA"])
restECG = st.selectbox("Resting ECG", ["Normal", "ST"])
exercise = st.selectbox("Exercise Angina", ["Yes", "No"])
slope = st.selectbox("ST Slope", ["Flat", "Up"])

# Create input array
input_data = np.zeros(len(columns))

# Fill numeric values
input_data[columns.index("Age")] = age
input_data[columns.index("RestingBP")] = restingBP
input_data[columns.index("Cholesterol")] = cholesterol
input_data[columns.index("FastingBS")] = fastingBS
input_data[columns.index("MaxHR")] = maxHR
input_data[columns.index("Oldpeak")] = oldpeak

# Encoding categorical values

# Sex
if "Sex_M" in columns:
    input_data[columns.index("Sex_M")] = 1 if sex == "Male" else 0

# Chest Pain
if "ChestPainType_ATA" in columns and chestPain == "ATA":
    input_data[columns.index("ChestPainType_ATA")] = 1
elif "ChestPainType_NAP" in columns and chestPain == "NAP":
    input_data[columns.index("ChestPainType_NAP")] = 1
elif "ChestPainType_TA" in columns and chestPain == "TA":
    input_data[columns.index("ChestPainType_TA")] = 1

# Resting ECG
if "RestingECG_Normal" in columns and restECG == "Normal":
    input_data[columns.index("RestingECG_Normal")] = 1
elif "RestingECG_ST" in columns and restECG == "ST":
    input_data[columns.index("RestingECG_ST")] = 1

# Exercise Angina
if "ExerciseAngina_Y" in columns:
    input_data[columns.index("ExerciseAngina_Y")] = 1 if exercise == "Yes" else 0

# ST Slope
if "ST_Slope_Flat" in columns and slope == "Flat":
    input_data[columns.index("ST_Slope_Flat")] = 1
elif "ST_Slope_Up" in columns and slope == "Up":
    input_data[columns.index("ST_Slope_Up")] = 1

# Prediction
if st.button("Predict"):
    input_scaled = scaler.transform([input_data])
    result = model.predict(input_scaled)

    if result[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")