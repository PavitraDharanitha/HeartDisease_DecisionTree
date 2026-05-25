import streamlit as st
import pickle
import numpy as np

# Load model
with open('heart_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below")

# User Inputs
age = st.number_input("Age", 20, 100, 45)

sex = st.selectbox(
    "Sex",
    [0, 1]
)

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    80,
    200,
    120
)

chol = st.number_input(
    "Cholesterol",
    100,
    600,
    200
)

fbs = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.number_input(
    "Oldpeak",
    0.0,
    6.0,
    1.0
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

# Prediction Button
if st.button("Predict"):

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        0,
        thalach,
        exang,
        oldpeak,
        slope,
        0,
        0
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:

        st.error(
            "⚠️ High Chance of Heart Disease"
        )

    else:

        st.success(
            "✅ Low Chance of Heart Disease"
        )
