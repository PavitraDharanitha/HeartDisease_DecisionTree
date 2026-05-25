import streamlit as st
import pickle
import pandas as pd

# Load model
with open('heart_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below")

# User Inputs
age = st.number_input(
    "Enter Age",
    20,
    100,
    45
)

chol = st.number_input(
    "Enter Cholesterol Level",
    100,
    600,
    200
)

trestbps = st.number_input(
    "Enter Resting Blood Pressure",
    80,
    200,
    120
)

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame([[
        age,
        chol,
        trestbps
    ]],
    columns=[
        'age',
        'chol',
        'trestbps'
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:

        st.error(
            "⚠️ Heart Disease Detected"
        )

    else:

        st.success(
            "✅ No Heart Disease"
        )
