# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# -------------------------------
# TITLE
# -------------------------------
st.title("❤️ Heart Disease Prediction System")
st.markdown("### Real-Time Machine Learning Project using Decision Tree")

st.write(
    "This application predicts whether a patient has heart disease "
    "based on medical attributes."
)

# -------------------------------
# LOAD DATASET
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("heart.csv")
    return data

df = load_data()

# -------------------------------
# SHOW DATA
# -------------------------------
with st.expander("📂 View Dataset"):
    st.dataframe(df.head())

# -------------------------------
# FEATURES & TARGET
# -------------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# MODEL TRAINING
# -------------------------------
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# MODEL ACCURACY
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# -------------------------------
# USER INPUT SECTION
# -------------------------------
st.subheader("🩺 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 45)
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 200)
    thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

with col2:
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    slope = st.selectbox("Slope", [0, 1, 2])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Heart Disease"):

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

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Chance of Heart Disease")
        st.write("Please consult a cardiologist.")
    else:
        st.success("✅ Low Chance of Heart Disease")
        st.write("Heart condition appears normal.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown(
    "Developed using ❤️ Streamlit, Scikit-Learn, and Decision Tree Algorithm"
)