import streamlit as st
import pandas as pd
import joblib

# Load Model and Scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Wine Quality Prediction")
st.write("Predict whether a wine is **Good** or **Bad** using Machine Learning.")

st.sidebar.header("About")
st.sidebar.write("""
This project predicts wine quality using a K-Nearest Neighbors (KNN) model.

**Algorithm:** KNN (Optimized using GridSearchCV)

Prediction:
- Good Quality = 1
- Bad Quality = 0
""")

st.header("Enter Wine Properties")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
    volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
    citric_acid = st.number_input("Citric Acid", value=0.00)
    residual_sugar = st.number_input("Residual Sugar", value=1.9)
    chlorides = st.number_input("Chlorides", value=0.076)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
    density = st.number_input("Density", value=0.9978)
    pH = st.number_input("pH", value=3.51)
    sulphates = st.number_input("Sulphates", value=0.56)
    alcohol = st.number_input("Alcohol", value=9.4)

if st.button("Predict Wine Quality"):

    data = pd.DataFrame([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]], columns=[
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol'
    ])

    scaled = scaler.transform(data)

    prediction = model.predict(scaled)[0]

    probability = model.predict_proba(scaled)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Good Quality Wine")
    else:
        st.error("❌ Bad Quality Wine")

    st.write(f"**Prediction Probability:** {probability.max()*100:.2f}%")