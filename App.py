import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Model dan Encoder
# ==========================

model = joblib.load("personality_model.pkl")

gender_encoder = joblib.load("gender_encoder.pkl")
education_encoder = joblib.load("education_encoder.pkl")
interest_encoder = joblib.load("interest_encoder.pkl")
personality_encoder = joblib.load("personality_encoder.pkl")

# ==========================
# Judul
# ==========================

st.set_page_config(
    page_title="Cloud Personality Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Cloud-Based Personality Prediction")
st.write("Masukkan data di bawah untuk memprediksi tipe personality.")

# ==========================
# Input
# ==========================

age = st.slider("Age",18,80,25)

gender = st.selectbox(
    "Gender",
    gender_encoder.classes_
)

education = st.selectbox(
    "Education",
    [
        "Undergraduate / High School / Uneducated",
        "Graduate or Higher"
    ]
)

introversion = st.slider(
    "Introversion Score",
    0,
    100,
    50
)

sensing = st.slider(
    "Sensing Score",
    0,
    100,
    50
)

thinking = st.slider(
    "Thinking Score",
    0,
    100,
    50
)

judging = st.slider(
    "Judging Score",
    0,
    100,
    50
)

interest = st.selectbox(
    "Interest",
    interest_encoder.classes_
)

# ==========================
# Encode
# ==========================

gender_value = gender_encoder.transform([gender])[0]

education_value = 0

if education == "Graduate or Higher":
    education_value = 1

interest_value = interest_encoder.transform([interest])[0]

# ==========================
# Predict
# ==========================

if st.button("Predict Personality"):

    input_data = pd.DataFrame({

        "Age":[age],
        "Gender":[gender_value],
        "Education":[education_value],
        "Introversion Score":[introversion],
        "Sensing Score":[sensing],
        "Thinking Score":[thinking],
        "Judging Score":[judging],
        "Interest":[interest_value]

    })

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    personality = personality_encoder.inverse_transform(prediction)

    confidence = probability.max()*100

    st.success(
        f"Predicted Personality : {personality[0]}"
    )

    st.info(
        f"Confidence : {confidence:.2f}%"
    )