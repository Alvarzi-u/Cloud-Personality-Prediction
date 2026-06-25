import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Cloud Personality Prediction",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("personality_model.pkl")

gender_encoder = joblib.load("gender_encoder.pkl")
education_encoder = joblib.load("education_encoder.pkl")
interest_encoder = joblib.load("interest_encoder.pkl")
personality_encoder = joblib.load("personality_encoder.pkl")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("☁ Cloud Computing Project")

st.sidebar.markdown("""
### Machine Learning

Random Forest Classifier

Accuracy : **89.94%**

---

### Cloud Platform

✅ Google Colab

✅ GitHub

✅ Streamlit Cloud

---

Dataset :

Predict People Personality Types
""")

# =====================================================
# TITLE
# =====================================================

st.title("🧠 Cloud-Based Personality Prediction")

st.write(
"""
Masukkan data di bawah untuk memprediksi tipe personality berdasarkan
Machine Learning Random Forest.
"""
)

st.divider()

# =====================================================
# LAYOUT
# =====================================================

left,right = st.columns(2)

# =====================================================
# LEFT COLUMN
# =====================================================

with left:

    st.subheader("👤 Personal Information")

    age = st.slider(
        "Age",
        min_value=18,
        max_value=80,
        value=25
    )

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

    interest = st.selectbox(
        "Interest",
        interest_encoder.classes_
    )

# =====================================================
# RIGHT COLUMN
# =====================================================

with right:

    st.subheader("🧩 Personality Assessment")

    introversion = st.slider(
        "Introversion Score",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.01
    )

    sensing = st.slider(
        "Sensing Score",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.01
    )

    thinking = st.slider(
        "Thinking Score",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.01
    )

    judging = st.slider(
        "Judging Score",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.01
    )

st.divider()

# =====================================================
# BUTTON
# =====================================================

predict = st.button(
    "🔍 Predict Personality",
    use_container_width=True
)

# =====================================================
# ENCODING
# =====================================================

gender_value = gender_encoder.transform([gender])[0]

education_value = 0

if education == "Graduate or Higher":
    education_value = 1

interest_value = interest_encoder.transform([interest])[0]

# =====================================================
# PREDICTION
# =====================================================

if predict:

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

    probabilities = model.predict_proba(input_data)[0]

    prediction_name = personality_encoder.inverse_transform(prediction)[0]

    confidence = np.max(probabilities) * 100

    st.divider()

    st.success("✅ Prediction Completed")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            label="Predicted Personality",
            value=prediction_name
        )

    with col2:

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(min(int(confidence), 100))

    st.divider()

    st.subheader("🏆 Top 3 Most Similar Personality Types")

    top3_index = np.argsort(probabilities)[-3:][::-1]

    for rank, idx in enumerate(top3_index, start=1):

        personality = personality_encoder.inverse_transform([idx])[0]

        score = probabilities[idx] * 100

        if rank == 1:
            medal = "🥇"
        elif rank == 2:
            medal = "🥈"
        else:
            medal = "🥉"

        st.write(f"### {medal} {personality}")

        st.progress(min(int(score), 100))

        st.write(f"Similarity : **{score:.2f}%**")

    st.divider()

    st.subheader("📋 Input Summary")

    summary = pd.DataFrame({

        "Feature":[
            "Age",
            "Gender",
            "Education",
            "Interest",
            "Introversion",
            "Sensing",
            "Thinking",
            "Judging"
        ],

        "Value":[
            age,
            gender,
            education,
            interest,
            introversion,
            sensing,
            thinking,
            judging
        ]

    })

    st.dataframe(
        summary,
        use_container_width=True
    )

st.divider()

st.caption(
"""
Developed by **Alvarezi Usamah**

Cloud Computing Project 2026

Random Forest • Google Colab • GitHub • Streamlit
"""
)