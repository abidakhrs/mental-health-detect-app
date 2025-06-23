import streamlit as st
import pandas as pd
import numpy as np
import joblib

# App Title
st.title("🧑‍💻 Worker Mental State Prediction")

# 🎯 Model Info
algorithm_name = "Random Forest Classifier"  # Change this to match your actual model
st.markdown(f"**🔍 Model Used:** {algorithm_name}")

# Load trained model
model = joblib.load("burnout_model.pkl")
encoder = joblib.load("state_encoder.pkl")

# Display the class-to-index mapping
inverse_map = {int(k): v for k, v in zip(encoder.transform(encoder.classes_), encoder.classes_)}

# Label maps (same as training)
sleep_quality_map = {'Poor': 1, 'Average': 2, 'Good': 3}
stress_level_map = {'Low': 1, 'Medium': 2, 'High': 3}

# Input form
with st.form("user_input"):
    st.subheader("📋 Enter Worker Details")
    
    hours = st.slider("Hours Worked Per Week", 0, 100, 40)
    balance = st.slider("Work-Life Balance Rating", 1, 5, 3)
    sleep = st.selectbox("Sleep Quality", list(sleep_quality_map.keys()))
    stress = st.selectbox("Stress Level", list(stress_level_map.keys()))
    
    submit = st.form_submit_button("🔎 Predict Mental State")

# Prediction logic
if submit:
    # Convert input to model format
    sleep_rank = sleep_quality_map[sleep]
    stress_rank = stress_level_map[stress]
    
    user_input = pd.DataFrame([{
        'Hours_Worked_Per_Week': hours,
        'Work_Life_Balance_Rating': balance,
        'Sleep_Quality_Ranked': sleep_rank,
        'Stress_Level_Ranked': stress_rank
    }])

    prediction = model.predict(user_input)[0]
    label = inverse_map[int(prediction)]
    st.session_state.prediction_label = label

# Show prediction if available
if "prediction_label" in st.session_state:
    st.success(f"🧠 The worker is predicted to be in: **{st.session_state.prediction_label}** mental state.")
    
