import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load data for encoder fitting
df = pd.read_csv("Impact_of_Remote_Work_on_Mental_Health.csv")

# App Title
st.title("🧑‍💻 Worker Mental State Prediction")
st.markdown("Predict the mental health status of a remote worker using a machine learning model.")

# Model info
algorithm_name = "Random Forest Classifier"
st.markdown(f"**🔍 Model Used:** {algorithm_name}")

# Load model and encoders
model = joblib.load("burnout_model.pkl")
# Binary mapping
inverse_map = {0: "Low Risk of Burnout", 1: "High Risk of Burnout"}

# Load encoders
encoderWL = joblib.load("work_location_encoder.pkl")

# Feature encoding maps
sleep_quality_map = {'Poor': 1, 'Average': 2, 'Good': 3}
stress_level_map = {'Low': 1, 'Medium': 2, 'High': 3}
productivity_change_map = {'Decrease': -1, 'No Change': 0, 'Increase': 1}
satisfaction_map = {'Unsatisfied': -1, 'Neutral': 0, 'Satisfied': 1}
physical_activity_map = {'No': 0, 'Weekly': 1, 'Daily': 2}
Access_map = {'No': 0, 'Yes': 1}

# Input form
with st.form("user_input"):
    st.subheader("📋 Enter Worker Details")

    age = st.slider("Age", 18, 65, 30)
    experience = st.slider("Years of Experience", 0, 40, 5)
    meetings = st.slider("Number of Virtual Meetings per Week", 0, 30, 5)
    social_isolation = st.slider("Social Isolation Rating", 1, 5, 3)
    support = st.slider("Company Support for Remote Work", 1, 5, 3)
    hours = st.slider("Hours Worked Per Week", 0, 100, 40)
    balance = st.slider("Work-Life Balance Rating", 1, 5, 3)
    sleep = st.selectbox("Sleep Quality", list(sleep_quality_map.keys()))
    stress = st.selectbox("Stress Level", list(stress_level_map.keys()))
    productivity = st.selectbox("Productivity", list(productivity_change_map.keys()))
    satisfaction = st.selectbox("Satisfaction with Remote Work", list(satisfaction_map.keys()))
    access = st.selectbox("Access to Mental Health Resources", list(Access_map.keys()))
    physical_activity = st.selectbox("Physical Activity", list(physical_activity_map.keys()))
    work_location = st.selectbox("Work Location", encoderWL.classes_)

    submit = st.form_submit_button("🔎 Predict Mental State")

# Prediction logic
if submit:
    input_data = {
    'Age': age,
    'Years_of_Experience': experience,
    'Hours_Worked_Per_Week': hours,
    'Number_of_Virtual_Meetings': meetings,
    'Work_Life_Balance_Rating': balance,
    'Social_Isolation_Rating': social_isolation,
    'Company_Support_for_Remote_Work': support,
    'Sleep_Quality_Ranked': sleep_quality_map[sleep],
    'Stress_Level_Ranked': stress_level_map[stress],
    'Productivity_Change_Ranked': productivity_change_map[productivity],
    'Satisfaction_with_Remote_Work_Ranked': satisfaction_map[satisfaction],
    'Access_to_Mental_Health_Resources_Ranked': Access_map[access],
    'Physical_Activity_Ranked': physical_activity_map[physical_activity],
    'Work_Location_Encoded': encoderWL.transform([work_location])[0]
    }

    user_df = pd.DataFrame([input_data])
    prediction = model.predict(user_df)[0]

    label = inverse_map[int(prediction)]
    st.session_state.prediction_label = label

# Show prediction result
if "prediction_label" in st.session_state:
    if st.session_state.prediction_label == "High Risk of Burnout":
        st.error(f"⚠️ Prediction: **{st.session_state.prediction_label}**")
        st.markdown("It is recommended to take a break and seek support.")
    else:
        st.success(f"✅ Prediction: **{st.session_state.prediction_label}**")
