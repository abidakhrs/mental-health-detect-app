import streamlit as st

st.set_page_config(page_title="Burnout Detection App", layout="centered")
st.title("🧠 Welcome to the Burnout Detection App")

st.markdown("""
This application helps you:
- 📊 Analyze employee mental health data
- 🤖 Predict if a worker is at risk of burnout based on stress, sleep, and work-life balance
- 📝 Submit new data for live prediction

Use the sidebar to navigate between:
1. **Model Evaluation**: View machine learning results.
2. **User Prediction**: Try it yourself!
""")