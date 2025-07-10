import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained ML model and scaler
model = joblib.load("diabetes_prediction_model_final.pkl")
scaler = joblib.load("diabetes_scalar_final.pkl")


# Set Streamlit page configuration
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🩺 Diabetes Risk Prediction")
st.markdown("Get a quick idea of your diabetes risk and some helpful advice to stay healthy.")

# ----- USER INPUTS ----- #

st.header("👤 Personal Information")
gender = st.radio("What is your gender?", ["Male", "Female"])
age = st.selectbox("Select your age group:", ["10-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90"])
race = st.selectbox("What is your background?", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])

if gender == "Female":
    pregnant = st.checkbox("Are you currently pregnant?")
else:
    pregnant = False

st.header("🩺 Health Information")
A1Cresult = st.selectbox("Do you know your A1C result?", ["None", "Norm", ">7", ">8"])
max_glu_serum = st.selectbox("Max blood sugar level (if tested)?", ["None", "Norm", ">200", ">300"])
num_medications = st.selectbox("How many medications do you currently take?", ["0-5", "6-10", "11-15", "16-20", ">20"])
time_in_hospital = st.selectbox("Recent hospital stay (last 12 months)?", ["No hospital visit", "1-3 days", "4-7 days", "8+ days"])

# ----- DATA PROCESSING ----- #

# Convert user input into model-compatible format
age_map = {
    "10-20": "[10-20)", "21-30": "[20-30)", "31-40": "[30-40)", "41-50": "[40-50)",
    "51-60": "[50-60)", "61-70": "[60-70)", "71-80": "[70-80)", "81-90": "[80-90)"
}
med_map = {"0-5": 3, "6-10": 8, "11-15": 13, "16-20": 18, ">20": 22}
stay_map = {"No hospital visit": 0, "1-3 days": 2, "4-7 days": 5, "8+ days": 9}

# Create input dataframe
input_data = pd.DataFrame({
    "num_medications": [med_map[num_medications]],
    "time_in_hospital": [stay_map[time_in_hospital]],
    "A1Cresult_>7": [1 if A1Cresult == ">7" else 0],
    "A1Cresult_>8": [1 if A1Cresult == ">8" else 0],
    "A1Cresult_None": [1 if A1Cresult == "None" else 0],
    "max_glu_serum_>200": [1 if max_glu_serum == ">200" else 0],
    "max_glu_serum_>300": [1 if max_glu_serum == ">300" else 0],
    "max_glu_serum_None": [1 if max_glu_serum == "None" else 0],
    f"race_{race}": [1],
    f"gender_{gender}": [1],
    f"age_{age_map[age]}": [1]
})

# Make sure all required model columns exist
model_columns = model.get_booster().feature_names
for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match the training data
input_data = input_data[model_columns]

# Scale the input
input_scaled = scaler.transform(input_data)

# ----- PREDICTION ----- #

if st.button("🔍 Predict My Diabetes Risk"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ You may be at HIGH risk of developing diabetes.")
        st.subheader("🧑‍⚕️ Doctor's Advice:")
        st.markdown("- Eat more vegetables, greens, and fiber-rich food")
        st.markdown("- Avoid sugar drinks, white rice, and junk food")
        st.markdown("- Walk or exercise at least 30 minutes daily")
        st.markdown("- Take a sugar test every 6 months")
    else:
        st.success("✅ You are currently at LOW risk of diabetes. Keep it up!")
        st.subheader("💡 Health Tips:")
        st.markdown("- Maintain a healthy diet")
        st.markdown("- Be active and avoid sitting for long hours")
        st.markdown("- Limit processed foods and sugar")
        st.markdown("- Stay hydrated and sleep well")

st.markdown("---")
st.caption("App developed by Ashok using Machine Learning & Streamlit ❤️")
