import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("diabetes_prediction_model_final.pkl")
scaler = joblib.load("diabetes_scaler_final.pkl")

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺")
st.title("🧠 Diabetes Risk Prediction")
st.write("Answer the questions below to check your risk of needing diabetes medication.")

# --- User Inputs ---
age = st.selectbox("1. What is your age range?", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
gender = st.radio("2. What is your gender?", ['Male', 'Female'])
time_in_hospital = st.slider("3. Days spent in hospital (last visit)", 1, 20, 5)
num_medications = st.slider("4. How many medications are you currently on?", 0, 50, 10)
a1c_result = st.selectbox("5. A1C test result?", ['None', 'Norm', '>7', '>8'])
max_glu_serum = st.selectbox("6. Max Glucose Serum level?", ['None', 'Norm', '>200', '>300'])
number_inpatient = st.slider("7. Number of inpatient visits in past year", 0, 10, 0)
number_outpatient = st.slider("8. Number of outpatient visits", 0, 10, 0)

# --- Feature Engineering ---
input_dict = {
    'age': age,
    'gender': gender,
    'time_in_hospital': time_in_hospital,
    'num_medications': num_medications,
    'A1Cresult': a1c_result,
    'max_glu_serum': max_glu_serum,
    'number_inpatient': number_inpatient,
    'number_outpatient': number_outpatient
}

# Encode categorical values manually to match model
def encode_inputs(data):
    encoded = []

    # Age
    age_map = {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
               '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9}
    encoded.append(age_map[data['age']])

    # Gender
    encoded.append(1 if data['gender'] == 'Male' else 0)

    # Other features
    encoded.append(data['time_in_hospital'])
    encoded.append(data['num_medications'])
    encoded.append(data['number_inpatient'])
    encoded.append(data['number_outpatient'])

    # A1Cresult encoding
    a1c_map = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
    encoded.append(a1c_map[data['A1Cresult']])

    # Glucose Serum encoding
    glu_map = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
    encoded.append(glu_map[data['max_glu_serum']])

    return np.array(encoded).reshape(1, -1)

# Predict
if st.button("🩺 Predict Risk"):
    input_data = encode_inputs(input_dict)
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    proba = model.predict_proba(scaled_data)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ High risk of requiring diabetes medication ({proba:.2f}% probability).")
        st.markdown("### 💡 Advice")
        st.markdown("- Maintain a low-sugar diet (less rice, sweets)")
        st.markdown("- Walk 30 mins daily 🏃")
        st.markdown("- Regular A1C and glucose checkups")
        st.markdown("- Reduce processed and oily food")
    else:
        st.success(f"✅ You are currently at low risk ({100 - proba:.2f}% confidence). Keep up your lifestyle!")
