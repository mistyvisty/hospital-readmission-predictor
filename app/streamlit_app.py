import streamlit as st
import requests

st.set_page_config(page_title="Hospital Readmission Risk Predictor", page_icon="🏥", layout="centered")

st.title("🏥 Hospital Readmission Risk Predictor")
st.markdown("Enter patient details to predict 30-day readmission risk.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox("Age Group", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
        time_in_hospital = st.slider("Days in Hospital", 1, 14, 5)
        num_lab_procedures = st.slider("Lab Procedures", 1, 132, 40)
        num_procedures = st.slider("Procedures", 0, 6, 1)
        num_medications = st.slider("Medications", 1, 81, 12)
        number_outpatient = st.slider("Outpatient Visits", 0, 42, 0)
        number_emergency = st.slider("Emergency Visits", 0, 76, 0)
        number_inpatient = st.slider("Inpatient Visits", 0, 21, 1)

    with col2:
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 8)
        admission_type_id = st.selectbox("Admission Type", [1, 2, 3, 4, 5, 6, 7, 8], index=0)
        discharge_disposition_id = st.selectbox("Discharge Disposition", list(range(1, 30)), index=0)
        admission_source_id = st.selectbox("Admission Source", list(range(1, 26)), index=6)
        change = st.selectbox("Medication Change", ["Ch", "No"])
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        A1Cresult = st.selectbox("A1C Result", ["None", "Normal", ">7", ">8"])
        diabetesMed = st.selectbox("On Diabetes Medication", ["Yes", "No"])

    submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

if submitted:
    payload = {
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "change": change,
        "insulin": insulin,
        "A1Cresult": A1Cresult,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "diabetesMed": diabetesMed
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post("https://hospital-readmission-api-irii.onrender.com/predict", json=payload)
            result = response.json()

            prob = result["probability"]
            label = result["readmitted"]

            st.markdown("---")

            if prob < 0.3:
                st.success(f"🟢 Low Risk — {prob:.1%} probability of readmission")
            elif prob < 0.5:
                st.warning(f"🟡 Medium Risk — {prob:.1%} probability of readmission")
            else:
                st.error(f"🔴 High Risk — {prob:.1%} probability of readmission")

            st.metric("Predicted Readmission", label, f"{prob:.1%}")

        except Exception as e:
            st.error(f"Could not connect to API. Error: {e}")