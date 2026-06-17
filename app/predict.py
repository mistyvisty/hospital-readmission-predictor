import joblib
import pandas as pd

model = None

def load_model():
    global model
    model = joblib.load('model/model.pkl')

def predict(data: dict):
    global model
    if model is None:
        load_model()
    features = ['time_in_hospital','num_lab_procedures','num_procedures',
                'num_medications','number_outpatient','number_emergency',
                'number_inpatient','number_diagnoses']
    df = pd.DataFrame([data])[features]
    prob = model.predict_proba(df)[0][1]
    label = 'YES' if prob >= 0.5 else 'NO'
    return {'readmitted': label, 'probability': round(float(prob), 3)}
