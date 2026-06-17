from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def sample_patient(**overrides):
    base = {
        "age": "[70-80)",
        "time_in_hospital": 5,
        "num_lab_procedures": 45,
        "num_procedures": 2,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 2,
        "number_diagnoses": 7,
        "change": "Ch",
        "diabetesMed": "Yes",
        "insulin": "Up",
        "A1Cresult": ">8",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7
    }
    base.update(overrides)
    return base

def test_predict():
    response = client.post("/predict", json=sample_patient())
    assert response.status_code == 200
    data = response.json()
    assert "readmitted" in data
    assert "probability" in data

def test_predict_high_risk():
    response = client.post("/predict", json=sample_patient(
        time_in_hospital=14, num_lab_procedures=80, num_procedures=6,
        num_medications=25, number_emergency=3, number_inpatient=5, number_diagnoses=9
    ))
    assert response.status_code == 200
    assert 0 <= response.json()["probability"] <= 1