from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_predict():
    response = client.post("/predict", json={
        "time_in_hospital": 5,
        "num_lab_procedures": 45,
        "num_procedures": 2,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 2,
        "number_diagnoses": 7
    })
    assert response.status_code == 200
    data = response.json()
    assert "readmitted" in data
    assert "probability" in data

def test_predict_high_risk():
    response = client.post("/predict", json={
        "time_in_hospital": 14,
        "num_lab_procedures": 80,
        "num_procedures": 6,
        "num_medications": 25,
        "number_outpatient": 0,
        "number_emergency": 3,
        "number_inpatient": 5,
        "number_diagnoses": 9
    })
    assert response.status_code == 200
    assert 0 <= response.json()["probability"] <= 1