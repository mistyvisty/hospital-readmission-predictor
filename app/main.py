from contextlib import asynccontextmanager
import time
import pandas as pd
from pathlib import Path
import tempfile, os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from evidently import Report
from evidently.presets import DataDriftPreset

from app.schemas import PatientData, PredictionResult
from app.predict import load_model, predict

# --- Prometheus metrics ---
PREDICTIONS = Counter("predictions_total", "Total predictions made")
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
DRIFT_DETECTED = Counter("drift_detected_total", "Times drift was detected")

# --- In-memory log for drift detection ---
prediction_log = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title='Hospital Readmission Predictor', lifespan=lifespan)

@app.get('/')
def root():
    return {'message': 'Hospital Readmission Predictor API is running'}

@app.post('/predict', response_model=PredictionResult)
def make_prediction(patient: PatientData):
    start = time.time()
    result = predict(patient.model_dump())
    
    # Log for drift detection
    prediction_log.append(patient.model_dump())
    
    PREDICTIONS.inc()
    LATENCY.observe(time.time() - start)
    
    return result

@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get('/health')
def health():
    return {'status': 'ok', 'predictions_logged': len(prediction_log)}

@app.get('/drift-report', response_class=HTMLResponse)
def drift_report():
    if len(prediction_log) < 10:
        return "<h3>Need at least 10 predictions first. Use /predict endpoint.</h3>"
    
    # Load your reference data (training data)
    ref = pd.read_csv('app/reference_data.csv')
    current = pd.DataFrame(prediction_log[-200:])
    
    # Keep only common columns
    common_cols = [c for c in ref.columns if c in current.columns]
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref[common_cols], current_data=current[common_cols])
    
    result = report.as_dict()
    drift_share = result['metrics'][0]['result']['share_of_drifted_columns']
    if drift_share > 0.3:
        DRIFT_DETECTED.inc()
    
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        report.save_html(f.name)
        html = Path(f.name).read_text()
        os.unlink(f.name)
    return html