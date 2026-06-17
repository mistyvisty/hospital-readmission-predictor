from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.schemas import PatientData, PredictionResult
from app.predict import load_model, predict

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
    result = predict(patient.model_dump())
    return result
