from pydantic import BaseModel

class PatientData(BaseModel):
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int

class PredictionResult(BaseModel):
    readmitted: str
    probability: float