from pydantic import BaseModel

class PatientData(BaseModel):
    age: str  # e.g. '[70-80)'
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    change: str  # 'Ch' or 'No'
    diabetesMed: str  # 'Yes' or 'No'
    insulin: str  # 'No', 'Up', 'Down', 'Steady'
    A1Cresult: str  # '>7', '>8', 'Norm', or None
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int

class PredictionResult(BaseModel):
    readmitted: str
    probability: float