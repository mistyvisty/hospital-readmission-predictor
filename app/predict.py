import joblib
import pandas as pd

model = None
metadata = None

def load_model():
    global model, metadata
    model = joblib.load('model/model.pkl')
    metadata = joblib.load('model/metadata.pkl')

def predict(data: dict):
    global model, metadata
    if model is None:
        load_model()

    df = pd.DataFrame([data])

    # Feature engineering — must match train_model.py exactly
    df['age_numeric'] = df['age'].map(metadata['age_map'])
    df['change_med'] = (df['change'] == 'Ch').astype(int)
    df['on_diabetes_med'] = (df['diabetesMed'] == 'Yes').astype(int)
    df['insulin_changed'] = df['insulin'].isin(['Up', 'Down']).astype(int)
    df['a1c_abnormal'] = df['A1Cresult'].isin(['>7', '>8']).astype(int)

    categorical_features = metadata['categorical_features']
    df[categorical_features] = metadata['encoder'].transform(df[categorical_features])

    feature_order = metadata['numeric_features'] + categorical_features
    X = df[feature_order]

    prob = model.predict_proba(X)[0][1]
    threshold = metadata['threshold']
    label = 'YES' if prob >= threshold else 'NO'
    return {'readmitted': label, 'probability': round(float(prob), 3)}
