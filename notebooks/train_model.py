import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/diabetic_data.csv')
print('Shape:', df.shape)

# Target variable
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
print('Target distribution:')
print(df['readmitted'].value_counts())

# Select only numeric features
features = ['time_in_hospital','num_lab_procedures','num_procedures',
            'num_medications','number_outpatient','number_emergency',
            'number_inpatient','number_diagnoses']

X = df[features]
y = df['readmitted']

print('Samples for training:', len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

import os
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
print('Model saved to model/model.pkl')
