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
print("Shape:", df.shape)

# Replace ? with nan only in specific columns, then drop those
df['race'] = df['race'].replace('?', 'Unknown')
df['diag_1'] = df['diag_1'].replace('?', '0')
df['diag_2'] = df['diag_2'].replace('?', '0')
df['diag_3'] = df['diag_3'].replace('?', '0')

# Target variable
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
print("Target distribution:\n", df['readmitted'].value_counts())

# Select only numeric features
features = ['time_in_hospital','num_lab_procedures','num_procedures',
            'num_medications','number_outpatient','number_emergency',
            'number_inpatient','number_diagnoses']

X = df[features]
y = df['readmitted']

print("Samples for training:", len(X))

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'model/model.pkl')
print("✅ Model saved to model/model.pkl")