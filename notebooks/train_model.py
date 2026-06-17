import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import OrdinalEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──
df = pd.read_csv('data/diabetic_data.csv')
print('Raw shape:', df.shape)

# ── Remove death/hospice discharges — readmission is not clinically meaningful for these ──
death_hospice_codes = [11, 13, 14, 19, 20, 21]
df = df[~df['discharge_disposition_id'].isin(death_hospice_codes)].copy()
print('Shape after removing death/hospice discharges:', df.shape)

# ── Target variable: 1 if readmitted within 30 days, else 0 ──
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
print('Target distribution:')
print(df['readmitted'].value_counts(normalize=True))

# ── Feature engineering ──
age_map = {f'[{i}-{i+10})': i + 5 for i in range(0, 100, 10)}
df['age_numeric'] = df['age'].map(age_map)

df['change_med'] = (df['change'] == 'Ch').astype(int)
df['on_diabetes_med'] = (df['diabetesMed'] == 'Yes').astype(int)
df['insulin_changed'] = df['insulin'].isin(['Up', 'Down']).astype(int)
df['a1c_abnormal'] = df['A1Cresult'].isin(['>7', '>8']).astype(int)

numeric_features = [
    'age_numeric', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'change_med', 'on_diabetes_med',
    'insulin_changed', 'a1c_abnormal'
]

categorical_features = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

X = df[numeric_features + categorical_features].copy()
y = df['readmitted']

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_features] = encoder.fit_transform(X[categorical_features])

print('Samples for training:', len(X))
print('Total features used:', X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                                  random_state=42, eval_metric='logloss'))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
print(f'Mean CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})')

pipeline.fit(X_train, y_train)

y_proba = pipeline.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_proba)
print(f'Test set AUC: {test_auc:.3f}')

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
valid_idx = np.where(recalls[:-1] >= 0.65)[0]
if len(valid_idx) > 0:
    best_idx = valid_idx[np.argmax(precisions[:-1][valid_idx])]
    best_threshold = thresholds[best_idx]
else:
    best_threshold = 0.5
print(f'Selected threshold: {best_threshold:.3f}')

y_pred_tuned = (y_proba >= best_threshold).astype(int)
print('Classification report (tuned threshold):')
print(classification_report(y_test, y_pred_tuned))

xgb_model = pipeline.named_steps['model']
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    'feature': X.columns.tolist(),
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)
print('Top features by SHAP importance:')
print(shap_importance.head(10))

import os
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/model.pkl')
joblib.dump({
    'threshold': best_threshold,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'encoder': encoder,
    'age_map': age_map
}, 'model/metadata.pkl')
print('Model and metadata saved to model/')
