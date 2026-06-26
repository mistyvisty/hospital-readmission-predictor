# 🏥 Hospital Readmission Risk Predictor

🔴 **Live App:** [View Dashboard](https://hospital-readmission-bz7hqjeye7fgsppfs3pwqm.streamlit.app)

End-to-end ML pipeline predicting 30-day hospital readmission risk for diabetic patients, with a production-style serving layer, monitoring, Streamlit UI, and automated CI/CD.

## 🎯 Problem
Hospitals are penalized financially for high readmission rates. Identifying which patients are at risk *before* discharge allows for targeted follow-up care. This project predicts the probability that a patient will be readmitted within 30 days, using the **UCI Diabetes 130-US Hospitals dataset** (101,766 encounters, 1999–2008).

## ⚙️ Pipeline
Raw CSV (101,766 rows) → Remove death/hospice discharges (data leakage prevention) → Feature engineering (age bucketing, medication flags) → SMOTE (inside ImbPipeline — no leakage across folds) → XGBoost + 5-fold Stratified Cross-Validation → Precision-recall threshold tuning → SHAP explainability → FastAPI serving layer → Streamlit UI → Prometheus metrics + Evidently drift detection → Docker + Prometheus + Grafana → GitHub Actions CI/CD

## 📊 Results & Honest Context
- **Mean CV AUC: 0.581** | **Test AUC: 0.582**
- Tuned threshold: 0.252 (optimized for recall, given the clinical cost of missing a true readmission)

This AUC may look modest — and that's expected, not a flaw. Published peer-reviewed research on this exact dataset reports XGBoost achieving **AUC 0.667**, with logistic regression at 0.642 ([Source](https://pubmed.ncbi.nlm.nih.gov/40385730/)). Hospital readmission is a genuinely hard, weakly-separable prediction problem. This project intentionally reports honest, reproducible numbers rather than chasing an inflated metric.

## 🔍 Top Predictors (via SHAP)
1. Admission type
2. Whether medication was changed during stay
3. Number of medications
4. Abnormal A1C result
5. Number of procedures

## 📡 Monitoring
Once deployed, the API exposes:

| Endpoint | Description |
|---|---|
| `/predict` | Returns readmission probability |
| `/health` | API health + prediction count |
| `/metrics` | Prometheus-compatible metrics |
| `/drift-report` | Evidently HTML drift report vs reference data |

Prometheus scrapes `/metrics` every 15 seconds. Grafana visualizes prediction volume, latency, and drift detection over time.

## 🛠️ Tech Stack
`Python` `XGBoost` `imbalanced-learn (SMOTE)` `SHAP` `FastAPI` `Streamlit` `Evidently` `Prometheus` `Grafana` `Docker` `GitHub Actions`

## 🚀 Running Locally
```bash
pip install -r requirements.txt
python notebooks/train_model.py
uvicorn app.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API testing.

## 🖥️ Streamlit UI
Run the API first:
```bash
uvicorn app.main:app --reload
```
Then in a second terminal:
```bash
streamlit run app/streamlit_app.py
```
Or just use the **[Live App](https://hospital-readmission-bz7hqjeye7fgsppfs3pwqm.streamlit.app)** — no setup needed.

## ✅ CI/CD
Every push to `main` triggers GitHub Actions to install dependencies, generate test data, retrain the model, and run the automated test suite — ensuring the model and API stay in sync.

## 📁 Project Structure
- `app/` — FastAPI application (`main.py`, `predict.py`, `schemas.py`, `reference_data.csv`)
- `app/streamlit_app.py` — Streamlit web UI
- `notebooks/train_model.py` — training pipeline
- `monitoring/` — `prometheus.yml` and `docker-compose.yml`
- `scripts/check_drift.py` — standalone drift check script
- `tests/test_api.py` — API test suite
- `Dockerfile`
- `.github/workflows/ci.yml`

## 👩‍💻 Built By
Preeti Bhardwaj — [github.com/mistyvisty](https://github.com/mistyvisty)
