# SageMaker Insurance Pricing Engine

End-to-End Modern Insurance Pricing Engine built on **AWS SageMaker**.

### Goal
Build an End-to-End Production Insurance Pricing Engine on AWS SageMaker that bridges traditional actuarial methods and modern AI/ML techniques.

### Tech Stack
- **AWS SageMaker** (Pipelines, Training, Endpoints, Model Monitor)
- Python, scikit-learn, TensorFlow
- Streamlit (demo app)
- Insurance pricing dataset (freMTPL2 — French Motor Third-Party Liability)

### Models Covered
- Generalized Linear Models (GLM) — Poisson/Gamma for frequency & severity (industry standard)
- Random Forest — Baseline tree ensemble
- XGBoost — High-performance gradient boosting (widely used in modern insurance pricing)
- Neural Networks (TensorFlow) — Deep learning upgrade

### Project Structure
- `notebooks/` — EDA and experiments
- `src/` — Reusable code (preprocessing, modeling, pipelines)
- `pipelines/` — SageMaker Pipeline definitions
- `deployment/` — Endpoint and monitoring config
- `streamlit_app/` — Interactive demo
- `data/` — Local raw data (gitignored)

### Progress
- [ ] Repository setup
- [ ] Dataset downloaded
- [ ] EDA
- [ ] Baseline modeling
- [ ] SageMaker Pipeline
- [ ] Deployment + Monitoring
- [ ] Streamlit Demo

Built by a Data Scientist pivoting from traditional GLM insurance pricing to production AI/ML.
