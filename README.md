# SageMaker Insurance Pricing Engine

End-to-End Modern Insurance Pricing Engine built on **AWS SageMaker**.

### Goal
Build a production-ready insurance pricing pipeline starting with Random Forest, upgrading to Neural Networks (TensorFlow), using SageMaker Pipelines, real-time inference endpoints, monitoring, and a Streamlit demo.

### Tech Stack
- **AWS SageMaker** (Pipelines, Training, Endpoints, Model Monitor)
- Python, scikit-learn, TensorFlow
- Streamlit (demo app)
- Insurance pricing dataset (freMTPL2 — French Motor Third-Party Liability)

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
