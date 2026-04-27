# Insurance Pricing Engine on AWS SageMaker

**End-to-End Modern Insurance Pricing Pipeline** built on AWS SageMaker.

### Project Goal
Build a production-ready insurance pricing engine that bridges traditional actuarial methods and modern AI/ML techniques.  

Starting with **GLM**, then Random Forest → **XGBoost** (widely used in insurance) → Neural Networks (TensorFlow). All wrapped in a full SageMaker Pipeline with real-time inference, monitoring, and a Streamlit demo.

### Models Covered
- **Generalized Linear Models (GLM)** — Poisson/Gamma using statsmodels (industry standard for frequency & severity)
- **Random Forest** — Simple baseline ensemble
- **XGBoost** — High-performance gradient boosting (very common in modern insurance pricing)
- **Neural Networks** — Deep learning upgrade with TensorFlow

### Tech Stack
- AWS SageMaker (Pipelines, Training Jobs, Real-time Endpoints, Model Monitor)
- Python, scikit-learn, XGBoost, statsmodels, TensorFlow
- Streamlit (interactive demo)
- Dataset: freMTPL2 French Motor Third-Party Liability (public insurance dataset)

### Project Structure
- `notebooks/` — EDA and experiments
- `src/` — Reusable code (preprocessing, modeling, pipelines)
- `pipelines/` — SageMaker Pipeline definitions
- `deployment/` — Endpoint configuration
- `streamlit_app/` — Interactive demo app

### Progress
- [x] Repository setup & structure
- [x] requirements.txt
- [x] Dataset downloaded
- [x] EDA (01_eda.ipynb)
- [x] Baseline modeling (GLM + RF + XGBoost + NN)
- [x] SageMaker Pipeline
- [x] Real-time endpoint + monitoring
- [x] Streamlit demo

Built by a Data Scientist with 7+ years in insurance pricing (GLMs for bind, retention, and loss models) pivoting to Production / AI Data Science roles.
