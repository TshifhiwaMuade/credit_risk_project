# Credit Risk Assessment System
**MLG382 | CYO Project | Group 2 | CRISP-DM Framework**

---

## Links
- **GitHub Repository:** https://github.com/TshifhiwaMuade/credit_risk_project
- **Web Application:** https://credit-risk-project-d895.onrender.com

---

## Project Overview
This project builds a machine learning powered Credit Risk Assessment System that predicts whether a loan applicant is likely to default on their loan. The system is built for financial institutions and credit providers who need data-driven tools to make faster, more accurate lending decisions.

The system addresses three core business problems that financial institutions face today:
- Loan officers making decisions based on intuition rather than data
- No clear visibility into which financial behaviours actually drive default risk
- No way to group applicants into meaningful risk profiles for targeted intervention

The solution delivers four capabilities:
1. **Risk Classification** — predicts whether an applicant will default (Yes/No) with probability scores
2. **Key Driver Analysis** — identifies which financial features most strongly influence default predictions using SHAP analysis
3. **Borrower Segmentation** — groups applicants into meaningful borrower profiles using K-Means clustering
4. **Decision Support Dashboard** — an interactive DASH web application that integrates all of the above for real-time use by credit analysts

---

## Dataset
| Attribute | Detail |
|-----------|--------|
| File | credit_risk_assessment_cyo_project_dataset.xlsx |
| Records | 32,581 loan applicants |
| Features | 11 (demographics, loan details, credit history) |
| Target | loan_status (0 = No Default, 1 = Default) |
| Class Split | 78.2% No Default vs 21.8% Default |

**Features:**
| Feature | Description | Type |
|---------|-------------|------|
| person_age | Age of the applicant | Numerical |
| person_income | Annual income | Numerical |
| person_home_ownership | Housing status (RENT, OWN, MORTGAGE, OTHER) | Categorical |
| person_emp_length | Years of employment | Numerical |
| loan_intent | Purpose of the loan | Categorical |
| loan_amnt | Loan amount requested | Numerical |
| loan_int_rate | Interest rate on the loan | Numerical |
| loan_status | Target — 0 = No Default, 1 = Default | Binary |
| loan_percent_income | Loan amount as % of income | Numerical |
| cb_person_default_on_file | Previous default on record (Y/N) | Categorical |
| cb_person_cred_hist_length | Length of credit history in years | Numerical |

---

## Model Results

### Risk Classification
Three classifiers were trained and evaluated using the same preprocessed pipeline:

| Model | Tuning Method | Best Parameters | Outcome |
|-------|--------------|-----------------|---------|
| Decision Tree | GridSearch | max_depth=10, criterion=gini | Baseline |
| Random Forest | RandomSearch | n_estimators=200, max_depth=10 | Strong |
| XGBoost | RandomSearch | lr=0.1, max_depth=5, n_estimators=200 | **Best** |

**XGBoost achieved 92.45% accuracy** with strong F1-scores across both classes.

### Patient Segmentation (K-Means)
| Cluster Name | Avg Income | Avg Loan Amount | Avg Credit History |
|-------------|------------|-----------------|-------------------|
| Medium Risk | 55,221 | 8,341 | 3.06 years |
| Low Risk | 60,075 | 8,772 | 8.07 years |

### SHAP Analysis — Top Risk Drivers
Global feature importance from SHAP TreeExplainer on XGBoost:
1. **Loan percent income** — strongest predictor, how much of income the loan represents
2. **Loan interest rate** — higher rates strongly predict default
3. **Person income** — lower income applicants are at higher risk
4. **Home ownership (RENT)** — renters are at higher default risk than owners
5. **Credit history length** — shorter credit history increases default risk

---

## Team Members & Branches
| Member | Role | Branch |
|--------|------|--------|
| Nicholas Sunnasy (601353) | Data Lead | `nicholas-data-lead` |
| Stephen van der Merwe (601789) | Feature Engineer | `stephen-feature-engineer` |
| Rivan Matitz (601530) | Classifier | `rivan-classifier` |
| Nathan Labuschagne (602113) | Clustering Specialist | `nathan-clustering` |
| Tshifhiwa Muade (576941) | Key Driver Analyst | `tshifhiwa-shap-analysis` |
| Nasisipho Mbana (602139) | Deployment Lead | `nasisipho-deployment` |

---

## Project Structure
```
credit_risk_project/
|________data/
|    |______raw/                        # Original unmodified dataset
|    |______processed/                  # Cleaned and split datasets
|
|________src/
|    |______prepare_data.py             # EDA and data cleaning (Nicholas)
|    |______preprocess_data.py          # Encoding, scaling, SMOTE (Stephen)
|    |______train_models.py             # DT, RF, XGBoost classifiers (Rivan)
|    |______cluster_models.py           # K-Means clustering (Nathan)
|    |______shap_analysis.py            # SHAP values and visualisations (Tshifhiwa)
|    |______web_app.py                  # DASH web application (Nasisipho)
|
|________artifacts/                     # Saved models (.pkl) and outputs (.csv)
|________notebooks/                     # Exploratory notebooks
|________reports/
|    |______figures/                    # All saved plots and visualisations
|
|________assets/                        # Dashboard CSS and SHAP images
|________Procfile                       # Render deployment configuration
|________requirements.txt              # Project dependencies
|________ReadMe.md
```

---

## Workflow & Handoff Dependencies
```
Nicholas (EDA & Cleaning)
        ↓
Stephen (Encoding, Scaling, SMOTE)
        ↓               ↓
Rivan (Classification)  Nathan (Clustering)
        ↓               ↓
      Tshifhiwa (SHAP Analysis)
              ↓
        Nasisipho (Dashboard & Deployment)
```

> Members 3 and 4 cannot begin until Members 1 and 2 hand off a clean, model-ready dataset.
> Member 5 cannot begin until both the best classifier and cluster labels are finalised.
> Nasisipho needs progressive access to all outputs — do not wait until the end.

---

## Getting Started

### Step 1 — Clone the Repository
```bash
git clone https://github.com/TshifhiwaMuade/credit_risk_project.git
cd credit_risk_project
```

### Step 2 — Set Up Your Anaconda Environment
1. Open **Anaconda Navigator**
2. Click **Environments** in the left sidebar
3. Click **Create** at the bottom
4. Name it `credit_risk_project`, select **Python 3.11**, click **Create**
5. Click the **play button** next to the environment
6. Select **Open Terminal**
7. Navigate to the project folder:
```bash
cd "path/to/credit_risk_project"
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Copy the Dataset
Place the raw dataset in the correct location:
```
data/raw/credit_risk_assessment_cyo_project_dataset.xlsx
```

### Step 5 — Create Your Branch
Each member must work on their own branch:
```bash
git checkout -b your-branch-name
```

### Step 6 — Launch JupyterLab
```bash
jupyter lab
```

---

## Pipeline Execution Order
Run scripts in this exact order to regenerate all artifacts from scratch:

| Step | Command | Owner | Output |
|------|---------|-------|--------|
| 1 | `python src/prepare_data.py` | Nicholas | `data/processed/cleaned_dataset.csv` |
| 2 | `python src/preprocess_data.py` | Stephen | X/y splits + `artifacts/preprocessor.pkl` |
| 3 | `python src/train_models.py` | Rivan | model .pkl files + predictions |
| 4 | `python src/cluster_models.py` | Nathan | cluster files + `kmeans_model.pkl` |
| 5 | `python src/shap_analysis.py` | Tshifhiwa | SHAP figures + csv files |
| 6 | `python src/web_app.py` | Nasisipho | Dashboard at `http://127.0.0.1:8050` |

> Steps 3 and 4 can be run in parallel once Step 2 is complete.
> Step 3 (train_models.py) takes 30 to 60 minutes due to GridSearch — do not close the terminal.

---

## Pushing Your Work to GitHub
After completing work on your branch:
```bash
git add .
git commit -m "meaningful description of what you did"
git push origin your-branch-name
```
Then open a **Pull Request** on GitHub to merge your branch into `main`.

---

## Important Git Rules
- Never push large CSV files — `data/processed/*.csv` is in `.gitignore`
- Never push `.pkl` model files unless needed for deployment — `artifacts/*.pkl` is in `.gitignore`
- Always pull before starting work: `git pull origin main`
- Always work on your own branch — never commit directly to `main`
- Write meaningful commit messages that describe what you actually did

---

## Artifacts Checklist
Before running the dashboard, verify these files exist:

### artifacts/
```
preprocessor.pkl          ← Stephen
model_dt.pkl              ← Rivan
model_rf.pkl              ← Rivan
model_xgb.pkl             ← Rivan
predictions.csv           ← Rivan
feature_importance.csv    ← Rivan
kmeans_model.pkl          ← Nathan
cluster_scaler.pkl        ← Nathan
cluster_labels.csv        ← Nathan
cluster_profiles.csv      ← Nathan
shap_values.csv           ← Tshifhiwa
shap_cluster_values.csv   ← Tshifhiwa
```

### reports/figures/
```
shap_bar.png              ← Tshifhiwa
shap_beeswarm.png         ← Tshifhiwa
shap_waterfall.png        ← Tshifhiwa
shap_cluster_bar.png      ← Tshifhiwa
cluster_pca.png           ← Nathan
```

---

## Limitations
- Class imbalance (78/22) addressed with SMOTE on training set only
- Silhouette score reflects natural overlap in borrower profiles which is expected in real financial data
- SHAP waterfall explains a fixed test applicant — dynamic per-prediction SHAP is a future enhancement

---

## Deliverables
- Technical Report (CRISP-DM structured, max 2 pages, includes GitHub and web app links)
- GitHub Repository with meaningful commit history from all 6 members
- Interactive DASH Web Application deployed on Render
- Video Demo covering problem understanding, data insights, model results, live app and recommendations
