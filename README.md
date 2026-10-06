# Engagement 3 — Claims Severity Prediction

**Client:** GuardianShield Insurance (fictional)
**Engagement:** Claims Severity Prediction
**Author:** Yinka Adebimpe
**Repo:** [guardianshield-claims](https://github.com/YinkaAdebimpe/guardianshield-claims)

---

## Executive Summary

GuardianShield Insurance handles thousands of claims per year across Motor, Home, Life, Travel, and Pet products. Leadership wanted to know: **how much will each claim cost us, and what drives severity?**

This engagement delivers a **regression model** that predicts claim amount in pounds, achieving **MAE £1,937** with a **Random Forest on raw £** target. The model was trained on **5,063 claims** spanning 2018–2025.

**Headline findings:**

| Metric | Value |
|--------|-------|
| Claims modelled | 5,063 |
| Median claim | £1,556 |
| Mean claim | £5,562 (3.6× median) |
| Best model MAE | **£1,937** (Random Forest, raw target) |
| Best model R² | **0.765** |
| Top predictor | `policy_type_Life` (importance 0.22) |

**Key insights:**

1. **Claims severity is driven by product, not customer.** Claim type and incident severity dominate feature importance. Age, tenure, prior claims, and days-to-report all rank in the bottom half.
2. **Log-normal target distribution.** Median £1,556 vs mean £5,562 — a heavy right tail demanding careful handling.
3. **Target transform matters for some models, not others.** Tree models barely benefit from log-transforming the target (MAE improves 2%). Linear Regression is rescued by it (RMSLE 2.96 → 0.50).
4. **77× spread across claim types.** Life critical illness (£25,853 median) vs Motor windscreen (£349 median).

**Recommended application:**

- **Reserving** — predict reserves per claim type and severity bucket, not per claim
- **Triage** — flag high-severity claims early using claim type + incident severity
- **Pricing** — product structure dominates; use it directly for risk-based pricing

---

## Method

### Data

- **Source:** Synthetic claims data (fully reproducible via `src/generate_claims_data.py`)
- **Scale:** 5,063 claims across 5 policy types, 11 claim types
- **Window:** January 2018 – December 2025
- **Target:** `claim_amount` (£) — heavily right-skewed

### Approach

1. **EDA** (`notebooks/01_eda.ipynb`) — distribution analysis, severity by claim type and severity
2. **Feature engineering** (`notebooks/02_feature_engineering.ipynb`) — encoding, log-target creation, derived flags
3. **Modelling** (`notebooks/03_modeling.ipynb`) — 4 models × 2 targets = 8 fits

### Models compared

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### Targets compared

- **Raw target** — predict `claim_amount` directly
- **Log target** — predict `log(claim_amount)`, exponentiate back to £

---

## Repository structure

```text
.
├── data/
│   ├── raw/                          # Generated CSVs (gitignored)
│   └── processed/                    # Committed summary CSVs for GitHub review
│       ├── features_v1.csv
│       └── predictions_v1.csv
├── images/                           # Analysis and model plots
├── models/                           # Trained model artifacts (.pkl)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── reports/                          # Executive summary, technical report
├── sql/                              # SQL scripts
├── src/
│   ├── config.py
│   ├── generate_claims_data.py
│   └── load_to_postgres.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## How to reproduce

### Prerequisites

- Python 3.12+
- PostgreSQL 14+ (running on `localhost:5432`)
- A database named `guardianshield_db`

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YinkaAdebimpe/guardianshield-claims.git
cd Engagement-3-Customer-Claim-Severity

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure database credentials
cp .env.example .env
# Edit .env with your Postgres password (DB_SCHEMA=claims)
```

### Run the pipeline

```bash
# Generate synthetic claims data (~5,000 claims)
cd src
python generate_claims_data.py

# Load into PostgreSQL (creates schema `claims`)
python load_to_postgres.py

# Launch Jupyter and run the notebooks in order
cd ..
jupyter lab
```

Then open the notebooks in `notebooks/` and run them in order:

1. `01_eda.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_modeling.ipynb`

---

## Results

### Model comparison

**Track A — Raw target (£)**

| Model | MAE (£) | RMSE (£) | R² | RMSLE |
|-------|---------|---------|-----|-------|
| **Random Forest** | **£1,937** | **£5,184** | **0.765** | 0.489 |
| Decision Tree | £2,089 | £5,674 | 0.719 | 0.519 |
| Gradient Boosting | £2,118 | £5,495 | 0.737 | 0.929 |
| Linear Regression | £2,744 | £5,694 | 0.717 | **2.957** |

**Track B — Log target (log £)**

| Model | MAE (£) | RMSE (£) | R² | RMSLE |
|-------|---------|---------|-----|-------|
| **Random Forest** | **£1,898** | £5,389 | 0.747 | **0.456** |
| Decision Tree | £1,983 | £5,456 | 0.740 | 0.471 |
| Gradient Boosting | £2,036 | £5,751 | 0.711 | 0.469 |
| Linear Regression | £2,461 | £7,393 | 0.523 | 0.498 |

### Key finding — the log transform rescues Linear Regression

Raw Linear Regression produces an **RMSLE of 2.957** — the model's normality assumption fails catastrophically on a log-normal target. On the log target, RMSLE drops to **0.498** — competitive with tree models.

**Tree models barely benefit from the transform** (Random Forest: £1,937 → £1,898, a 2% improvement). Trees partition the feature space without assuming a distribution, so the transform matters less.

**Conclusion:** Target distribution assumptions matter most for models that make them explicitly.

### Feature importance (Random Forest, log target)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `policy_type_Life` | 0.220 |
| 2 | `claim_type_Life critical illness` | 0.174 |
| 3 | `incident_severity_Severe` | 0.167 |
| 4 | `incident_severity_Moderate` | 0.077 |
| 5 | `claim_type_Pet vet` | 0.064 |

Every top feature describes the **claim**, not the customer.

### Visualisations

- `images/target_distribution.png` — raw vs log-transformed claim amounts
- `images/claims_by_type_severity.png` — median claim by type and severity
- `images/claim_amount_by_type_log.png` — log-scale boxplot by claim type
- `images/correlation_matrix.png` — numeric feature correlations
- `images/predicted_vs_actual.png` — scatter plots for best model on both tracks
- `images/feature_importance.png` — top 20 feature importances

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.12 |
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn |
| Database | PostgreSQL 18, SQLAlchemy, psycopg2 |
| Visualisation | matplotlib, seaborn |
| Notebooks | JupyterLab |
| Version control | Git, GitHub |

---

## Limitations

1. **Synthetic data** — reproduces realistic claims severity but is not real client data.
2. **Cross-engagement consistency note** — Unlike Engagements 1 and 2 where premium and renewal behaviour did not vary by policy type, Engagement 3 finds policy type to be the dominant driver of claims severity. This asymmetry is intentional and mirrors real insurance portfolios: customer behaviour drives retention, product structure drives claims.
3. **No hyperparameter tuning** — sensible defaults were used. Production deployment would tune via cross-validation.
4. **Random split, not temporal** — Production validation should use a holdout period.
5. **No fraud flag** — the model predicts legitimate claim severity. A production system would layer a fraud-detection stage.
6. **No uncertainty quantification** — predictions are point estimates. A probabilistic model (quantile regression, or conformal prediction) would provide prediction intervals.

---

## Author

**Yinka Adebimpe**
Data Scientist | Customer Analytics & Predictive Modelling

I build end-to-end data science solutions for customer-facing businesses — from data engineering and feature design to survival analysis, machine learning, and business-ready reporting.

- GitHub: [@YinkaAdebimpe](https://github.com/YinkaAdebimpe)
- Email: adebimpey@gmail.com

---

*Part of the OkYeahInsight Analytics portfolio — a three-engagement consulting series for GuardianShield Insurance.*

**Note on shared environment:** All three GuardianShield engagements share the Python virtual environment at `C:\venvs\guardianshield-clv`. This keeps dependencies consistent across engagements and avoids duplicating ~500 MB of packages per project.