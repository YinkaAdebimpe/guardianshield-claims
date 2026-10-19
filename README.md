
# Engagement 3 — Claims Severity Prediction

**Client:** GuardianShield Insurance (fictional)  
**Engagement:** Claims Severity Prediction  
**Author:** Yinka Adebimpe  
**Repo:** [guardianshield-claims](https://github.com/YinkaAdebimpe/guardianshield-claims)

---

## Business Context & Impact

**The Business Problem:**
GuardianShield Insurance handles thousands of claims per year. Unpredictable claim costs made pricing and reserving difficult, creating a risk of underwriting losses if high-severity claims were under-reserved.

**The Data Solution:**
I built a regression model predicting insurance claim cost (£). The model benchmarks raw vs. log-transformed targets across four model families (Linear Regression, Decision Tree, Random Forest, Gradient Boosting) to find the most accurate pricing tool.

**The Business Impact:**
The best model achieved a Mean Absolute Error (MAE) of **£1,937**. This improves reserving accuracy, allowing the business to price risk more competitively and forecast high-severity claims with precision. By accurately predicting claim severity, the underwriting team can ensure premiums cover expected payouts, directly protecting profit margins.

---

## Technical Summary

### Headline Findings
- **Claims modelled:** 5,063
- **Median claim:** £1,556
- **Mean claim:** £5,562 (heavily right-skewed)
- **Best model MAE:** **£1,937** (Random Forest, raw target)
- **Best model R²:** **0.765**
- **Dominant predictor:** `policy_type_Life` (importance 0.22).

### Key Insights
1. **Claims severity is product-driven, not customer-driven.** Claim type and incident severity dominate feature importance. Demographics add almost nothing.
2. **Log-transforming the target is critical for Linear Regression.** It rescues the model from catastrophic failure (RMSLE 2.96 → 0.50) but has minimal impact on tree-based models.
3. **77× spread across claim types.** Life critical illness (£25,853 median) vs Motor windscreen (£349 median).

---

## Methodology

### Data
- **Source:** Synthetic claims data (fully reproducible via `src/generate_claims_data.py`).
- **Scale:** 5,063 claims across 5 policy types and 11 claim types.
- **Target:** `claim_amount` (£) — heavily right-skewed (log-normal).

### Approach
1. **EDA** (`01_eda.ipynb`) — distribution analysis, severity by claim type and severity.
2. **Feature Engineering** (`02_feature_engineering.ipynb`) — encoding, log-target creation, derived flags.
3. **Modelling** (`03_modeling.ipynb`) — 4 models × 2 targets = 8 fits.

### Models compared
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting

---

## Repository Structure

```text
.
├── data/
│   ├── raw/                          # Generated CSVs (gitignored)
│   └── processed/                    # Committed summary CSVs
│       ├── features_v1.csv
│       └── predictions_v1.csv
├── images/                           # Analysis and model plots
├── models/                           # Trained model artifacts (.pkl)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── reports/                          # Executive summary, technical report
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

## How to Reproduce

### Prerequisites
- Python 3.12+
- PostgreSQL 14+ (running on `localhost:5432`)
- A database named `guardianshield_db`

### Setup
```bash
git clone https://github.com/YinkaAdebimpe/guardianshield-claims.git
cd guardianshield-claims
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env with Postgres credentials (DB_SCHEMA=claims)
```

### Run the pipeline
```bash
cd src
python generate_claims_data.py
python load_to_postgres.py
cd ..
jupyter lab
```
*Prepared as part of the OkYeahInsight Analytics portfolio.*
