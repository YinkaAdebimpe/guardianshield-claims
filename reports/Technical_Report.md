# Claims Severity Prediction — Technical Report

**Client:** GuardianShield Insurance *(fictional)*  
**Prepared by:** Yinka Adebimpe, Data Scientist | Customer Analytics & Predictive Modelling  
**Date:** October 2026

---

## 1. Introduction & Scope

### 1.1 Objective
To predict the £ cost of each claim to support reserving decisions and pricing accuracy.

### 1.2 Scope
5,063 claims across 5 policy types and 11 claim types, spanning 2018–2025. Prediction is at the individual claim level.

---

## 2. Data

### 2.1 Source
Synthetic claims data generated via `src/generate_claims_data.py`. Claim severity is generated via a log-normal distribution with multipliers based on claim type, severity, and customer attributes.

### 2.2 Target Distribution
| Statistic | Value |
|-----------|-------|
| Median | £1,556.25 |
| Mean | £5,562.46 (3.6× median) |
| Max | £114,601.31 |

---

## 3. Methodology

### 3.1 Feature Engineering
- Dropped identifiers and high-cardinality fields.
- Created `log_claim_amount` (target transform) and `is_high_value` (top 10% flag).
- One-hot encoded `claim_type`, `policy_type`, `customer_segment`, `incident_severity`.

### 3.2 Models and Targets
- **Track A:** Predict raw `claim_amount` (£).
- **Track B:** Predict `log(claim_amount)`, then exponentiate back to £.
- **Models:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting.

### 3.3 Evaluation Metrics
- **MAE (£):** Average absolute error in pounds (headline metric).
- **RMSLE:** Enables cross-track comparison (insensitive to scale).

---

## 4. Results

### 4.1 Track A (Raw Target)
| Model | MAE (£) | RMSE (£) | R² |
|-------|---------|---------|-----|
| **Random Forest** | **1,936.64** | **5,184.13** | **0.7654** |
| Decision Tree | 2,088.86 | 5,673.73 | 0.7191 |
| Linear Regression | 2,744.19 | 5,694.46 | 0.7170 |

### 4.2 Track B (Log Target)
| Model | MAE (£) | RMSE (£) | R² |
|-------|---------|---------|-----|
| **Random Forest** | **1,898.48** | 5,389.03 | 0.7465 |
| Linear Regression | 2,461.28 | 7,392.96 | 0.5230 |

### 4.3 The Log Transform Finding
Linear Regression on raw data failed catastrophically (RMSLE 2.96). On log-data, it became competitive (RMSLE 0.50). Tree models barely changed (Random Forest MAE improved 2%).

---

## 5. Business Impact & Financial Translation

The model's business value is calculated through **reserving accuracy** and **pricing confidence**.

**Calculation Methodology:**
1. **MAE as Financial Precision:** An MAE of £1,937 means the model's predictions are, on average, within £1,937 of the actual claim cost.
2. **Segment-Level Accuracy:** Because the model relies heavily on claim type and severity, it is highly accurate at the **segment level** (e.g., predicting average Home Fire claims). 
3. **Margin Protection:** If GuardianShield sets reserves based on this model, they avoid the risk of under-reserving high-severity claims. For a book of 5,000 claims, an error reduction of ~£800 per claim (vs baseline Linear Regression) protects roughly **£4,000,000 in capital allocation**.

*(Note: Capital protection is a directional estimate based on reducing aggregate prediction error across the portfolio).*

---

## 6. Limitations

**1. Synthetic data** — Reproduces realistic claims severity but is not real client data.

**2. Financial assumptions** — The £4M capital protection estimate is based on reducing aggregate MAE. It does not account for reinsurance structures or regulatory capital requirements (Solvency II), which would require actuarial modelling.

**3. No fraud detection** — The model predicts legitimate severity. A production system needs a fraud-classification stage before using predictions for reserving.

**4. No uncertainty quantification** — Point estimates only. A probabilistic model would provide prediction intervals.

---

## 7. Reproducibility

```bash
git clone https://github.com/YinkaAdebimpe/guardianshield-claims.git
cd guardianshield-claims
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with Postgres credentials (DB_SCHEMA=claims)
cd src
python generate_claims_data.py
python load_to_postgres.py
cd ..
jupyter lab
```

---
*Prepared as part of the OkYeahInsight Analytics portfolio.*
