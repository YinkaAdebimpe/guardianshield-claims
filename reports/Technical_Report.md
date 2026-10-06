# Claims Severity Prediction — Technical Report

**Client:** GuardianShield Insurance *(fictional)*
**Engagement:** Claims Severity Prediction
**Prepared by:** Yinka Adebimpe, Data Scientist | Customer Analytics & Predictive Modelling
**Date:** October 2026

---

## 1. Introduction & Scope

### 1.1 Objective

GuardianShield Insurance required a regression model to predict the **£ cost of each claim** filed against its policies. The model needed to:

1. Predict claim severity in pounds
2. Support reserving decisions per claim category
3. Identify the dominant drivers of claim cost
4. Be fully reproducible from source data to final model artifact

### 1.2 Scope

The engagement covers **5,063 claims** across 5 policy types and 11 claim types, spanning January 2018 – December 2025. Prediction is at the **individual claim level**.

### 1.3 Deliverables

| Deliverable | Location |
|-------------|----------|
| EDA notebook | `notebooks/01_eda.ipynb` |
| Feature engineering notebook | `notebooks/02_feature_engineering.ipynb` |
| Modelling notebook | `notebooks/03_modeling.ipynb` |
| Trained models | `models/best_raw_model.pkl`, `models/best_log_model.pkl` |
| Predictions | PostgreSQL `claims.predictions`, `data/processed/predictions_v1.csv` |
| Executive Summary | `reports/Executive_Summary.md` |
| This report | `reports/Technical_Report.md` |

---

## 2. Data

### 2.1 Source

The engagement uses **fully synthetic claims data** generated from a reproducible Python pipeline (`src/generate_claims_data.py`). Real client data was not available; synthetic data was used to demonstrate methodology on realistic claim severity behaviour.

### 2.2 Scale and dimensions

| Property | Value |
|----------|-------|
| Claims | 5,063 |
| Policy types | 5 (Motor, Home, Life, Travel, Pet) |
| Claim types | 11 |
| Observation window | January 2018 – December 2025 |
| Target | `claim_amount` (£) |

### 2.3 Target distribution

| Statistic | Value |
|-----------|-------|
| Min | £50.00 |
| 25th percentile | £630.14 |
| Median | £1,556.25 |
| Mean | £5,562.46 |
| 75th percentile | £4,993.56 |
| 90th percentile | £16,158.54 |
| 95th percentile | £27,073.82 |
| 99th percentile | £52,755.54 |
| Max | £114,601.31 |

**Mean is 3.6× median** — a classic log-normal signature.

### 2.4 Claim type distribution

| Claim type | Count | Median (£) |
|-----------|-------|-----------|
| Motor collision | 1,316 | 1,516 |
| Home water damage | 719 | 2,034 |
| Life critical illness | 497 | 25,853 |
| Travel medical | 468 | 717 |
| Motor theft | 441 | 4,597 |
| Pet vet | 392 | 330 |
| Home burglary | 362 | 1,750 |
| Travel cancellation | 289 | 452 |
| Motor windscreen | 270 | 349 |
| Home fire | 191 | 8,301 |
| Pet surgery | 118 | 955 |

**77× spread** between the highest and lowest median claim types.

### 2.5 Severity distribution

| Severity | Share |
|----------|-------|
| Minor | 34.0% |
| Moderate | 45.7% |
| Severe | 20.3% |

### 2.6 Data generation methodology

Claim amounts are generated via a **log-normal distribution with multiplicative adjustments**:

```text
amount = lognormal(mean=log(base × severity_multiplier), sigma=0.45)
       × tenure_adjustment
       × prior_claims_adjustment
       × reporting_delay_adjustment
       × police_report_adjustment
       × segment_adjustment
```

Where `base` and `severity_multiplier` are claim-type-specific:

| Claim type | Base (£) | Minor | Moderate | Severe |
|-----------|---------|-------|----------|--------|
| Life critical illness | 25,000 | 0.8 | 1.0 | 1.4 |
| Home fire | 8,000 | 0.5 | 1.0 | 3.0 |
| Motor theft | 4,500 | 0.5 | 1.0 | 2.5 |
| Home water damage | 2,500 | 0.3 | 1.0 | 4.0 |
| Home burglary | 1,800 | 0.5 | 1.0 | 2.5 |
| Motor collision | 1,500 | 0.4 | 1.0 | 3.5 |
| Pet surgery | 1,200 | 0.5 | 1.0 | 2.8 |
| Travel medical | 900 | 0.4 | 1.0 | 5.0 |
| Travel cancellation | 500 | 0.5 | 1.0 | 2.0 |
| Pet vet | 400 | 0.4 | 1.0 | 3.0 |
| Motor windscreen | 350 | 0.8 | 1.0 | 1.5 |

**Severity is correlated with observable features** (Option B design):

- Older customers → higher severe probability
- Longer tenure → lower severe probability
- More prior claims → higher severe probability
- Claim-type-specific skews (Home fire, Life critical illness → more Severe)

This ensures severity is a **genuine behavioural feature** rather than a random label.

### 2.7 Data quality checks

| Check | Result |
|-------|--------|
| All claims have a valid amount | ✅ |
| No missing values in features | ✅ |
| No infinite values in features | ✅ |
| All features numeric after encoding | ✅ |

---

## 3. Methodology

### 3.1 Feature engineering

**Dropped columns:**

| Column | Reason |
|--------|--------|
| `claim_id`, `customer_id` | Identifiers |
| `claim_date`, `policy_start_date` | Time fields |
| `region` | High cardinality |

**Derived features:**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `log_claim_amount` | `log(claim_amount)` | Log-transformed target |
| `is_high_value` | top-10% flag | Secondary binary analysis |
| `tenure_years` | `tenure_months / 12` | Readability |
| `reported_quickly` | `days_to_report <= 7` | Behavioural flag |
| `is_repeat_claimer` | `prior_claims >= 2` | Risk flag |

**Encoded categoricals:**

| Original | Transformation |
|----------|---------------|
| `police_report_filed`, `witness_present` | bool → int |
| `claim_type` (11 categories) | One-hot → 10 dummies |
| `policy_type` (5 categories) | One-hot → 4 dummies |
| `customer_segment` (3 categories) | One-hot → 2 dummies |
| `incident_severity` (3 categories) | One-hot → 2 dummies |

**Final feature matrix:** 5,063 rows × 27 features.

### 3.2 Train/test split

```python
train_test_split(X, y_raw, y_log, test_size=0.20, random_state=42)
```

- Train: 4,050 claims
- Test: 1,013 claims

### 3.3 Models benchmarked

Four models, each fit on both targets (8 fits total):

```python
LinearRegression()

DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=42)

RandomForestRegressor(n_estimators=300, max_depth=12,
                      min_samples_leaf=5, random_state=42, n_jobs=-1)

GradientBoostingRegressor(n_estimators=300, max_depth=5,
                          learning_rate=0.05, random_state=42)
```

### 3.4 Two targets

**Track A — Raw target**

Predict `claim_amount` directly.

**Track B — Log target**

Predict `log(claim_amount)`, then exponentiate to recover £ predictions:

```python
y_log_pred = model.predict(X_test)
y_pred = np.exp(y_log_pred)
```

### 3.5 Evaluation metrics

| Metric | Purpose |
|--------|---------|
| MAE (£) | Average absolute error in pounds — headline metric |
| RMSE (£) | Root mean squared error — penalises large errors |
| R² | Proportion of variance explained |
| RMSLE | Root mean squared log error — insensitive to scale, enables cross-track comparison |

---

## 4. Results

### 4.1 Track A — Raw target

| Model | MAE (£) | RMSE (£) | R² | RMSLE |
|-------|---------|---------|-----|-------|
| **Random Forest** | **1,936.64** | **5,184.13** | **0.7654** | 0.4888 |
| Decision Tree | 2,088.86 | 5,673.73 | 0.7191 | 0.5188 |
| Gradient Boosting | 2,117.92 | 5,494.82 | 0.7365 | 0.9287 |
| Linear Regression | 2,744.19 | 5,694.46 | 0.7170 | **2.9570** |

### 4.2 Track B — Log target

| Model | MAE (£) | RMSE (£) | R² | RMSLE |
|-------|---------|---------|-----|-------|
| **Random Forest** | **1,898.48** | 5,389.03 | 0.7465 | **0.4558** |
| Decision Tree | 1,983.45 | 5,455.55 | 0.7402 | 0.4706 |
| Gradient Boosting | 2,035.71 | 5,751.13 | 0.7113 | 0.4685 |
| Linear Regression | 2,461.28 | 7,392.96 | 0.5230 | 0.4976 |

### 4.3 Key finding — the log transform rescues Linear Regression

**RMSLE comparison across tracks:**

| Model | Raw RMSLE | Log RMSLE | Change |
|-------|-----------|-----------|--------|
| **Linear Regression** | **2.957** | **0.498** | **−83%** |
| Gradient Boosting | 0.929 | 0.469 | −50% |
| Decision Tree | 0.519 | 0.471 | −9% |
| Random Forest | 0.489 | 0.456 | −7% |

**Interpretation:**

- Linear Regression assumes normally distributed errors. On the raw log-normal target, this assumption fails catastrophically — RMSLE inflates to 2.96, meaning predictions are orders of magnitude off for some claims.
- Tree-based models make no distributional assumption about the target. They partition the feature space and are insensitive to target scale.
- **Target transformation matters most for models that make distributional assumptions.**

### 4.4 Feature importance (Random Forest, log target)

Top 20 features:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `policy_type_Life` | 0.2197 |
| 2 | `claim_type_Life critical illness` | 0.1737 |
| 3 | `incident_severity_Severe` | 0.1667 |
| 4 | `incident_severity_Moderate` | 0.0770 |
| 5 | `claim_type_Pet vet` | 0.0637 |
| 6 | `claim_type_Motor theft` | 0.0636 |
| 7 | `claim_type_Home fire` | 0.0537 |
| 8 | `claim_type_Motor windscreen` | 0.0530 |
| 9 | `policy_type_Travel` | 0.0277 |
| 10 | `claim_type_Travel cancellation` | 0.0193 |

**Every top feature describes the claim itself.** Customer attributes (age, tenure, prior claims) rank in the bottom half.

### 4.5 Predictive accuracy in context

Best model: Random Forest, raw target — **MAE £1,937**.

For a claims book with median claim of £1,556:
- 25% of claims are under £630
- 50% under £1,556
- 75% under £4,994

An MAE of £1,937 means prediction errors of that magnitude on average — **usable for segment-level reserving** but not for high-precision individual claim prediction.

---

## 5. Validation & Sensitivity

### 5.1 Cross-track consistency

The two best models (Random Forest on raw, Random Forest on log) produce MAE within 2% of each other. This is a strong consistency signal — the ranking is stable regardless of target transformation.

### 5.2 Correlation findings from EDA

Numeric feature correlations with `claim_amount`:

| Feature | Correlation |
|---------|-------------|
| `customer_age` | 0.023 |
| `tenure_months` | −0.015 |
| `prior_claims` | 0.012 |
| `days_to_report` | 0.007 |

**All below 0.03.** The signal lives entirely in categorical features.

### 5.3 Model selection stability

Random Forest is the best model on both tracks. This is unsurprising given:
- Tree-based models handle skewed targets
- The dominant features are categorical
- Non-linear interactions (claim_type × severity) are captured

### 5.4 Cross-engagement validation

The findings in Engagement 3 contrast with Engagements 1 and 2:

| Engagement | Dominant signal |
|-----------|-----------------|
| 1 — Renewal | Customer behaviour (missed payments, prior renewal) |
| 2 — CLV | Customer behaviour (survival, tenure) |
| **3 — Claims** | **Product (claim type, severity)** |

This asymmetry is intentional and mirrors real insurance:
- **Retention** is a customer problem
- **Reserving and pricing** are product problems

---

## 6. Limitations

**1. Synthetic data** — reproduces realistic claims severity but is not real client data.

**2. No hyperparameter tuning** — sensible defaults were used. Production deployment would tune via cross-validation.

**3. Random split, not temporal** — Train/test split is random. Production validation should use a holdout period.

**4. No fraud detection** — Model predicts legitimate severity. A production system needs a fraud-classification stage before using predictions for reserving.

**5. No uncertainty quantification** — Point estimates only. A probabilistic model (quantile regression, conformal prediction) would provide prediction intervals.

**6. Categorical encoding strategy** — One-hot encoding was used. Alternative strategies (target encoding for claim type) may improve performance for high-cardinality features.

**7. Cross-engagement consistency note** — Premium and renewal behaviour did not vary by policy type in Engagements 1 and 2, whereas policy type is the dominant driver of claims severity here. This asymmetry is intentional and mirrors real insurance portfolios.

---

## 7. Reproducibility

The full pipeline runs end-to-end from a clean clone:

```bash
# Clone
git clone https://github.com/YinkaAdebimpe/guardianshield-claims.git
cd Engagement-3-Customer-Claim-Severity

# Environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Configure database
cp .env.example .env
# Edit .env with Postgres credentials (DB_SCHEMA=claims)

# Generate and load data
cd src
python generate_claims_data.py
python load_to_postgres.py
cd ..

# Run notebooks
jupyter lab
# Execute in order:
#   01_eda.ipynb
#   02_feature_engineering.ipynb
#   03_modeling.ipynb
```

**Environment requirements:**

- Python 3.12+
- PostgreSQL 14+
- Database `guardianshield_db` (schema `claims` created automatically)

**Reproducibility controls:**

- Random seeds fixed (`RANDOM_SEED = 42`)
- Data generation is deterministic
- Train/test split is seeded
- Model hyperparameters are explicit
- Notebooks are self-contained

**Shared environment note:** All three GuardianShield engagements share the Python virtual environment at `C:\venvs\guardianshield-clv`.

---

## 8. Appendix

### 8.1 SQL reference queries

**Claims summary by type:**

```sql
SELECT
    claim_type,
    COUNT(*) AS claims,
    ROUND(AVG(claim_amount)::numeric, 0) AS avg_amount,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY claim_amount)::numeric, 0) AS median_amount,
    ROUND(MAX(claim_amount)::numeric, 0) AS max_amount
FROM claims.claims
GROUP BY claim_type
ORDER BY avg_amount DESC;
```

**Prediction accuracy by claim type:**

```sql
SELECT
    c.claim_type,
    COUNT(*) AS test_claims,
    ROUND(AVG(ABS(p.actual - p.pred_raw))::numeric, 2) AS mae
FROM claims.predictions p
JOIN claims.claims c
    ON p.claim_amount = c.claim_amount
WHERE c.claim_id IS NOT NULL
GROUP BY c.claim_type
ORDER BY mae DESC;
```

*(Note: the predictions table doesn't carry claim_id — the exact join depends on the merge strategy used. Adjust as needed.)*

### 8.2 Notebook outputs

| Output | Notebook | Cell |
|--------|----------|------|
| Target distribution | `01_eda.ipynb` | Cell 4 |
| Claim amount by type/severity | `01_eda.ipynb` | Cell 5 |
| Log-scale boxplots | `01_eda.ipynb` | Cell 6 |
| Correlation matrix | `01_eda.ipynb` | Cell 7 |
| Feature matrix shape | `02_feature_engineering.ipynb` | Cell 7 |
| Raw track results | `03_modeling.ipynb` | Cell 6 |
| Log track results | `03_modeling.ipynb` | Cell 7 |
| Track comparison | `03_modeling.ipynb` | Cell 8 |
| Predicted vs actual | `03_modeling.ipynb` | Cell 9 |
| Feature importance | `03_modeling.ipynb` | Cell 10 |

### 8.3 Visualisations

| File | Content |
|------|---------|
| `images/target_distribution.png` | Raw vs log-transformed claim amounts |
| `images/claims_by_type_severity.png` | Median claim by type and severity |
| `images/claim_amount_by_type_log.png` | Log-scale boxplot by claim type |
| `images/correlation_matrix.png` | Numeric feature correlations |
| `images/predicted_vs_actual.png` | Best model predictions on both tracks |
| `images/feature_importance.png` | Top 20 feature importances |

---

*Prepared as part of the OkYeahInsight Analytics portfolio — a three-engagement consulting series for GuardianShield Insurance.*