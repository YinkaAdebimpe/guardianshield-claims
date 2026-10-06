# Claims Severity Prediction — Executive Summary

**Client:** GuardianShield Insurance *(fictional)*
**Engagement:** Claims Severity Prediction
**Prepared by:** Yinka Adebimpe, Data Scientist | Customer Analytics & Predictive Modelling
**Date:** October 2026

---

## 1. Overview

GuardianShield Insurance handles thousands of claims each year across Motor, Home, Life, Travel, and Pet products. Leadership wanted to answer: **how much will each claim cost us, and what drives severity?**

The result is a **regression model** that predicts claim amount in pounds, trained on **5,063 claims** spanning 2018–2025. The best model — Random Forest on the raw £ target — achieves **MAE £1,937** and **R² 0.765**.

---

## 2. Key Findings

**1. Claims severity is driven by product, not customer**

The five strongest predictors all describe the claim itself:
- `policy_type_Life`
- `claim_type_Life critical illness`
- `incident_severity_Severe`
- `incident_severity_Moderate`
- `claim_type_Pet vet`

Age, tenure, prior claims, and days-to-report all rank in the bottom half of feature importance.

**2. Claim amount is heavily right-skewed**

| Metric | Value |
|--------|-------|
| Median claim | £1,556 |
| Mean claim | £5,562 |
| 90th percentile | £16,159 |
| Max claim | £114,601 |

The mean is 3.6× the median — driven by a small number of catastrophic claims.

**3. There is a 77× spread across claim types**

| Claim type | Median |
|-----------|--------|
| Life critical illness | £25,853 |
| Home fire | £8,301 |
| Motor theft | £4,597 |
| Motor collision | £1,516 |
| Pet vet | £330 |

**4. Target transformation matters for some models, not others**

| Model | Raw target MAE | Log target MAE |
|-------|---------------|----------------|
| Random Forest | £1,937 | £1,898 (2% better) |
| **Linear Regression** | **£2,744 (RMSLE 2.96)** | **£2,461 (RMSLE 0.50)** |

Tree models barely benefit from log-transforming the target. Linear Regression is **rescued** by it.

---

## 3. Model Comparison

**Best models per track:**

| Track | Best model | MAE (£) | RMSE (£) | R² |
|-------|-----------|---------|---------|-----|
| Raw £ | **Random Forest** | **£1,937** | £5,184 | 0.765 |
| Log £ | Random Forest | £1,898 | £5,389 | 0.747 |

The two tracks perform within 2% of each other for tree models. This confirms that target distribution assumptions matter most for models that make them explicitly.

---

## 4. Recommendations

### Priority 1 — Segment-level reserving, not claim-level

**Action:** Use the model for **reserve estimation per claim type and severity bucket**, not for individual claim prediction.

**Impact:** Individual claim MAE of £1,937 is meaningful but bounded — many claims are under £1,000. At the segment level (e.g., "Motor collision, Severe"), the model's error drops sharply because it uses the strongest signals directly.

### Priority 2 — Early triage using claim type + severity

**Action:** Route claims to the appropriate reserving and handling track using `claim_type` and `incident_severity` at intake.

**Impact:** Life critical illness claims (£25,853 median) should not follow the same workflow as Motor windscreen claims (£349 median). Early triage saves operational time and improves reserve accuracy.

### Priority 3 — Pricing should use product structure

**Action:** Continue to structure pricing around product (Motor vs Life vs Pet), not around customer demographics.

**Impact:** Customer-level features add almost nothing to severity prediction. Pricing decisions should follow the risk structure of the product itself.

### Priority 4 — Layer fraud detection

**Action:** The current model predicts **legitimate** claim severity. Deploy a separate fraud-detection stage before using the model for reserve setting.

**Impact:** The model assumes honest claims. Real-world deployment needs a fraud flag, especially for very large claims.

---

## 5. Method Summary

Three notebook stages:

1. **EDA** — established log-normal target distribution, 77× spread across claim types, and near-zero correlation with customer attributes
2. **Feature engineering** — one-hot encoded claim type, policy type, severity, and customer segment; created both raw and log targets
3. **Modelling** — benchmarked 4 models × 2 targets = 8 fits, using MAE (£) as the headline metric

### The core analytical finding

The log transform is not universally beneficial. It is:

- **Neutral** for tree models (Random Forest, Gradient Boosting) — they partition the feature space and don't assume a target distribution
- **Critical** for Linear Regression — the model assumes normality; without the transform, its RMSLE inflates to 2.96

**Lesson:** Match the target transformation to the model's distributional assumptions.

---

## 6. Scope & Limitations

**Synthetic data** — This engagement uses fully reproducible synthetic data. Claim-type distributions were tuned to reflect UK insurance benchmarks.

**Cross-engagement contrast** — Unlike Engagements 1 and 2 where policy type did not predict the outcome, policy type is the dominant driver of claims severity. This asymmetry is intentional and realistic: customer behaviour drives retention, product structure drives claims.

**No hyperparameter tuning** — Sensible defaults were used. Production deployment would tune via cross-validation.

**Random split** — Train/test split is random, not temporal. Production validation should use a holdout period.

**No fraud flag** — Model predicts legitimate severity; a fraud layer is a natural extension.

---

## 7. Appendix

**Data**
- 5,063 claims
- 5 policy types (Motor, Home, Life, Travel, Pet)
- 11 claim types
- Window: January 2018 – December 2025
- Target: `claim_amount` (£)

**Model performance (Random Forest, raw target)**
- MAE: **£1,937**
- RMSE: £5,184
- R²: **0.765**
- RMSLE: 0.489

**Repository**
- Full code, notebooks, and documentation: [github.com/YinkaAdebimpe/guardianshield-claims](https://github.com/YinkaAdebimpe/guardianshield-claims)
- Notebooks: EDA, feature engineering, modelling
- Data pipeline: synthetic generator + PostgreSQL loader
- Trained models: `models/best_raw_model.pkl`, `models/best_log_model.pkl`

---

*Prepared as part of the OkYeahInsight Analytics portfolio — a three-engagement consulting series for GuardianShield Insurance.*