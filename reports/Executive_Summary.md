# Claims Severity Prediction — Executive Summary

**Client:** GuardianShield Insurance *(fictional)*  
**Prepared by:** Yinka Adebimpe, Data Scientist | Customer Analytics & Predictive Modelling  
**Date:** October 2026

---

## 1. Overview

GuardianShield Insurance handles thousands of claims each year across Motor, Home, Life, Travel, and Pet products. Leadership wanted to answer: **how much will each claim cost us, and what drives severity?**

The result is a **regression model** predicting claim cost in pounds, trained on 5,063 claims. The best model achieves **MAE £1,937** and **R² 0.765**, providing the actuarial team with a statistical basis for pricing and reserving.

---

## 2. Key Findings

**1. Claims severity is driven by product, not customer**
The five strongest predictors describe the claim itself (policy type, claim type, incident severity). Age, tenure, prior claims, and days-to-report rank in the bottom half.

**2. Claim amount is heavily right-skewed**
Median claim: £1,556. Mean claim: £5,562. Max claim: £114,601. The mean is 3.6× the median, driven by a small number of catastrophic claims.

**3. There is a 77× spread across claim types**
Life critical illness (£25,853 median) vs Pet vet (£330 median).

**4. Target transformation matters for some models**
Tree models barely benefit from log-transforming the target. Linear Regression is **rescued** by it (RMSLE drops from 2.96 to 0.50).

---

## 3. Model Comparison

| Track | Best model | MAE (£) | RMSE (£) | R² |
|-------|-----------|---------|---------|-----|
| Raw £ | **Random Forest** | **£1,937** | £5,184 | 0.765 |
| Log £ | Random Forest | £1,898 | £5,389 | 0.747 |

---

## 4. Recommendations & Business Impact

### Priority 1 — Segment-level reserving, not claim-level
**Action:** Use the model for reserve estimation per claim type and severity bucket.
**Impact:** Individual claim MAE of £1,937 is meaningful. At the segment level, error drops sharply because the model uses the strongest signals directly.

### Priority 2 — Early triage using claim type + severity
**Action:** Route claims using `claim_type` and `incident_severity` at intake.
**Impact:** Life critical illness claims (£25,853 median) should not follow the same workflow as Motor windscreen claims (£349). Early triage saves operational time and improves reserve accuracy.

### Priority 3 — Pricing should use product structure
**Action:** Structure pricing around product (Motor vs Life vs Pet), not customer demographics.
**Impact:** Customer-level features add almost nothing. Pricing decisions should follow the risk structure of the product.

### Financial Impact
With an MAE of £1,937, the reserving team can set aside capital more accurately. Over a book of 5,000 claims, this precision directly protects underwriting margins by ensuring premiums adequately cover expected payouts.

---

## 5. Method Summary

1. **EDA** — established log-normal distribution, 77× spread, zero correlation with customer attributes.
2. **Feature Engineering** — encoded claim type, policy type, severity; created raw and log targets.
3. **Modelling** — benchmarked 4 models × 2 targets = 8 fits.

**Core Lesson:** Match the target transformation to the model's assumptions. Trees don't need it; Linear Regression does.

---

## 6. Limitations

**Synthetic data** — Distribution parameters were tuned to reflect UK benchmarks but require real-world validation.

**Financial assumptions** — The impact of MAE £1,937 on margins is a directional estimate. A full profit-and-loss simulation would be required to quantify the exact margin protection.

**No fraud flag** — The model predicts legitimate severity. Production would need a fraud-detection layer.

---
*Prepared as part of the OkYeahInsight Analytics portfolio.*
