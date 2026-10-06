# ============================================
# GUARDIANSHIELD INSURANCE
# ENGAGEMENT 3 — CLAIMS SEVERITY DATA GENERATION
# ============================================
# Produces one row per insurance claim.
# Target: claim_amount (£) — heavily right-skewed.
# ============================================

from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from config import RAW_DATA_DIR, RANDOM_SEED, N_CUSTOMERS

# ============================================
# SETUP
# ============================================

np.random.seed(RANDOM_SEED)
fake = Faker("en_GB")
Faker.seed(RANDOM_SEED)

SIMULATION_START = date(2018, 1, 1)
SIMULATION_END = date(2025, 12, 31)

# Segment shares (consistent with Engagements 1 & 2)
SEGMENT_WEIGHTS = {"loyal": 0.25, "standard": 0.50, "price_sensitive": 0.25}

# Policy type distribution (consistent)
POLICY_TYPES = ["Motor", "Home", "Life", "Travel", "Pet"]
POLICY_TYPE_WEIGHTS = [0.40, 0.25, 0.10, 0.15, 0.10]

# Claim type → policy type mapping
CLAIMS_BY_POLICY = {
    "Motor":   ["Motor collision", "Motor theft", "Motor windscreen"],
    "Home":    ["Home water damage", "Home burglary", "Home fire"],
    "Life":    ["Life critical illness"],
    "Travel":  ["Travel medical", "Travel cancellation"],
    "Pet":     ["Pet vet", "Pet surgery"],
}

# Base means (£) and severity multipliers per claim type
CLAIM_BASE = {
    "Motor collision":      {"base": 1500, "severity": {"Minor": 0.4, "Moderate": 1.0, "Severe": 3.5}},
    "Motor theft":          {"base": 4500, "severity": {"Minor": 0.5, "Moderate": 1.0, "Severe": 2.5}},
    "Motor windscreen":     {"base": 350,  "severity": {"Minor": 0.8, "Moderate": 1.0, "Severe": 1.5}},
    "Home water damage":    {"base": 2500, "severity": {"Minor": 0.3, "Moderate": 1.0, "Severe": 4.0}},
    "Home burglary":        {"base": 1800, "severity": {"Minor": 0.5, "Moderate": 1.0, "Severe": 2.5}},
    "Home fire":            {"base": 8000, "severity": {"Minor": 0.5, "Moderate": 1.0, "Severe": 3.0}},
    "Life critical illness":{"base": 25000,"severity": {"Minor": 0.8, "Moderate": 1.0, "Severe": 1.4}},
    "Travel medical":       {"base": 900,  "severity": {"Minor": 0.4, "Moderate": 1.0, "Severe": 5.0}},
    "Travel cancellation":  {"base": 500,  "severity": {"Minor": 0.5, "Moderate": 1.0, "Severe": 2.0}},
    "Pet vet":              {"base": 400,  "severity": {"Minor": 0.4, "Moderate": 1.0, "Severe": 3.0}},
    "Pet surgery":          {"base": 1200, "severity": {"Minor": 0.5, "Moderate": 1.0, "Severe": 2.8}},
}

# Claim type base weights (probability of each claim type if policy is X)
def claim_type_weights(policy_type: str) -> tuple[list, list]:
    types = CLAIMS_BY_POLICY[policy_type]
    if policy_type == "Motor":
        w = [0.65, 0.20, 0.15]
    elif policy_type == "Home":
        w = [0.55, 0.30, 0.15]
    elif policy_type == "Travel":
        w = [0.60, 0.40]
    elif policy_type == "Pet":
        w = [0.75, 0.25]
    else:
        w = [1.0]
    return types, w


# ============================================
# HELPERS
# ============================================

def add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, 28)
    return date(year, month, day)


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=int(np.random.randint(0, delta + 1)))


def assign_segment() -> str:
    segments = list(SEGMENT_WEIGHTS.keys())
    probs = list(SEGMENT_WEIGHTS.values())
    return np.random.choice(segments, p=probs)


# ============================================
# CLAIM GENERATION
# ============================================

def generate_customer_metadata(customer_index: int) -> dict:
    """Base customer metadata (aligns with Engagement 1 & 2 books)."""
    segment = assign_segment()
    policy_type = np.random.choice(POLICY_TYPES, p=POLICY_TYPE_WEIGHTS)

    # Segment affects premium and tenure behaviour
    segment_premium_mean = {"loyal": 900, "standard": 650, "price_sensitive": 420}[segment]

    return {
        "customer_id": f"GS-{customer_index:06d}",
        "policy_type": policy_type,
        "customer_segment": segment,
        "region": fake.county(),
        "customer_age": int(np.clip(np.random.normal(45, 14), 18, 80)),
        "base_premium": round(float(np.random.lognormal(np.log(segment_premium_mean), 0.35)), 2),
    }


def assign_severity(customer: dict, claim_type: str, tenure_months: int, prior_claims: int) -> str:
    """
    Option B: severity correlated with customer + policy features.
    - Older customers → slightly higher severity
    - Longer tenure → slightly lower severity (better risk behaviour)
    - More prior claims → higher severity
    - Home fire, Life critical illness → skewed to Severe
    """
    base_probs = np.array([0.40, 0.45, 0.15])  # Minor, Moderate, Severe

    # Adjust probabilities
    probs = base_probs.copy()

    # Age
    if customer["customer_age"] > 55:
        probs[2] += 0.05  # more Severe
        probs[0] -= 0.05

    # Tenure
    if tenure_months > 48:
        probs[0] += 0.05  # more Minor (long-tenure customers = lower risk)
        probs[2] -= 0.05
    elif tenure_months < 12:
        probs[2] += 0.05  # new customers = more severe
        probs[0] -= 0.05

    # Prior claims
    if prior_claims >= 2:
        probs[2] += 0.05
        probs[1] -= 0.05

    # Claim-type-specific adjustment
    severe_skewed = ["Home fire", "Life critical illness", "Motor collision"]
    if claim_type in severe_skewed:
        probs[2] += 0.10
        probs[0] -= 0.10

    # Normalise
    probs = np.clip(probs, 0.05, 0.90)
    probs = probs / probs.sum()

    return np.random.choice(["Minor", "Moderate", "Severe"], p=probs)


def compute_claim_amount(
    claim_type: str,
    severity: str,
    customer: dict,
    tenure_months: int,
    prior_claims: int,
    days_to_report: int,
    police_report: bool,
) -> float:
    """
    Log-normal claim amount with adjustments.
    """
    config = CLAIM_BASE[claim_type]
    base = config["base"]
    sev_mult = config["severity"][severity]

    # Base log-normal draw
    amount = float(np.random.lognormal(mean=np.log(base * sev_mult), sigma=0.45))

    # Tenure adjustment (longer tenure = slightly lower claims — careful customers)
    if tenure_months > 48:
        amount *= 0.92

    # Prior claims adjustment (repeat claimers → higher amounts)
    if prior_claims >= 2:
        amount *= 1.15

    # Late reporting → higher amounts (more damage accumulates)
    if days_to_report > 14:
        amount *= 1.10

    # Police report (theft/fire) → typically larger claims
    if police_report:
        amount *= 1.20

    # Segment adjustment
    segment_mult = {"loyal": 1.00, "standard": 1.00, "price_sensitive": 0.95}[customer["customer_segment"]]
    amount *= segment_mult

    # Floor and ceiling
    amount = max(amount, 50)
    amount = min(amount, 500_000)

    return round(amount, 2)


def generate_claim(claim_index: int, customer_index: int) -> dict | None:
    """Generate one claim for a customer, or None if no claim occurs."""
    customer = generate_customer_metadata(customer_index)

    # Determine whether this customer has a claim at all
    # Claims are relatively rare — about 1 per customer over 8 years
    if np.random.random() > 0.75:
        # 75% of customers will have at least one claim
        pass

    # Policy start date (mirrors Eng 1 logic)
    earliest_start = SIMULATION_START
    policy_start = random_date(
        earliest_start,
        SIMULATION_END - timedelta(days=365),
    )

    # Claim date — somewhere after policy start
    claim_date = random_date(policy_start, SIMULATION_END)

    # Tenure at claim
    tenure_months = (claim_date.year - policy_start.year) * 12 + \
                    (claim_date.month - policy_start.month)
    tenure_months = max(tenure_months, 1)

    # Prior claims (cumulative)
    prior_claims = int(np.random.poisson(0.3))

    # Claim type
    types, weights = claim_type_weights(customer["policy_type"])
    claim_type = np.random.choice(types, p=weights)

    # Severity
    severity = assign_severity(customer, claim_type, tenure_months, prior_claims)

    # Incident behavioural flags
    police_report = bool(
        claim_type in ["Motor theft", "Home burglary", "Home fire"] and np.random.random() < 0.85
    ) or bool(np.random.random() < 0.15)
    witness_present = bool(np.random.random() < 0.30)
    days_to_report = int(np.random.exponential(7)) + 1
    days_to_report = min(days_to_report, 90)

    # Claim amount
    amount = compute_claim_amount(
        claim_type, severity, customer, tenure_months,
        prior_claims, days_to_report, police_report,
    )

    return {
        "claim_id": f"CLM-{claim_index:07d}",
        "customer_id": customer["customer_id"],
        "claim_date": claim_date,
        "policy_start_date": policy_start,
        "claim_type": claim_type,
        "policy_type": customer["policy_type"],
        "customer_segment": customer["customer_segment"],
        "customer_age": customer["customer_age"],
        "region": customer["region"],
        "tenure_months": tenure_months,
        "incident_severity": severity,
        "police_report_filed": police_report,
        "witness_present": witness_present,
        "days_to_report": days_to_report,
        "prior_claims": prior_claims,
        "claim_amount": amount,
    }


# ============================================
# BUILD DATASET
# ============================================

def build_dataset(n_customers: int = N_CUSTOMERS):
    print(f"Generating claims for {n_customers} customers...")

    claim_records = []
    claim_counter = 0

    for i in range(1, n_customers + 1):
        # Each customer has ~1 claim over the window (75% chance of at least one)
        n_claims = np.random.choice([0, 1, 2, 3], p=[0.25, 0.55, 0.15, 0.05])
        for _ in range(n_claims):
            claim_counter += 1
            claim = generate_claim(claim_counter, i)
            if claim is not None:
                claim_records.append(claim)

    claims_df = pd.DataFrame(claim_records)

    print(f"Claims generated:    {len(claims_df):,}")
    print(f"\nClaim amount stats (£):")
    print(claims_df["claim_amount"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
    ).round(2))
    print(f"\nClaim type distribution:")
    print(claims_df["claim_type"].value_counts())
    print(f"\nSeverity distribution:")
    print(claims_df["incident_severity"].value_counts(normalize=True).round(3))
    print(f"\nBy policy type:")
    print(claims_df.groupby("policy_type")["claim_amount"].agg(["count", "mean", "median"]).round(2))

    return claims_df


if __name__ == "__main__":
    claims_df = build_dataset()

    path = RAW_DATA_DIR / "claims.csv"
    claims_df.to_csv(path, index=False)
    print(f"\nSaved: {path}")