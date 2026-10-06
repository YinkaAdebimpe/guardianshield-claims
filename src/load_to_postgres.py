# ============================================
# GUARDIANSHIELD INSURANCE
# ENGAGEMENT 3 — LOAD CLAIMS TO POSTGRES
# ============================================

import pandas as pd
from sqlalchemy import create_engine, text

from config import DB_URL, DB_SCHEMA, RAW_DATA_DIR


def load_csvs():
    claims = pd.read_csv(RAW_DATA_DIR / "claims.csv")

    claims["claim_date"] = pd.to_datetime(claims["claim_date"]).dt.date
    claims["policy_start_date"] = pd.to_datetime(claims["policy_start_date"]).dt.date

    return claims


def ensure_schema(engine):
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA};"))
        print(f"Schema '{DB_SCHEMA}' ready.")


def write_table(df: pd.DataFrame, table_name: str, engine):
    df.to_sql(
        name=table_name,
        con=engine,
        schema=DB_SCHEMA,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000,
    )
    print(f"Wrote {len(df):,} rows → {DB_SCHEMA}.{table_name}")


if __name__ == "__main__":
    print("Reading CSVs...")
    claims = load_csvs()

    print(f"Loaded claims: {len(claims):,}")

    print("\nConnecting to Postgres...")
    engine = create_engine(DB_URL)

    ensure_schema(engine)

    print("\nWriting table...")
    write_table(claims, "claims", engine)

    with engine.connect() as conn:
        count = conn.execute(
            text(f"SELECT COUNT(*) FROM {DB_SCHEMA}.claims;")
        ).scalar()

    print(f"\nVerification:")
    print(f"  {DB_SCHEMA}.claims:  {count:,} rows")

    print("\n✓ Data loaded successfully.")