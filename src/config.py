# ============================================
# GUARDIANSHIELD INSURANCE
# ENGAGEMENT 3 — CLAIMS SEVERITY PREDICTION
# CONFIGURATION
# ============================================

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, IMAGES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "guardianshield_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

DB_SCHEMA = os.getenv("DB_SCHEMA", "claims")

DB_URL = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

N_CUSTOMERS = 5_000
SIMULATION_START = "2018-01-01"
SIMULATION_END = "2025-12-31"
RANDOM_SEED = 42