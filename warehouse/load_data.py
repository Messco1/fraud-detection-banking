import json
import duckdb
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "warehouse/fraud.duckdb"
DATA_PATH = "data/transactions.jsonl"


def load_transactions():
    logger.info("Connexion a DuckDB...")
    con = duckdb.connect(DB_PATH)

    logger.info("Chargement des transactions depuis JSONL...")
    records = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    logger.info("Creation de la table raw_transactions...")
    con.execute("DROP TABLE IF EXISTS raw_transactions")
    con.execute("""
        CREATE TABLE raw_transactions (
            transaction_id VARCHAR,
            client_id VARCHAR,
            merchant_id VARCHAR,
            merchant_category VARCHAR,
            merchant_country VARCHAR,
            amount DOUBLE,
            currency VARCHAR,
            timestamp TIMESTAMP,
            hour_of_day INTEGER,
            day_of_week INTEGER,
            is_online BOOLEAN,
            is_international BOOLEAN,
            is_fraud INTEGER,
            fraud_type VARCHAR
        )
    """)

    con.execute("INSERT INTO raw_transactions SELECT * FROM df")

    count = con.execute("SELECT COUNT(*) FROM raw_transactions").fetchone()[0]
    fraud_count = con.execute("SELECT COUNT(*) FROM raw_transactions WHERE is_fraud = 1").fetchone()[0]

    logger.info(f"Table raw_transactions creee : {count} transactions, {fraud_count} fraudes")

    logger.info("Creation des vues analytiques...")

    con.execute("""
        CREATE OR REPLACE VIEW fraud_by_category AS
        SELECT
            merchant_category,
            COUNT(*) as total_transactions,
            SUM(is_fraud) as total_frauds,
            ROUND(AVG(is_fraud) * 100, 2) as fraud_rate_pct,
            ROUND(AVG(amount), 2) as avg_amount,
            ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount END), 2) as avg_fraud_amount
        FROM raw_transactions
        GROUP BY merchant_category
        ORDER BY fraud_rate_pct DESC
    """)

    con.execute("""
        CREATE OR REPLACE VIEW fraud_by_hour AS
        SELECT
            hour_of_day,
            COUNT(*) as total_transactions,
            SUM(is_fraud) as total_frauds,
            ROUND(AVG(is_fraud) * 100, 2) as fraud_rate_pct
        FROM raw_transactions
        GROUP BY hour_of_day
        ORDER BY hour_of_day
    """)

    con.execute("""
        CREATE OR REPLACE VIEW fraud_by_country AS
        SELECT
            merchant_country,
            COUNT(*) as total_transactions,
            SUM(is_fraud) as total_frauds,
            ROUND(AVG(is_fraud) * 100, 2) as fraud_rate_pct,
            ROUND(AVG(amount), 2) as avg_amount
        FROM raw_transactions
        GROUP BY merchant_country
        ORDER BY total_frauds DESC
    """)

    con.execute("""
        CREATE OR REPLACE VIEW daily_summary AS
        SELECT
            CAST(timestamp AS DATE) as date,
            COUNT(*) as total_transactions,
            SUM(is_fraud) as total_frauds,
            ROUND(AVG(is_fraud) * 100, 2) as fraud_rate_pct,
            ROUND(SUM(amount), 2) as total_amount,
            ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) as fraud_amount
        FROM raw_transactions
        GROUP BY CAST(timestamp AS DATE)
        ORDER BY date
    """)

    logger.info("Verification des vues...")
    print("\n--- FRAUDES PAR CATEGORIE ---")
    print(con.execute("SELECT * FROM fraud_by_category").df().to_string(index=False))

    print("\n--- TOP 5 PAYS FRAUDULEUX ---")
    print(con.execute("SELECT * FROM fraud_by_country WHERE total_frauds > 0 LIMIT 5").df().to_string(index=False))

    con.close()
    logger.info(f"Base DuckDB sauvegardee dans {DB_PATH}")


if __name__ == "__main__":
    load_transactions()