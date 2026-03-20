"""
Générateur de transactions bancaires synthétiques.
"""

import uuid
import random
import json
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from faker import Faker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

fake = Faker("fr_FR")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FRAUD_RATE = 0.02
NUM_CLIENTS = 500
NUM_MERCHANTS = 200

MERCHANT_CATEGORIES = {
    "supermarche":  {"mean": 65,  "std": 40,  "weight": 25},
    "restaurant":   {"mean": 35,  "std": 20,  "weight": 20},
    "carburant":    {"mean": 80,  "std": 25,  "weight": 15},
    "ecommerce":    {"mean": 120, "std": 90,  "weight": 15},
    "pharmacie":    {"mean": 45,  "std": 30,  "weight": 10},
    "transport":    {"mean": 55,  "std": 35,  "weight": 8},
    "luxe":         {"mean": 800, "std": 500, "weight": 3},
    "electronique": {"mean": 350, "std": 200, "weight": 4},
}

LEGIT_COUNTRIES  = ["FR", "FR", "FR", "FR", "BE", "ES", "DE", "IT"]
FRAUD_COUNTRIES  = ["NG", "RO", "UA", "CN", "BR", "RU", "PH", "ID"]


# ---------------------------------------------------------------------------
# Modèles de données
# ---------------------------------------------------------------------------

@dataclass
class Client:
    client_id: str
    name: str
    email: str
    age: int
    country: str
    avg_monthly_spend: float
    home_city: str


@dataclass
class Merchant:
    merchant_id: str
    name: str
    category: str
    city: str
    country: str


@dataclass
class Transaction:
    transaction_id: str
    client_id: str
    merchant_id: str
    merchant_category: str
    merchant_country: str
    amount: float
    currency: str
    timestamp: str
    hour_of_day: int
    day_of_week: int
    is_online: bool
    is_international: bool
    is_fraud: int
    fraud_type: Optional[str]


# ---------------------------------------------------------------------------
# Génération des entités
# ---------------------------------------------------------------------------

def generate_clients(n: int = NUM_CLIENTS) -> list:
    clients = []
    for _ in range(n):
        age = int(np.random.normal(42, 15))
        age = max(18, min(85, age))
        base_spend = np.random.lognormal(mean=7.0, sigma=0.6)
        clients.append(Client(
            client_id=str(uuid.uuid4()),
            name=fake.name(),
            email=fake.email(),
            age=age,
            country="FR",
            avg_monthly_spend=round(base_spend, 2),
            home_city=fake.city(),
        ))
    logger.info(f"{len(clients)} clients générés")
    return clients


def generate_merchants(n: int = NUM_MERCHANTS) -> list:
    merchants = []
    categories = list(MERCHANT_CATEGORIES.keys())
    weights = [MERCHANT_CATEGORIES[c]["weight"] for c in categories]
    for _ in range(n):
        category = random.choices(categories, weights=weights, k=1)[0]
        country = random.choices(
            ["FR", "FR", "FR", "BE", "DE", "ES"],
            weights=[60, 10, 10, 8, 6, 6], k=1
        )[0]
        merchants.append(Merchant(
            merchant_id=str(uuid.uuid4()),
            name=fake.company(),
            category=category,
            city=fake.city(),
            country=country,
        ))
    logger.info(f"{len(merchants)} marchands générés")
    return merchants


# ---------------------------------------------------------------------------
# Génération des transactions
# ---------------------------------------------------------------------------

def _legit_amount(category: str) -> float:
    cfg = MERCHANT_CATEGORIES.get(category, {"mean": 80, "std": 50})
    amount = np.random.normal(cfg["mean"], cfg["std"])
    return round(max(1.0, amount), 2)


def _fraud_amount(category: str) -> float:
    fraud_pattern = random.choices(
        ["high", "round", "low"],
        weights=[50, 30, 20], k=1
    )[0]
    if fraud_pattern == "high":
        return round(random.uniform(500, 4999), 2)
    elif fraud_pattern == "round":
        return random.choice([100, 200, 300, 500, 1000, 1500, 2000])
    else:
        return round(random.uniform(1, 20), 2)


def generate_transaction(client, merchants: list, base_time: datetime, force_fraud: bool = False):
    is_fraud = force_fraud or (random.random() < FRAUD_RATE)
    merchant = random.choice(merchants)

    if is_fraud:
        hour = random.choices(
            list(range(24)),
            weights=[8,8,7,6,5,4,2,1,1,1,1,1,1,1,1,1,1,2,3,4,5,6,7,8],
            k=1
        )[0]
    else:
        hour = random.choices(
            list(range(24)),
            weights=[1,1,1,1,1,2,3,5,7,8,8,9,9,8,8,8,7,7,6,5,4,3,2,2],
            k=1
        )[0]

    tx_time = base_time.replace(
        hour=hour,
        minute=random.randint(0, 59),
        second=random.randint(0, 59)
    )

    if is_fraud:
        merchant_country = random.choice(FRAUD_COUNTRIES)
        is_international = True
    else:
        merchant_country = merchant.country
        is_international = merchant_country != client.country

    amount = _fraud_amount(merchant.category) if is_fraud else _legit_amount(merchant.category)

    fraud_type = None
    if is_fraud:
        fraud_type = random.choice([
            "card_not_present",
            "account_takeover",
            "identity_theft",
            "lost_stolen_card",
            "friendly_fraud",
        ])

    return Transaction(
        transaction_id=str(uuid.uuid4()),
        client_id=client.client_id,
        merchant_id=merchant.merchant_id,
        merchant_category=merchant.category,
        merchant_country=merchant_country,
        amount=amount,
        currency="EUR",
        timestamp=tx_time.isoformat(),
        hour_of_day=hour,
        day_of_week=tx_time.weekday(),
        is_online=random.random() < 0.35,
        is_international=is_international,
        is_fraud=int(is_fraud),
        fraud_type=fraud_type,
    )


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def generate_batch(n_transactions: int = 10_000, output_path: str = "data/transactions.jsonl", start_date=None) -> list:
    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)

    clients = generate_clients()
    merchants = generate_merchants()
    transactions = []
    n_fraud = 0

    logger.info(f"Génération de {n_transactions} transactions...")

    for i in range(n_transactions):
        progress = i / n_transactions
        current_time = start_date + timedelta(days=90 * progress)
        client = random.choice(clients)
        tx = generate_transaction(client, merchants, current_time)
        tx_dict = asdict(tx)
        transactions.append(tx_dict)
        if tx.is_fraud:
            n_fraud += 1
        if (i + 1) % 1000 == 0:
            logger.info(f"  {i+1}/{n_transactions} transactions générées...")

    with open(output_path, "w", encoding="utf-8") as f:
        for tx in transactions:
            f.write(json.dumps(tx, ensure_ascii=False) + "\n")

    fraud_pct = n_fraud / n_transactions * 100
    logger.info(f"✅ {n_transactions} transactions sauvegardées dans '{output_path}'")
    logger.info(f"   → {n_fraud} fraudes ({fraud_pct:.2f}%) — cible : ~2%")
    return transactions


def stream_transactions(clients: list, merchants: list, delay_seconds: float = 0.5):
    logger.info("Démarrage du stream (Ctrl+C pour arrêter)...")
    count = 0
    while True:
        client = random.choice(clients)
        tx = generate_transaction(client, merchants, datetime.now())
        count += 1
        if tx.is_fraud:
            logger.warning(f"[FRAUDE #{count}] {tx.transaction_id} | {tx.amount}€ | {tx.merchant_country}")
        yield asdict(tx)
        time.sleep(delay_seconds)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--output", type=str, default="data/transactions.jsonl")
    args = parser.parse_args()
    generate_batch(n_transactions=args.n, output_path=args.output)