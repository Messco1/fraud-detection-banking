"""
Configuration centralisée du projet.
"""

from dataclasses import dataclass
from pathlib import Path

# ── Chemins ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
MODELS_DIR = BASE_DIR.parent / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ── Génération de données ─────────────────────────────────────────────────────

@dataclass
class GeneratorConfig:
    fraud_rate: float = 0.02
    num_clients: int = 500
    num_merchants: int = 200
    n_transactions: int = 10_000
    output_path: str = str(DATA_DIR / "transactions.jsonl")
    random_seed: int = 42


# ── Kafka ─────────────────────────────────────────────────────────────────────

@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    topic_transactions: str = "bank-transactions"
    topic_scored: str = "bank-transactions-scored"
    consumer_group: str = "fraud-detection-group"
    producer_delay_seconds: float = 0.3


# ── Base de données ───────────────────────────────────────────────────────────

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "fraud_db"
    user: str = "fraud_user"
    password: str = "fraud_pass"

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


# ── MLflow ────────────────────────────────────────────────────────────────────

@dataclass
class MLflowConfig:
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "fraud-detection"
    model_name: str = "fraud-classifier"


# ── API ───────────────────────────────────────────────────────────────────────

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    fraud_threshold: float = 0.5


# ── Instances partagées ───────────────────────────────────────────────────────

generator_config = GeneratorConfig()
kafka_config = KafkaConfig()
db_config = DatabaseConfig()
mlflow_config = MLflowConfig()
api_config = APIConfig()