import json
import time
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kafka_pipeline import KafkaProducer
from data_generator.generator import generate_clients, generate_merchants, generate_transaction
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TOPIC = "bank-transactions"
BOOTSTRAP_SERVERS = "localhost:9092"
DELAY_SECONDS = 0.5


def create_producer():
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )


def run_producer(delay: float = DELAY_SECONDS):
    logger.info("Chargement des clients et marchands...")
    clients = generate_clients()
    merchants = generate_merchants()

    logger.info(f"Connexion a Kafka sur {BOOTSTRAP_SERVERS}...")
    producer = create_producer()
    logger.info(f"Producer connecte. Envoi vers le topic '{TOPIC}'...")
    logger.info("Ctrl+C pour arreter.")

    count = 0
    fraud_count = 0

    try:
        while True:
            import random
            client = random.choice(clients)
            from dataclasses import asdict
            tx = generate_transaction(client, merchants, datetime.now())
            tx_dict = asdict(tx)

            producer.send(
                topic=TOPIC,
                key=tx_dict["client_id"],
                value=tx_dict,
            )

            count += 1
            if tx_dict["is_fraud"]:
                fraud_count += 1
                logger.warning(
                    f"[FRAUDE #{fraud_count}] {tx_dict['transaction_id'][:8]}... "
                    f"| {tx_dict['amount']}EUR | {tx_dict['merchant_country']}"
                )
            else:
                if count % 10 == 0:
                    logger.info(f"[OK] {count} transactions envoyees | {fraud_count} fraudes")

            time.sleep(delay)

    except KeyboardInterrupt:
        logger.info(f"Arret. {count} transactions envoyees, {fraud_count} fraudes.")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    run_producer()