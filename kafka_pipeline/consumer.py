import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kafka_pipeline import KafkaConsumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TOPIC = "bank-transactions"
BOOTSTRAP_SERVERS = "localhost:9092"
GROUP_ID = "fraud-detection-group"


def run_consumer():
    logger.info(f"Connexion a Kafka sur {BOOTSTRAP_SERVERS}...")

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=GROUP_ID,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )

    logger.info(f"Consumer connecte. Ecoute du topic '{TOPIC}'...")
    logger.info("Ctrl+C pour arreter.")

    count = 0
    fraud_count = 0

    try:
        for message in consumer:
            tx = message.value
            count += 1

            if tx.get("is_fraud"):
                fraud_count += 1
                logger.warning(
                    f"[FRAUDE DETECTEE] "
                    f"ID: {tx['transaction_id'][:8]}... | "
                    f"Montant: {tx['amount']}EUR | "
                    f"Pays: {tx['merchant_country']} | "
                    f"Type: {tx['fraud_type']}"
                )
            else:
                if count % 10 == 0:
                    logger.info(
                        f"[OK] {count} recus | {fraud_count} fraudes | "
                        f"Dernier: {tx['amount']}EUR - {tx['merchant_category']}"
                    )

    except KeyboardInterrupt:
        logger.info(f"Arret. {count} transactions recues, {fraud_count} fraudes.")
    finally:
        consumer.close()


if __name__ == "__main__":
    run_consumer()