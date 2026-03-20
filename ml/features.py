import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features pour le modele ML.
    Prend un DataFrame de transactions brutes et retourne les features.
    """
    df = df.copy()

    # Features numeriques directes
    df["amount_log"] = np.log1p(df["amount"])

    # Features binaires
    df["is_online"] = df["is_online"].astype(int)
    df["is_international"] = df["is_international"].astype(int)

    # Features temporelles
    df["is_night"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 5)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = ((df["hour_of_day"] >= 10) & (df["hour_of_day"] <= 14)).astype(int)

    # Encodage des categories
    df["category_encoded"] = df["merchant_category"].map({
        "supermarche": 0,
        "restaurant": 1,
        "carburant": 2,
        "ecommerce": 3,
        "pharmacie": 4,
        "transport": 5,
        "luxe": 6,
        "electronique": 7,
    }).fillna(-1)

    # Pays a risque
    HIGH_RISK_COUNTRIES = ["NG", "RO", "UA", "CN", "BR", "RU", "PH", "ID"]
    df["is_high_risk_country"] = df["merchant_country"].isin(HIGH_RISK_COUNTRIES).astype(int)

    # Montant anormalement rond (signe de fraude)
    df["is_round_amount"] = (df["amount"] % 100 == 0).astype(int)

    # Montant tres eleve
    df["is_high_amount"] = (df["amount"] > 500).astype(int)

    return df


# Liste des features utilisees par le modele
FEATURE_COLUMNS = [
    "amount",
    "amount_log",
    "hour_of_day",
    "day_of_week",
    "is_online",
    "is_international",
    "is_night",
    "is_weekend",
    "is_peak_hour",
    "category_encoded",
    "is_high_risk_country",
    "is_round_amount",
    "is_high_amount",
]

TARGET_COLUMN = "is_fraud"