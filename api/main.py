import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
from xgboost import XGBClassifier
from ml.features import build_features, FEATURE_COLUMNS
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="API de scoring de fraude bancaire en temps reel",
    version="1.0.0",
)

MODEL_PATH = "models/fraud_model.json"
model = None


def load_model():
    global model
    try:
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        logger.info(f"Modele charge depuis {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Erreur chargement modele: {e}")
        raise


@app.on_event("startup")
def startup_event():
    load_model()


class TransactionRequest(BaseModel):
    transaction_id: str
    client_id: str
    merchant_id: str
    merchant_category: str
    merchant_country: str
    amount: float
    currency: str = "EUR"
    hour_of_day: int
    day_of_week: int
    is_online: bool
    is_international: bool


class FraudResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    risk_level: str
    explanation: dict


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/score", "/health", "/docs"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/score", response_model=FraudResponse)
def score_transaction(transaction: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    try:
        df = pd.DataFrame([transaction.dict()])
        df = build_features(df)
        X = df[FEATURE_COLUMNS]

        fraud_score = float(model.predict_proba(X)[0][1])
        is_fraud = fraud_score >= 0.5

        if fraud_score >= 0.8:
            risk_level = "CRITIQUE"
        elif fraud_score >= 0.5:
            risk_level = "ELEVE"
        elif fraud_score >= 0.3:
            risk_level = "MODERE"
        else:
            risk_level = "FAIBLE"

        HIGH_RISK_COUNTRIES = ["NG", "RO", "UA", "CN", "BR", "RU", "PH", "ID"]
        explanation = {
            "montant_eleve": transaction.amount > 500,
            "pays_a_risque": transaction.merchant_country in HIGH_RISK_COUNTRIES,
            "transaction_internationale": transaction.is_international,
            "heure_suspecte": transaction.hour_of_day < 6,
            "montant_rond": transaction.amount % 100 == 0,
        }

        logger.info(
            f"Transaction {transaction.transaction_id[:8]}... | "
            f"Score: {fraud_score:.4f} | "
            f"Risque: {risk_level}"
        )

        return FraudResponse(
            transaction_id=transaction.transaction_id,
            fraud_score=round(fraud_score, 4),
            is_fraud=is_fraud,
            risk_level=risk_level,
            explanation=explanation,
        )

    except Exception as e:
        logger.error(f"Erreur scoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    return {
        "model": "XGBoost",
        "features": FEATURE_COLUMNS,
        "threshold": 0.5,
        "fraud_rate_training": "2%",
    }