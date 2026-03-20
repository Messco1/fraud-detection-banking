import json
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features import build_features, FEATURE_COLUMNS, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/transactions.jsonl")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "fraud-detection"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    logger.info(f"Dataset charge: {len(df)} transactions, {df['is_fraud'].sum()} fraudes ({df['is_fraud'].mean()*100:.2f}%)")
    return df


def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Chargement des donnees...")
    df = load_data()

    logger.info("Construction des features...")
    df = build_features(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    logger.info("Split train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    logger.info("Application de SMOTE pour equilibrer les classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(f"Apres SMOTE: {y_train_res.value_counts().to_dict()}")

    with mlflow.start_run(run_name="xgboost-fraud-detector"):

        params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": 1,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        mlflow.log_params(params)

        logger.info("Entrainement du modele XGBoost...")
        model = XGBClassifier(**params)
        model.fit(X_train_res, y_train_res)

        logger.info("Evaluation sur le jeu de test...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        print("\n" + "="*55)
        print("RESULTATS DU MODELE")
        print("="*55)
        print(f"  Precision  : {precision:.4f}")
        print(f"  Recall     : {recall:.4f}")
        print(f"  F1 Score   : {f1:.4f}")
        print(f"  ROC AUC    : {auc:.4f}")
        print()
        print("RAPPORT DE CLASSIFICATION")
        print("-"*40)
        print(classification_report(y_test, y_pred, target_names=["Legitime", "Fraude"]))
        print("MATRICE DE CONFUSION")
        print("-"*40)
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Vrais Negatifs  : {cm[0][0]}")
        print(f"  Faux Positifs   : {cm[0][1]}")
        print(f"  Faux Negatifs   : {cm[1][0]}")
        print(f"  Vrais Positifs  : {cm[1][1]}")
        print()

        logger.info("Sauvegarde du modele...")
        mlflow.sklearn.log_model(model, "xgboost-fraud-model")
        model_path = MODELS_DIR / "fraud_model.json"
        model.save_model(str(model_path))
        logger.info(f"Modele sauvegarde dans {model_path}")

        feature_importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        print("IMPORTANCE DES FEATURES")
        print("-"*40)
        for feat, imp in feature_importance.items():
            bar = "#" * int(imp * 100)
            print(f"  {feat:<25} {bar:<30} {imp:.4f}")

    logger.info("Entrainement termine avec succes !")
    return model


if __name__ == "__main__":
    train()