# 🏦 Fraud Detection System — Real-Time Banking Pipeline

Pipeline complète de détection de fraude bancaire en temps réel.

## Stack technique

| Couche | Technologies |
|---|---|
| Ingestion | Python, Faker, Apache Kafka |
| Processing | Kafka Consumer, XGBoost |
| Versioning ML | MLflow |
| Stockage | PostgreSQL, DuckDB |
| Transformation | dbt |
| Orchestration | Apache Airflow |
| API | FastAPI |
| Visualisation | Streamlit, Plotly |
| Infra | Docker, Docker Compose |

## Lancement rapide
```bash
python data_generator/generator.py --n 10000
python data_generator/eda.py
```

## Étapes du projet

- [x] Étape 1 — Génération de données synthétiques
- [ ] Étape 2 — Pipeline Kafka
- [ ] Étape 3 — Feature Engineering + XGBoost + MLflow
- [ ] Étape 4 — Stockage PostgreSQL + dbt
- [ ] Étape 5 — API FastAPI
- [ ] Étape 6 — Dashboard Streamlit
- [ ] Étape 7 — Docker Compose
```

---

## Fichier 7 — `.gitignore`
```
data/*.jsonl
data/*.csv
data/*.parquet
models/
mlruns/
__pycache__/
*.py[cod]
.env
venv/
.venv/
.DS_Store