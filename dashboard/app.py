import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🏦",
    layout="wide",
)

DATA_PATH = Path("data/transactions.jsonl")


@st.cache_data
def load_data():
    records = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main():
    st.title("Fraud Detection Dashboard")
    st.markdown("Pipeline de detection de fraude bancaire en temps reel")

    df = load_data()

    # KPIs
    total = len(df)
    n_fraud = int(df["is_fraud"].sum())
    fraud_pct = n_fraud / total * 100
    fraud_amount = df[df["is_fraud"] == 1]["amount"].sum()
    avg_fraud = df[df["is_fraud"] == 1]["amount"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Fraudes Detectees", f"{n_fraud:,}", f"{fraud_pct:.2f}%")
    col3.metric("Montant Fraude Total", f"{fraud_amount:,.0f} EUR")
    col4.metric("Montant Moyen Fraude", f"{avg_fraud:,.0f} EUR")

    st.markdown("---")

    # Graphiques ligne 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transactions par heure")
        hourly = df.groupby(["hour_of_day", "is_fraud"]).size().reset_index(name="count")
        hourly["type"] = hourly["is_fraud"].map({0: "Legitime", 1: "Fraude"})
        fig = px.bar(
            hourly, x="hour_of_day", y="count", color="type",
            color_discrete_map={"Legitime": "#2196F3", "Fraude": "#F44336"},
            labels={"hour_of_day": "Heure", "count": "Nombre"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraudes par categorie")
        cat = df.groupby("merchant_category").agg(
            total=("is_fraud", "count"),
            fraudes=("is_fraud", "sum")
        ).reset_index()
        cat["taux"] = (cat["fraudes"] / cat["total"] * 100).round(2)
        cat = cat.sort_values("taux", ascending=True)
        fig = px.bar(
            cat, x="taux", y="merchant_category", orientation="h",
            color="taux",
            color_continuous_scale="Reds",
            labels={"taux": "Taux de fraude (%)", "merchant_category": "Categorie"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Graphiques ligne 2
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution des montants")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[df["is_fraud"] == 0]["amount"].clip(upper=500),
            name="Legitime", opacity=0.7,
            marker_color="#2196F3", nbinsx=50
        ))
        fig.add_trace(go.Histogram(
            x=df[df["is_fraud"] == 1]["amount"].clip(upper=5000),
            name="Fraude", opacity=0.7,
            marker_color="#F44336", nbinsx=50
        ))
        fig.update_layout(barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top pays frauduleux")
        countries = df[df["is_fraud"] == 1]["merchant_country"].value_counts().reset_index()
        countries.columns = ["pays", "fraudes"]
        fig = px.bar(
            countries.head(8), x="fraudes", y="pays", orientation="h",
            color="fraudes", color_continuous_scale="Reds",
            labels={"fraudes": "Nombre de fraudes", "pays": "Pays"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Evolution journaliere
    st.subheader("Evolution journaliere des fraudes")
    daily = df.groupby(df["timestamp"].dt.date).agg(
        total=("is_fraud", "count"),
        fraudes=("is_fraud", "sum")
    ).reset_index()
    daily["taux"] = (daily["fraudes"] / daily["total"] * 100).round(2)
    fig = px.line(
        daily, x="timestamp", y="taux",
        labels={"timestamp": "Date", "taux": "Taux de fraude (%)"},
        color_discrete_sequence=["#F44336"],
    )
    fig.add_hline(y=2, line_dash="dash", line_color="gray", annotation_text="Cible 2%")
    st.plotly_chart(fig, use_container_width=True)

    # Tableau des dernieres fraudes
    st.subheader("Dernieres fraudes detectees")
    fraudes = df[df["is_fraud"] == 1].sort_values("timestamp", ascending=False).head(10)
    st.dataframe(
        fraudes[[
            "timestamp", "transaction_id", "merchant_category",
            "merchant_country", "amount", "fraud_type"
        ]].rename(columns={
            "timestamp": "Date",
            "transaction_id": "ID Transaction",
            "merchant_category": "Categorie",
            "merchant_country": "Pays",
            "amount": "Montant (EUR)",
            "fraud_type": "Type de fraude",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Testeur de transaction
    st.markdown("---")
    st.subheader("Testeur de scoring en temps reel")
    st.markdown("Teste une transaction via l'API FastAPI")

    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Montant (EUR)", min_value=1.0, max_value=10000.0, value=2000.0)
        category = st.selectbox("Categorie", ["electronique", "supermarche", "luxe", "ecommerce", "carburant"])
    with col2:
        country = st.selectbox("Pays marchand", ["FR", "NG", "CN", "RO", "BE", "DE", "BR", "PH"])
        hour = st.slider("Heure de la transaction", 0, 23, 2)
    with col3:
        is_online = st.checkbox("Transaction en ligne", value=True)
        is_international = st.checkbox("Internationale", value=True)

    if st.button("Scorer la transaction", type="primary"):
        import requests
        tx = {
            "transaction_id": "test-dashboard-001",
            "client_id": "client-test",
            "merchant_id": "merchant-test",
            "merchant_category": category,
            "merchant_country": country,
            "amount": amount,
            "currency": "EUR",
            "hour_of_day": hour,
            "day_of_week": 1,
            "is_online": is_online,
            "is_international": is_international,
        }
        try:
            r = requests.post("http://127.0.0.1:8000/score", json=tx, timeout=5)
            result = r.json()
            score = result["fraud_score"]
            risk = result["risk_level"]

            if result["is_fraud"]:
                st.error(f"FRAUDE DETECTEE — Score: {score:.4f} — Risque: {risk}")
            else:
                st.success(f"Transaction legitime — Score: {score:.4f} — Risque: {risk}")

            st.json(result["explanation"])
        except Exception as e:
            st.warning(f"API non disponible. Lance d'abord: uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()