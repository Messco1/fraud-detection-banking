import json
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/transactions.jsonl")


def load_data(path=DATA_PATH):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def print_summary(df):
    total = len(df)
    n_fraud = int(df["is_fraud"].sum())
    fraud_pct = n_fraud / total * 100

    print("=" * 55)
    print("RESUME DU DATASET")
    print("=" * 55)
    print(f"  Total transactions   : {total:>10,}")
    print(f"  Fraudes              : {n_fraud:>10,}  ({fraud_pct:.2f}%)")
    print(f"  Legitimes            : {total - n_fraud:>10,}")
    print(f"  Clients uniques      : {df['client_id'].nunique():>10,}")
    print(f"  Marchands uniques    : {df['merchant_id'].nunique():>10,}")
    print(f"  Periode              : {df['timestamp'].min().date()} - {df['timestamp'].max().date()}")
    print()

    print("MONTANTS (euros)")
    print("-" * 40)
    for label, mask in [("Legitimes", df["is_fraud"] == 0), ("Fraudes", df["is_fraud"] == 1)]:
        sub = df.loc[mask, "amount"]
        print(f"  {label:<12} | moy={sub.mean():>8.2f} | med={sub.median():>8.2f} | max={sub.max():>8.2f}")
    print()

    print("REPARTITION PAR CATEGORIE")
    print("-" * 40)
    cat_stats = (
        df.groupby("merchant_category")
        .agg(n=("transaction_id", "count"), fraud_rate=("is_fraud", "mean"))
        .sort_values("n", ascending=False)
    )
    for cat, row in cat_stats.iterrows():
        bar = "#" * int(row["n"] / len(df) * 40)
        print(f"  {cat:<16} {bar:<42} {row['n']:>5} txns  |  fraude: {row['fraud_rate']*100:.1f}%")
    print()

    print("PAYS MARCHANDS (fraudes)")
    print("-" * 40)
    fraud_countries = df[df["is_fraud"] == 1]["merchant_country"].value_counts().head(10)
    for country, count in fraud_countries.items():
        print(f"  {country}: {count}")
    print()

    print("TYPES DE FRAUDE")
    print("-" * 40)
    fraud_types = df[df["is_fraud"] == 1]["fraud_type"].value_counts()
    for ftype, count in fraud_types.items():
        print(f"  {ftype:<25}: {count:>4}  ({count/n_fraud*100:.1f}%)")
    print()

    print("FEATURES CLES POUR LE MODELE")
    print("-" * 40)
    print("  is_international :", df.groupby("is_fraud")["is_international"].mean().to_dict())
    print("  is_online        :", df.groupby("is_fraud")["is_online"].mean().to_dict())
    print()


def export_csv(df, path="data/transactions.csv"):
    df.to_csv(path, index=False)
    print(f"CSV exporte : {path}")


if __name__ == "__main__":
    print("Chargement des donnees...")
    df = load_data()
    print_summary(df)
    export_csv(df)