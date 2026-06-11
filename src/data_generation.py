# -*- coding: utf-8 -*-
"""Génération de données simulées — churn bancaire (16% taux d'attrition)."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_churn_data(n: int = 15_000, seed: int = 42) -> pd.DataFrame:
    """Génère un dataset simulé de clients bancaires avec churn ~16%."""
    np.random.seed(seed)
    df = pd.DataFrame({
        "credit_score":     np.random.randint(300, 850, n),
        "age":              np.random.randint(18, 75, n),
        "anciennete_ans":   np.random.randint(0, 20, n),
        "solde":            np.abs(np.random.normal(60_000, 30_000, n)),
        "nb_produits":      np.random.randint(1, 5, n),
        "carte_credit":     np.random.randint(0, 2, n),
        "membre_actif":     np.random.randint(0, 2, n),
        "salaire_annuel":   np.abs(np.random.normal(55_000, 20_000, n)),
        "pays":             np.random.choice(["France", "Espagne", "Allemagne"], n,
                                              p=[0.50, 0.25, 0.25]),
        "sexe":             np.random.choice(["H", "F"], n),
    })
    prob = (
        0.05
        + (df["credit_score"]   < 500).astype(float) * 0.15
        + (df["age"]            > 60).astype(float)  * 0.10
        + (df["nb_produits"]    == 1).astype(float)  * 0.12
        + (df["membre_actif"]   == 0).astype(float)  * 0.20
        + (df["anciennete_ans"] < 2).astype(float)   * 0.08
        + np.random.uniform(-0.05, 0.05, n)
    )
    df["churn"] = (prob > 0.25).astype(int)
    return df


def load_or_generate(csv_path=None, **kwargs) -> pd.DataFrame:
    if csv_path and Path(csv_path).exists():
        return pd.read_csv(csv_path)
    return generate_churn_data(**kwargs)


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "data_sample" / "bank_churn_simulated.csv"
    df = generate_churn_data()
    df.to_csv(out, index=False)
    print(f"Créé : {out} | {len(df):,} lignes | churn : {df.churn.mean():.1%}")
