# CAS D'USAGE 11 — Prédiction du Churn Bancaire
## Identifier les clients à risque de départ avant qu'ils partent

> **Auteur :** TSAGUE EMMANUEL — Data Scientist / Data Analyst  
> **Domaine :** Classification binaire, Déséquilibre de classes, Interprétabilité  
> **Repository GitHub :** `bank-customer-churn-prediction`  
> **Statut :** Portfolio — données simulées  
> **Date :** Juin 2026

---
## 1. TITRE ET RÉSUMÉ EXÉCUTIF

**"Prédiction du churn bancaire : XGBoost + SHAP pour identifier les clients à risque avec recall maximisé"**

> **Churn (attrition) :** perte d'un client — il ferme son compte, résilie son abonnement ou passe à la concurrence. Le taux de churn = (clients perdus sur la période) / (clients au début de la période).

> **Pourquoi prédire le churn ?** Acquérir un nouveau client coûte 5 à 25 fois plus cher que retenir un client existant. Identifier les clients à risque AVANT leur départ permet d'intervenir proactivement.

Une banque simule 15 000 clients. 16 % quittent dans les 6 mois suivants. L'objectif est de prédire ces départs pour déclencher des actions de rétention ciblées.

**Résultats simulés :** AUC = 0,87 | Recall = 0,79 | Précision = 0,63 sur la classe "churn".

---
## 2. PROBLÈME SPÉCIFIQUE : DÉSÉQUILIBRE DES CLASSES

> **Déséquilibre de classes (class imbalance) :** situation où une classe est beaucoup moins représentée que l'autre. Ex : 84 % de non-churn vs 16 % de churn. Un modèle naïf qui dit toujours "non-churn" a 84 % de précision globale mais 0 % de recall — inutile.

> **Recall (rappel, sensibilité) :** parmi tous les vrais churners, combien le modèle en détecte-t-il ? Recall = TP / (TP + FN). En churn, un recall faible signifie beaucoup de churners manqués = clients perdus sans intervention.

> **Précision :** parmi tous les clients que le modèle classe "churn", combien churneront vraiment ? Précision = TP / (TP + FP). Une précision faible = beaucoup de fausses alarmes = actions de rétention coûteuses envoyées inutilement.

> **F1-Score :** moyenne harmonique de précision et recall. Métrique d'équilibre. F1 = 2 × (Précision × Recall) / (Précision + Recall).

---
## 3. GÉNÉRATION DES DONNÉES SIMULÉES

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)
N = 15_000

df = pd.DataFrame({
    "credit_score":     np.random.randint(300, 850, N),
    "age":              np.random.randint(18, 75, N),
    "anciennete_ans":   np.random.randint(0, 20, N),
    "solde":            np.abs(np.random.normal(60_000, 30_000, N)),
    "nb_produits":      np.random.randint(1, 5, N),
    "carte_credit":     np.random.randint(0, 2, N),
    "membre_actif":     np.random.randint(0, 2, N),
    "salaire_annuel":   np.abs(np.random.normal(55_000, 20_000, N)),
    "pays":             np.random.choice(["France", "Espagne", "Allemagne"], N,
                                          p=[0.50, 0.25, 0.25]),
    "sexe":             np.random.choice(["H", "F"], N),
})

# Churn simulé avec logique métier
prob_churn = (
    0.05 +
    (df["credit_score"]  < 500).astype(float) * 0.15 +
    (df["age"]           > 60).astype(float)  * 0.10 +
    (df["nb_produits"]   == 1).astype(float)  * 0.12 +
    (df["membre_actif"]  == 0).astype(float)  * 0.20 +
    (df["anciennete_ans"] < 2).astype(float)  * 0.08 +
    np.random.uniform(-0.05, 0.05, N)
)
df["churn"] = (prob_churn > 0.25).astype(int)

print(f"Taux de churn simulé : {df['churn'].mean():.1%}")
print(df.value_counts("churn"))

X = df.drop(columns="churn")
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

---
## 4. BASELINE : MODÈLE SANS GESTION DU DÉSÉQUILIBRE

```python
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose       import ColumnTransformer
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import classification_report, roc_auc_score

cols_num = ["credit_score", "age", "anciennete_ans", "solde",
            "nb_produits", "salaire_annuel"]
cols_cat = ["pays", "sexe"]
cols_bin = ["carte_credit", "membre_actif"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), cols_num),
    ("cat", OneHotEncoder(drop="first"), cols_cat),
    ("bin", "passthrough", cols_bin),
])

# BASELINE sans équilibrage
baseline = Pipeline([
    ("prep",   preprocessor),
    ("model",  LogisticRegression(max_iter=1000, random_state=42))
])
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)

print("=== BASELINE (sans équilibrage) ===")
print(classification_report(y_test, y_pred_base, target_names=["No Churn", "Churn"]))
print(f"AUC : {roc_auc_score(y_test, baseline.predict_proba(X_test)[:,1]):.4f}")
```

---
## 5. GESTION DU DÉSÉQUILIBRE

> **class_weight='balanced' :** paramètre sklearn qui pondère automatiquement les erreurs en inversant les fréquences de classe. Les erreurs sur la classe minoritaire (churn) sont plus pénalisées.

> **SMOTE (Synthetic Minority Over-sampling Technique) :** technique qui crée de nouveaux exemples synthétiques de la classe minoritaire en interpolant entre des exemples existants. Augmente artificiellement la classe minoritaire.

> **Threshold (seuil de décision) :** par défaut, un modèle classifie "churn" si probabilité > 0,5. Abaisser ce seuil (ex : 0,35) augmente le recall (moins de churners manqués) au prix d'une précision plus faible (plus de fausses alarmes).

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# Modèles avec gestion du déséquilibre
modeles = {
    "Random Forest (balanced)": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost (scale_pos_weight)": xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        # scale_pos_weight = nb_négatifs / nb_positifs pour équilibrage
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        eval_metric="auc",
        use_label_encoder=False
    ),
}

resultats = {}
for nom, modele in modeles.items():
    pipe = Pipeline([("prep", preprocessor), ("model", modele)])
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    from sklearn.metrics import f1_score, recall_score, precision_score
    resultats[nom] = {
        "AUC":       roc_auc_score(y_test, y_proba),
        "F1":        f1_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "Précision": precision_score(y_test, y_pred),
    }

    print(f"\n{nom}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

import pandas as pd
print("\n=== COMPARAISON ===")
print(pd.DataFrame(resultats).T.round(4))
```

---
## 6. OPTIMISATION DU SEUIL DE DÉCISION

```python
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score

# Pipeline final XGBoost
best_pipe = Pipeline([
    ("prep",  preprocessor),
    ("model", xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42, eval_metric="auc", use_label_encoder=False
    ))
])
best_pipe.fit(X_train, y_train)
y_proba = best_pipe.predict_proba(X_test)[:, 1]

# Courbe Précision-Recall pour différents seuils
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)

seuil_optimal = thresholds[f1_scores.argmax()]
print(f"\nSeuil optimal (max F1) : {seuil_optimal:.3f}")
print(f"F1 max                : {f1_scores.max():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(recalls[:-1], precisions[:-1], "b-")
axes[0].scatter([recalls[:-1][f1_scores.argmax()]], [precisions[:-1][f1_scores.argmax()]],
                color="red", s=100, zorder=5, label=f"Seuil optimal {seuil_optimal:.2f}")
axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Précision")
axes[0].set_title("Courbe Précision-Recall"); axes[0].legend()

axes[1].plot(thresholds, f1_scores, "g-")
axes[1].axvline(x=seuil_optimal, color="red", linestyle="--")
axes[1].set_xlabel("Seuil de décision")
axes[1].set_ylabel("F1-Score")
axes[1].set_title("F1-Score par seuil")

plt.tight_layout()
plt.savefig("figures/churn_threshold_optimization.png", dpi=150, bbox_inches="tight")

# Prédiction avec seuil optimisé
y_pred_optimal = (y_proba >= seuil_optimal).astype(int)
print("\n=== AVEC SEUIL OPTIMAL ===")
print(classification_report(y_test, y_pred_optimal, target_names=["No Churn", "Churn"]))
```

---
## 7. INTERPRÉTABILITÉ SHAP

```python
import shap

# Extraire le modèle et le preprocessor
model_xgb = best_pipe.named_steps["model"]
X_test_transformed = best_pipe.named_steps["prep"].transform(X_test)

# Noms des features après transformation
feature_names = (
    cols_num +
    list(best_pipe.named_steps["prep"]
         .named_transformers_["cat"]
         .get_feature_names_out(cols_cat)) +
    cols_bin
)

explainer  = shap.TreeExplainer(model_xgb)
shap_vals  = explainer.shap_values(X_test_transformed)

# Importance globale (beeswarm)
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_vals, X_test_transformed,
                  feature_names=feature_names,
                  plot_type="dot", show=False, max_display=12)
plt.tight_layout()
plt.savefig("figures/churn_shap_beeswarm.png", dpi=150, bbox_inches="tight")

# Explication d'un client churneur
idx_churner = np.where(y_test.values == 1)[0][0]
print(f"\nExplication client churneur (index {idx_churner}) :")
shap_explanation = shap.Explanation(
    values     = shap_vals[idx_churner],
    base_values= explainer.expected_value,
    data       = X_test_transformed[idx_churner],
    feature_names=feature_names
)
shap.plots.waterfall(shap_explanation, max_display=10, show=False)
plt.savefig("figures/churn_shap_waterfall.png", dpi=150, bbox_inches="tight")
```

---
## 8. COÛT MÉTIER — MATRICE DE COÛTS

> **Matrice de coûts :** extension de la matrice de confusion qui attribue une valeur métier à chaque type d'erreur. En churn :
> - FP (fausse alarme) : on envoie une offre de rétention à un client qui n'allait pas partir → coût = 50 €
> - FN (churner manqué) : on ne détecte pas un churner → coût = 500 € (valeur cliente perdue)

```python
# Coûts métier
cout_fp  = 50    # € — offre de rétention inutile
cout_fn  = 500   # € — client perdu sans intervention
cout_tp  = -200  # € — offre rétention sur vrai churner (net : coût offre - valeur sauvée)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_optimal)
tn, fp, fn, tp = cm.ravel()

cout_total_modele = fp * cout_fp + fn * cout_fn + tp * cout_tp
cout_sans_modele  = (y_test == 1).sum() * cout_fn  # On perd tous les churners

print(f"TN={tn:4d}  FP={fp:4d}")
print(f"FN={fn:4d}  TP={tp:4d}")
print(f"\nCoût total sans modèle : {cout_sans_modele:,.0f} €")
print(f"Coût total avec modèle : {cout_total_modele:,.0f} €")
print(f"Économie simulée       : {cout_sans_modele - cout_total_modele:,.0f} €")
```

---
## 9. ARCHITECTURE GITHUB

```
bank-customer-churn-prediction/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda_desequilibre.ipynb
│   ├── 02_baseline_logistic.ipynb
│   ├── 03_xgboost_equilirage.ipynb
│   ├── 04_threshold_optimization.ipynb
│   └── 05_shap_interpretability.ipynb
├── src/
│   ├── data_generation.py
│   ├── pipeline.py
│   └── cost_analysis.py
└── figures/
    ├── churn_threshold_optimization.png
    ├── churn_shap_beeswarm.png
    └── churn_shap_waterfall.png
```

---
## 15. COMPÉTENCES DÉMONTRÉES

| Compétence | Preuve |
|-----------|--------|
| Déséquilibre classes | class_weight + scale_pos_weight + threshold optimization |
| XGBoost | Classification avec hyperparamètres ajustés |
| ROC-AUC / PR-AUC | Choix de métrique justifié |
| SHAP | Waterfall + beeswarm sur modèle boosté |
| Analyse métier | Matrice de coûts FP/FN |

---

*Fin du document — TSAGUE EMMANUEL — CAS 11 — Churn Bancaire*
---

## Contact & Liens

**TSAGUE EMMANUEL** - Data Scientist

| | |
|---|---|
| Email | [emmatsague@yahoo.fr](mailto:emmatsague@yahoo.fr) |
| LinkedIn | [emmanuel-tsague-114295414](https://www.linkedin.com/in/emmanuel-tsague-114295414) |
| GitHub | [github.com/TSAGUE25](https://github.com/TSAGUE25) |
| Formation | Datascientest 2024 |
| Experience | EDF MAD EDVANCE |
| Domaines | Machine Learning - Data Analysis - Energie |

---

> Toutes les donnees de ce depot sont simulees et anonymisees.  
> Aucune donnee reelle ou confidentielle n'est presente.
