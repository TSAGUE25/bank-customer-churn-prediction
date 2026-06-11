# Bank Customer Churn Prediction

> Pipeline ML complet de prédiction du désabonnement bancaire : comparaison Logistic Regression, Random Forest et Gradient Boosting, validation croisée stratifiée, analyse des variables prédictives et recommandations de rétention actionnables.
> **Stack :** Python · scikit-learn · pandas · matplotlib · seaborn

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Portfolio](https://img.shields.io/badge/Portfolio-TSAGUE%20Emmanuel-purple)](https://github.com/TSAGUE25)

---

## Table des matières

1. [Contexte métier](#1-contexte-métier)
2. [Problème résolu](#2-problème-résolu)
3. [Données utilisées](#3-données-utilisées)
4. [Méthodes et algorithmes](#4-méthodes-et-algorithmes)
5. [Démarche analytique](#5-démarche-analytique)
6. [Métriques clés](#6-métriques-clés)
7. [Résultats obtenus](#7-résultats-obtenus)
8. [Valeur métier](#8-valeur-métier)
9. [Architecture du projet](#9-architecture-du-projet)
10. [Installation et usage](#10-installation-et-usage)
11. [Compétences démontrées](#11-compétences-démontrées)
12. [Limites et améliorations](#12-limites-et-améliorations)
13. [Contributors](#13-contributors)

---

## 1. Contexte métier

Acquérir un nouveau client bancaire coûte **5 à 7 fois plus cher** que retenir un client existant. Le churn (désabonnement) est donc un enjeu critique pour la rentabilité des banques de détail.

Les signaux de départ existent dans les données : un client qui vide son compte, qui n'utilise plus ses services, dont le score de crédit se dégrade — ces comportements sont détectables avant le départ effectif.

Ce projet simule ce cas sur **150 clients** (données fictives) avec pour objectif d'identifier les clients à risque suffisamment tôt pour déclencher des actions de rétention ciblées.

---

## 2. Problème résolu

> *"Notre taux de churn mensuel est de 3%. On l'apprend quand le client a déjà fermé son compte. Comment identifier les clients en risque de départ 30 à 60 jours à l'avance pour intervenir ?"*

Ce projet apporte :
- **Score de probabilité de churn** par client (0 à 1)
- **Identification des variables** les plus prédictives
- **Pipeline reproductible** sklearn — scaler + classifier encapsulés pour éviter le data leakage
- **Recommandations d'action** par niveau de risque

| Objectif | Méthode |
|----------|---------|
| Explorer les corrélations avec le churn | EDA, groupby, distributions |
| Construire 3 modèles comparables | LR, RF, GB via Pipeline sklearn |
| Valider par CV stratifiée (k=5) | StratifiedKFold + roc_auc |
| Identifier les variables prédictives | Feature importance RF/GB |
| Analyser les faux négatifs | Clients churned non détectés |

---

## 3. Données utilisées

> **Données entièrement simulées — aucune donnée réelle ou confidentielle.**
> Structure inspirée du dataset public Kaggle "Bank Customer Churn".

### `bank_customers.csv` — 150 clients fictifs

| Colonne | Description | Exemple |
|---------|-------------|---------|
| `age` | Âge du client | 42 |
| `genre` | F / M | F |
| `pays` | France / Espagne | France |
| `score_credit` | Score de solvabilité (350–850) | 619 |
| `solde` | Solde du compte (€) | 0.00 |
| `nb_produits` | Nb de produits bancaires (1–4) | 1 |
| `membre_actif` | Activité récente (0/1) | 1 |
| `salaire_estime` | Salaire annuel estimé (€) | 101 348 |
| `anciennete_annees` | Ancienneté en banque | 2 |
| `churn` | **Cible : 0=Resté, 1=Parti** | 1 |

---

## 4. Méthodes et algorithmes

| Algorithme | Avantage | Inconvénient |
|-----------|----------|-------------|
| **Logistic Regression** | Interprétable, rapide, baseline solide | Linéaire, capte mal les interactions |
| **Random Forest** | Robuste, feature importance, parallélisable | Moins interprétable |
| **Gradient Boosting** | Souvent meilleur AUC, gère les non-linéarités | Plus lent à entraîner |

**Validation :** StratifiedKFold (k=5) — garantit la représentativité des churners dans chaque fold.

**Pipeline sklearn :** `StandardScaler + Classifier` en un seul objet — évite le data leakage entre train et test.

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier(n_estimators=200, random_state=42))
])
```

---

## 5. Démarche analytique

```
Dataset bank_customers.csv (150 clients)
        │
        ▼
    EDA + Profil churners (distributions, taux par variable)
        │
        ▼
    Encodage + split stratifié (75/25)
        │
        ├──→ Pipeline LR     ──→ AUC / CV-AUC / F1
        ├──→ Pipeline RF     ──→ AUC / CV-AUC / F1
        └──→ Pipeline GB     ──→ AUC / CV-AUC / F1
                │
                ▼
        Sélection meilleur modèle (AUC)
                │
                ├──→ Feature importance
                ├──→ Matrice de confusion
                ├──→ Courbe ROC + PR
                ├──→ Analyse faux négatifs
                └──→ Recommandations métier
                        │
                        ▼
                Rapport Markdown + 8 figures
```

---

## 6. Métriques clés

| Métrique | Pourquoi l'utiliser |
|----------|---------------------|
| **AUC ROC** | Robuste au déséquilibre, insensible au seuil |
| **CV-AUC** | Estimation fiable sur 5 folds — évite l'overfitting |
| **Rappel (churners)** | Minimiser les churners manqués |
| **Précision (churners)** | Éviter les fausses alarmes coûteuses |
| **F1-score** | Équilibre précision/rappel |
| **AP Score** | Meilleur que AUC sur classes très déséquilibrées |

**Accuracy vs AUC :** l'Accuracy est trompeuse sur un problème déséquilibré. Un modèle qui prédit "jamais de churn" obtient 85% d'accuracy si le taux de churn est 15% — et détecte 0 churner. L'AUC ROC mesure la capacité réelle à distinguer les deux classes.

---

## 7. Résultats obtenus

### Comparaison des 3 modèles

| Modèle | AUC Test | CV-AUC | F1-Churn |
|--------|---------|--------|---------|
| Logistic Regression | ~0.72 | ~0.70 | ~0.58 |
| **Random Forest** | **~0.81** | **~0.78** | **~0.65** |
| Gradient Boosting | ~0.79 | ~0.76 | ~0.62 |

### Variables les plus prédictives (Random Forest)

| Rang | Variable | Interprétation |
|------|---------|----------------|
| 1 | `age` | Clients > 50 ans churne davantage |
| 2 | `solde` | Solde nul = signal fort de départ |
| 3 | `nb_produits` | 1 seul produit = faible engagement |
| 4 | `score_credit` | Score faible corrélé au churn |
| 5 | `membre_actif` | Inactifs = risque ×2 |

---

## 8. Valeur métier

- **Score de risque individuel** : chaque client reçoit une probabilité de churn de 0 à 100%
- **Priorisation** : concentrer les efforts de rétention sur les top 20% risques
- **Ciblage précis** : campagne personnalisée selon le facteur déclenchant identifié

**Exemple de ROI :**
- 50 churners détectés sur 150 clients
- 20% retenus par offre ciblée → ~10 clients × 1 500 €/an = **15 000 €/an préservés**
- Coût campagne ciblée : 10 × 50 € = 500 €
- **ROI estimé : 3 000%**

**Adapté pour :** banque, assurance, télécoms, e-commerce, SaaS — tout secteur avec un problème de rétention client.

---

## 9. Architecture du projet

```
bank-customer-churn-prediction/
│
├── data_sample/
│   ├── bank_customers.csv    # 150 clients fictifs, 12 colonnes
│   └── schema_reference.md  # Dictionnaire + signaux de churn
│
├── src/
│   ├── __init__.py
│   ├── churn_predictor.py   # Classe ChurnPredictor (Pipeline LR/RF/GB)
│   └── visualization.py     # Classe ChurnVisualizer (8 figures)
│
├── notebooks/
│   └── 01_churn_prediction.py  # Pipeline complet 9 sections
│
├── figures/                 # 8 visualisations générées
├── reports/
│   └── churn_report_sample.md
│
├── docs/
│   ├── methodologie.md
│   └── schema_reference.md
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 10. Installation et usage

```bash
git clone https://github.com/TSAGUE25/bank-customer-churn-prediction
cd bank-customer-churn-prediction
pip install -r requirements.txt
python notebooks/01_churn_prediction.py
```

**Utilisation directe :**

```python
import pandas as pd
from src.churn_predictor import ChurnPredictor

df        = pd.read_csv('data_sample/bank_customers.csv')
predictor = ChurnPredictor(df)
results   = predictor.train_all()

print(predictor.summary())
print(predictor.feature_importance())
print(predictor.false_negatives())
```

**Sorties produites :**
- `figures/` — 8 visualisations PNG (ROC, PR, confusion matrix, feature importance...)
- `reports/churn_report_sample.md` — rapport Markdown avec métriques et recommandations

---

## 11. Compétences démontrées

| Compétence | Mise en œuvre | Fichier |
|-----------|--------------|---------|
| **Pipeline sklearn** | `Pipeline([scaler, clf])` — anti data leakage | `src/churn_predictor.py` |
| **3 modèles ML** | LR / RF / GB — comparaison rigoureuse | `src/churn_predictor.py` |
| **StratifiedKFold** | CV stratifiée 5-fold | `src/churn_predictor.py` |
| **Feature importance** | RF + GB — variables prédictives classées | `src/churn_predictor.py` |
| **Analyse faux négatifs** | Profil des churners non détectés | `src/churn_predictor.py` |
| **8 visualisations** | ROC, PR, confusion matrix, distributions | `src/visualization.py` |
| **Python OOP** | Classes `ChurnPredictor`, `ChurnVisualizer` | `src/` |
| **Rapport automatisé** | Markdown généré programmatiquement | `notebooks/` |

**Stack technique :** `pandas` · `scikit-learn` (Pipeline, LR, RF, GB, StratifiedKFold) · `matplotlib` · `seaborn`

---

## 12. Limites et améliorations

**Limites actuelles :**

| Limite | Impact |
|--------|--------|
| 150 clients seulement | Overfitting possible — à déployer sur 10 000+ |
| Pas de SMOTE | Déséquilibre des classes non traité |
| Seuil fixe à 0.50 | Sous-optimal selon les coûts métier réels |
| Pas de SHAP | Interprétabilité individuelle limitée |

**Pistes d'amélioration :**
- **SMOTE** : rééquilibrage synthétique de la classe minoritaire
- **SHAP** : explication individuelle de chaque prédiction (pourquoi CE client va churner)
- **XGBoost / LightGBM** : souvent supérieurs au Gradient Boosting sklearn
- **Optimisation du seuil** : courbe coût-bénéfice pour maximiser le ROI de la campagne
- **Déploiement API** : Flask / FastAPI pour servir le modèle en temps réel

---

## Ce projet démontre

- La maîtrise du **Pipeline sklearn** (`StandardScaler + Classifier`) pour éliminer le data leakage et industrialiser la chaîne d'entraînement
- La **comparaison rigoureuse de 3 modèles** (Logistic Regression, Random Forest, Gradient Boosting) avec validation croisée stratifiée 5-fold
- Le choix de l'**AUC ROC comme métrique principale** sur un problème déséquilibré — et la démonstration de pourquoi l'accuracy est trompeuse
- L'**analyse des faux négatifs** : profiler les churners non détectés pour comprendre les limites du modèle et les améliorer
- La traduction des **feature importances en recommandations métier** : quoi offrir à un client dont le signal est un solde nul ou une inactivité
- Un pipeline **réutilisable sur tout problème de rétention** : banque, assurance, télécoms, SaaS — adapter les colonnes et les données

---

## 13. Contributors

| Nom | Rôle | GitHub |
|-----|------|--------|
| **TSAGUE Emmanuel** | Data Scientist — auteur principal | [@TSAGUE25](https://github.com/TSAGUE25) |

---

*Auteur : Emmanuel TSAGUE — Data Scientist / Data Analyst*
*Formation : DataScientest | Domaines : Finance · Banque · Énergie · Commerce*
*Contact : emmatsague@yahoo.fr | [LinkedIn](https://www.linkedin.com/in/emmanuel-tsague-114295414)*
*Données : entièrement simulées — aucune donnée réelle ou confidentielle*
