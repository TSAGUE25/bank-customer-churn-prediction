# Prédiction du Churn Bancaire

> **Identifier les clients à risque de départ avec XGBoost et SHAP**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Domaine](https://img.shields.io/badge/Domaine-Banque-green)
![Statut](https://img.shields.io/badge/Statut-Portfolio-orange)
![Données](https://img.shields.io/badge/Données-Simulées%2FAnonymisées-lightgrey)

---

## Contexte métier

La rétention client est un enjeu stratégique dans le secteur bancaire. Acquérir un nouveau client coûte 5 à 25 fois plus cher que retenir un client existant. La prédiction proactive du churn permet d'agir avant le départ.

---

## Problème traité

15 000 clients simulés avec 16 % de taux de churn. Construire un modèle robuste malgré le déséquilibre de classes, maximiser le recall et expliquer les prédictions individuelles aux équipes commerciales.

---

## Solution proposée

XGBoost avec scale_pos_weight pour le déséquilibre, optimisation du seuil de décision par courbe Précision-Rappel (F1 maximal), SHAP waterfall pour les explications individuelles, analyse coût FP/FN.

---

## Technologies utilisées

| Outil | Usage |
|-------|-------|
| Python 3.10+ | Langage principal |
| pandas / numpy | Manipulation des données |
| scikit-learn | Machine Learning & preprocessing |
| matplotlib / seaborn | Visualisation |
| Jupyter Notebook | Exploration interactive |

> Voir `requirements.txt` pour la liste complète.

---

## Structure du projet

```
bank-customer-churn-prediction/
├── README.md              ← Ce fichier
├── PORTFOLIO.md           ← Documentation complète du cas d'usage
├── .gitignore
├── requirements.txt
├── notebooks/             ← Jupyter Notebooks d'exploration
├── src/                   ← Code Python modulaire
├── data_sample/           ← Données simulées (anonymisées)
├── figures/               ← Graphiques et visualisations
├── reports/               ← Rapports et synthèses
└── docs/                  ← Documentation complémentaire
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/TSAGUE25/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter
jupyter notebook
```

---

## Métriques clés (données simulées)

```
ROC-AUC = 0.87 | Recall = 0.79 | F1 = 0.70 (simulés)
```

---

## Valeur métier

Économie simulée de 485 000 € via actions de rétention ciblées.

---

## Limites

Données simulées. Pas de déploiement temps réel. Seuil figé à l'entraînement.

---

## Prochaines améliorations

API FastAPI pour scoring temps réel. Monitoring drift. Réentraînement automatique.

---

## Avertissement — Confidentialité

> **Toutes les données utilisées dans ce projet sont simulées, synthétiques ou anonymisées.**
> Aucune donnée réelle, confidentielle ou propriétaire n'est présente dans ce dépôt.
> Ce projet est un cas d'usage pédagogique à destination du portfolio professionnel d'Emmanuel TSAGUE.

---

## Contributors

**TSAGUE EMMANUEL** - Data Scientist  
Specialise en Machine Learning, Data Analysis et systemes decisionnels.  
Formation Datascientest 2024 | EDF MAD EDVANCE  
Email : [emmatsague@yahoo.fr](mailto:emmatsague@yahoo.fr)  
LinkedIn : [emmanuel-tsague-114295414](https://www.linkedin.com/in/emmanuel-tsague-114295414)  
GitHub : [github.com/TSAGUE25](https://github.com/TSAGUE25)

