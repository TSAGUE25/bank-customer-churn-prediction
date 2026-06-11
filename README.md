# Bank Customer Churn Prediction

> Prédiction du churn bancaire par Machine Learning : Logistic Regression, Random Forest et Gradient Boosting, comparés par AUC ROC, avec analyse des variables prédictives et recommandations de rétention.  
> **Stack :** Python · scikit-learn · pandas · matplotlib · seaborn

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Portfolio](https://img.shields.io/badge/Portfolio-Data%20Science-orange)](https://github.com/TSAGUE25)

---

## Table des matières

1. [Titre et accroche](#1-titre-et-accroche)
2. [Contexte métier](#2-contexte-métier)
3. [Pourquoi ce projet existe](#3-pourquoi-ce-projet-existe)
4. [Problème métier](#4-problème-métier)
5. [Objectifs](#5-objectifs)
6. [Données utilisées](#6-données-utilisées)
7. [Préparation des données](#7-préparation-des-données)
8. [Méthodes et algorithmes](#8-méthodes-et-algorithmes)
9. [Démarche analytique](#9-démarche-analytique)
10. [Métriques clés](#10-métriques-clés)
11. [Explication des métriques](#11-explication-des-métriques)
12. [Résultats obtenus](#12-résultats-obtenus)
13. [Valeur métier](#13-valeur-métier)
14. [Limites du projet](#14-limites-du-projet)
15. [Améliorations possibles](#15-améliorations-possibles)
16. [Architecture du dépôt](#16-architecture-du-dépôt)
17. [README technique](#17-readme-technique)
18. [Version CV](#18-version-cv)
19. [Version entretien](#19-version-entretien)
20. [Version portfolio](#20-version-portfolio)
21. [Post LinkedIn](#21-post-linkedin)
22. [Questions d'entretien](#22-questions-dentretien)
23. [Compétences démontrées](#23-compétences-démontrées)
24. [Tableau compétences / preuves](#24-tableau-compétences--preuves)
25. [Conseils GitHub](#25-conseils-github)

---

## 1. Titre et accroche

**Bank Customer Churn Prediction** — Pipeline ML complet de prédiction du désabonnement bancaire : EDA, feature engineering, comparaison de 3 algorithmes (Logistic Regression, Random Forest, Gradient Boosting), validation croisée stratifiée, analyse des variables prédictives et recommandations de rétention actionnables.

> *Ce projet démontre la capacité à construire un pipeline ML de bout en bout, à choisir la bonne métrique selon le contexte métier, et à traduire un modèle en actions business concrètes.*

---

## 2. Contexte métier

Acquérir un nouveau client bancaire coûte **5 à 7 fois plus cher** que retenir un client existant. Le churn (désabonnement) est donc un enjeu critique pour la rentabilité des banques de détail.

Les signaux de départ existent dans les données : un client qui vide son compte, qui n'utilise plus ses services, dont le score de crédit se dégrade — ces comportements sont détectables avant le départ effectif.

Ce projet simule ce cas sur **150 clients** (données fictives) avec pour objectif d'identifier les clients à risque suffisamment tôt pour déclencher des actions de rétention ciblées.

---

## 3. Pourquoi ce projet existe

**Problème concret :** Une banque perd chaque mois des clients sans savoir pourquoi, ni lesquels elle aurait pu retenir. Elle envoie des offres de rétention à tout le monde — coûteux et peu efficace.

**Ce que ce projet apporte :**
- Score de probabilité de churn pour chaque client (0 à 1)
- Identification des variables les plus prédictives
- Pipeline reproductible et extensible (sklearn Pipeline)
- Recommandations d'action par niveau de risque

---

## 4. Problème métier

> *"Notre taux de churn mensuel est de 3%. On l'apprend quand le client a déjà fermé son compte. Comment identifier les clients en risque de départ 30 à 60 jours à l'avance pour intervenir ?"*

**Traduction analytique :**
- Variable cible binaire : `churn` = 1 si le client quitte la banque
- Maximiser le rappel (ne pas manquer de churners) tout en contrôlant la précision
- AUC ROC comme métrique principale (robuste au déséquilibre)

---

## 5. Objectifs

| # | Objectif | Méthode |
|---|----------|---------|
| 1 | Explorer les corrélations avec le churn | EDA, groupby, distributions |
| 2 | Profiler les churners vs non-churners | Statistiques comparatives |
| 3 | Construire 3 modèles comparables | LR, RF, GB via Pipeline sklearn |
| 4 | Valider par CV stratifiée (k=5) | StratifiedKFold + roc_auc |
| 5 | Comparer AUC, F1, Précision, Rappel | Tableau de bord des métriques |
| 6 | Identifier les variables prédictives | Feature importance (RF/GB) |
| 7 | Analyser les faux négatifs | Clients churned non détectés |
| 8 | Produire des recommandations métier | Plan d'action par segment de risque |

---

## 6. Données utilisées

> **Données entièrement simulées — aucune donnée réelle ou confidentielle.**  
> Structure inspirée du dataset public Kaggle "Bank Customer Churn" (anonymisé et reéchantillonné).

### bank_customers.csv — 150 clients fictifs

| Colonne | Description | Exemple |
|---------|-------------|---------|
| `age` | Âge du client | 42 |
| `genre` | F / M | F |
| `pays` | France / Espagne | France |
| `score_credit` | Score de solvabilité (350–850) | 619 |
| `solde` | Solde du compte (€) | 0.00 |
| `nb_produits` | Nb de produits bancaires (1–4) | 1 |
| `carte_credit` | Carte crédit (0/1) | 1 |
| `membre_actif` | Activité récente (0/1) | 1 |
| `salaire_estime` | Salaire annuel estimé (€) | 101 348 |
| `anciennete_annees` | Ancienneté en banque | 2 |
| `churn` | **Cible : 0=Resté, 1=Parti** | 1 |

---

## 7. Préparation des données

### Encodage des variables catégorielles

```python
from sklearn.preprocessing import LabelEncoder
df['genre_enc'] = LabelEncoder().fit_transform(df['genre'])  # F=0, M=1
df['pays_enc']  = LabelEncoder().fit_transform(df['pays'])   # Espagne=0, France=1
```

### Découpage train/test stratifié

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
```

### Pipeline sklearn (StandardScaler + Classifier)

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier(n_estimators=200, random_state=42))
])
```

---

## 8. Méthodes et algorithmes

| Algorithme | Avantage | Inconvénient |
|-----------|----------|-------------|
| **Logistic Regression** | Interprétable, rapide, baseline solide | Linéaire, capte mal les interactions |
| **Random Forest** | Robuste, feature importance, parallélisable | Moins interprétable, mémoire |
| **Gradient Boosting** | Souvent meilleur AUC, gère les non-linéarités | Plus lent à entraîner |

**Validation :** StratifiedKFold (k=5) pour garantir la représentativité des churners dans chaque fold.

---

## 9. Démarche analytique

```
Dataset bank_customers.csv (150 clients)
        │
        ▼
    EDA + Profil churners
    (distributions, taux par variable)
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

## 10. Métriques clés

| Métrique | Formule | Pourquoi l'utiliser |
|----------|---------|---------------------|
| **AUC ROC** | Aire sous courbe ROC | Robuste au déséquilibre, insensible au seuil |
| **CV-AUC** | Moyenne AUC sur 5 folds | Estimation plus fiable, évite l'overfitting |
| **Rappel (churners)** | TP / (TP + FN) | Minimiser les churners manqués |
| **Précision (churners)** | TP / (TP + FP) | Éviter les fausses alarmes coûteuses |
| **F1-score** | 2 × (P × R) / (P + R) | Équilibre précision/rappel |
| **AP Score** | Aire sous courbe PR | Meilleur que AUC sur classes très déséquilibrées |

---

## 11. Explication des métriques

### AUC ROC vs Accuracy

L'Accuracy est trompeuse sur un problème déséquilibré : un modèle qui prédit "jamais de churn" obtient 85% d'accuracy si le taux de churn est 15%. L'AUC ROC mesure la capacité à distinguer les churners des non-churners indépendamment du seuil.

### Précision vs Rappel — le dilemme du churn

- **Rappel élevé** : on détecte plus de churners, mais on contacte aussi des non-churners inutilement (coût de campagne)
- **Précision élevée** : on ne contacte que les vrais churners, mais on en manque certains (coût du churn)

Le bon équilibre dépend du coût de rétention vs le coût d'acquisition d'un nouveau client.

### Feature importance (Random Forest)

Mesure la réduction d'impureté (Gini) apportée par chaque variable à travers tous les arbres. Une variable avec une importance élevée est très discriminante pour séparer churners et non-churners.

---

## 12. Résultats obtenus

### Comparaison des 3 modèles (dataset simulé, 150 clients)

| Modèle | AUC Test | CV-AUC | AP Score | F1-Churn |
|--------|---------|--------|---------|---------|
| Logistic Regression | ~0.72 | ~0.70 | ~0.55 | ~0.58 |
| Random Forest | **~0.81** | **~0.78** | **~0.68** | **~0.65** |
| Gradient Boosting | ~0.79 | ~0.76 | ~0.65 | ~0.62 |

### Variables les plus prédictives (Random Forest)

| Rang | Variable | Interprétation |
|------|---------|----------------|
| 1 | `age` | Clients > 50 ans churne davantage |
| 2 | `solde` | Solde nul = signal fort de départ |
| 3 | `nb_produits` | 1 seul produit = faible engagement |
| 4 | `score_credit` | Score faible corrélé au churn |
| 5 | `membre_actif` | Inactifs = risque x2 |

---

## 13. Valeur métier

### Pour un directeur de la relation client
- **Score de risque individuel** : chaque client reçoit une probabilité de churn de 0 à 100%
- **Priorisation** : concentrer les efforts de rétention sur les top 20% risques
- **Ciblage précis** : campagne personnalisée selon le facteur déclenchant identifié

### Exemple de ROI concret
- Taux de churn actuel : 33% × 150 clients = 50 churners
- Si 20% des churners prédits avec proba > 0.70 sont retenus par une offre ciblée :
  - ~10 clients retenus × 1 500 €/an de revenu moyen = **15 000 €/an préservés**
  - Coût de la campagne ciblée : 10 × 50 € = 500 €
  - **ROI = 3 000%**

---

## 14. Limites du projet

| Limite | Impact | Mitigation |
|--------|--------|-----------|
| 150 clients seulement | Overfitting possible | Déployer sur 10 000+ clients |
| Pas de SMOTE | Déséquilibre non traité | Ajouter imbalanced-learn |
| Seuil fixe à 0.50 | Sous-optimal selon les coûts | Optimiser par courbe coût-bénéfice |
| Pas de données temporelles | Tendances non capturées | Ajouter features rolling (30j, 90j) |
| Pas de SHAP | Interprétabilité limitée | Ajouter shap library |

---

## 15. Améliorations possibles

- **SMOTE** : rééquilibrage synthétique de la classe minoritaire (imbalanced-learn)
- **SHAP** : explication individuelle de chaque prédiction (pourquoi CE client va churner)
- **XGBoost / LightGBM** : souvent supérieurs au Gradient Boosting sklearn
- **Optimisation du seuil** : courbe coût-bénéfice pour maximiser le ROI de la campagne
- **Features comportementales** : fréquence de connexion, volume de transactions, délai dernier virement
- **Déploiement API** : Flask / FastAPI pour servir le modèle en temps réel

---

## 16. Architecture du dépôt

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
│   └── 01_churn_prediction.py  # Script complet 9 sections
│
├── figures/                 # 8 visualisations générées
├── reports/
│   └── churn_report_sample.md  # Rapport avec métriques et recommandations
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

## 17. README technique

### Installation

```bash
git clone https://github.com/TSAGUE25/bank-customer-churn-prediction
cd bank-customer-churn-prediction
pip install -r requirements.txt
```

### Exécution

```bash
python notebooks/01_churn_prediction.py
```

### Utilisation directe

```python
import pandas as pd
from src.churn_predictor import ChurnPredictor

df = pd.read_csv('data_sample/bank_customers.csv')
predictor = ChurnPredictor(df)
results   = predictor.train_all()

print(predictor.summary())
print(predictor.feature_importance())
print(predictor.false_negatives())
```

---

## 18. Version CV

> *À copier dans la section "Projets" du CV*

**Bank Customer Churn Prediction** | Python, scikit-learn, pandas, matplotlib  
Pipeline ML de prédiction du churn bancaire : comparaison LR / Random Forest / Gradient Boosting, validation croisée stratifiée (k=5), AUC ROC ~0.81 (RF), analyse de l'importance des variables, courbes ROC et PR, matrice de confusion. Architecture Pipeline sklearn, rapport Markdown automatisé, 8 visualisations.

---

## 19. Version entretien

*Question : "Comment construiriez-vous un modèle de prédiction du churn ?"*

> "La première décision n'est pas le choix de l'algorithme — c'est le choix de la métrique. Sur un problème de churn, l'Accuracy est inutile. Ce qui compte, c'est l'AUC ROC et surtout le rappel : un churner non détecté est perdu définitivement, alors qu'une fausse alarme coûte juste une campagne de rétention.
>
> Ensuite je construis un pipeline sklearn — Scaler + Classifier en un seul objet — pour éviter le data leakage. Je compare 3 algorithmes : Logistic Regression comme baseline interprétable, Random Forest pour la robustesse, Gradient Boosting pour potentiellement le meilleur AUC.
>
> La validation est stratifiée (StratifiedKFold) pour garantir que chaque fold contient la bonne proportion de churners. Sur ce dataset simulé, le Random Forest donne AUC ~0.81 et CV-AUC ~0.78.
>
> Enfin, l'étape souvent oubliée : l'analyse des faux négatifs — les churners non détectés. Ont-ils des caractéristiques communes ? C'est souvent là qu'on identifie les features manquantes."

---

## 20. Version portfolio

Ce projet illustre le cycle complet d'un projet ML de classification binaire : de l'EDA jusqu'au rapport d'aide à la décision, en passant par la comparaison rigoureuse de modèles. Il démontre la maîtrise des pipelines sklearn, le choix de métriques adaptées au contexte métier et la capacité à produire des recommandations actionnables.

**Adapté pour :** Banque, assurance, télécoms, e-commerce, SaaS — tout secteur avec un problème de rétention client.

---

## 21. Post LinkedIn

> **J'ai construit un modèle de prédiction de churn bancaire — voici la leçon sur les métriques**
>
> 🚨 L'erreur classique : optimiser l'Accuracy sur un problème de churn.
>
> Si 85% de vos clients ne churne pas, un modèle qui prédit "jamais de churn" obtient 85% d'accuracy. Et détecte 0 churner.
>
> La bonne approche :
>
> **AUC ROC** — robuste au déséquilibre, indépendant du seuil  
> **Rappel** — minimiser les churners non détectés  
> **Validation croisée stratifiée** — chaque fold représentatif  
>
> Résultats sur mon dataset simulé (150 clients) :
> - Logistic Regression : AUC 0.72 (baseline interprétable)
> - Random Forest : AUC **0.81** (meilleur modèle)
> - Gradient Boosting : AUC 0.79
>
> Variables les plus prédictives : solde nul, membre inactif, âge > 50 ans.
>
> Code sur GitHub  
> #DataScience #MachineLearning #Churn #Python #sklearn #Portfolio

---

## 22. Questions d'entretien

**Q1 : Pourquoi ne pas utiliser l'Accuracy comme métrique ?**  
Sur un problème déséquilibré (5% de churners), un modèle naïf qui prédit toujours 0 a 95% d'accuracy mais détecte 0 churner. L'AUC ROC et le F1-score sur la classe minoritaire sont bien plus pertinents.

**Q2 : Quelle est la différence entre AUC ROC et Average Precision (AP) ?**  
L'AUC ROC mesure la capacité à distinguer les deux classes sur toute la plage FPR. L'AP (aire sous la courbe Précision-Rappel) est plus sensible aux performances sur la classe positive — préférable quand le déséquilibre est très fort (< 5% de churners).

**Q3 : Pourquoi utiliser un Pipeline sklearn plutôt que scaler/fit séparément ?**  
Le Pipeline évite le data leakage : si on fit le scaler sur l'ensemble train+test avant de splitter, les informations du test contaminent l'entraînement. Le Pipeline garantit que le scaler est uniquement fité sur le train.

**Q4 : Comment choisiriez-vous le seuil de décision optimal ?**  
Tracer la courbe coût-bénéfice : `ROI(seuil) = gains_rétention × TP - coûts_campagne × (TP + FP)`. Le seuil optimal maximise ce ROI selon les paramètres métier réels (coût d'acquisition, valeur client, coût de la campagne).

**Q5 : Quand préférerait-on DBSCAN à K-means pour de la segmentation client ?**  
DBSCAN quand les clusters ont des formes irrégulières ou quand on a des outliers à isoler (clients "VIP exceptionnel"). K-means suppose des clusters sphériques et de taille similaire.

---

## 23. Compétences démontrées

- **scikit-learn Pipeline** — StandardScaler + Classifier encapsulés
- **Classification ML** — LR, Random Forest, Gradient Boosting
- **Métriques ML** — AUC ROC, AP Score, F1, précision, rappel
- **Validation croisée** — StratifiedKFold(k=5), cross_val_score
- **Feature importance** — GradientBoostingClassifier, RandomForestClassifier
- **Visualisations** — ROC, confusion matrix, distributions, PR curve, feature importance
- **Analyse métier** — profil churners, faux négatifs, ROI de rétention
- **Python OOP** — classes `ChurnPredictor` et `ChurnVisualizer`

---

## 24. Tableau compétences / preuves

| Compétence | Preuve | Fichier |
|-----------|--------|---------|
| Pipeline sklearn | `Pipeline([scaler, clf])` | `src/churn_predictor.py` |
| 3 modèles comparés | Méthode `train_all()` | `src/churn_predictor.py` |
| StratifiedKFold | CV stratifiée 5-fold | `src/churn_predictor.py` |
| Feature importance | Méthode `feature_importance()` | `src/churn_predictor.py` |
| Faux négatifs | Méthode `false_negatives()` | `src/churn_predictor.py` |
| Profil churners | Méthode `churn_profile()` | `src/churn_predictor.py` |
| 8 visualisations | Classe `ChurnVisualizer` | `src/visualization.py` |
| ROC + PR + CM | Figures 3, 4, 6 | `src/visualization.py` |
| Script complet | 9 sections | `notebooks/01_churn_prediction.py` |

---

## 25. Conseils GitHub

**Description :** "Prédiction du churn bancaire — LR / RF / GB comparés, AUC ROC 0.81, feature importance, recommandations de rétention. Portfolio Data Science."

**Topics :** `python` `data-science` `machine-learning` `churn-prediction` `scikit-learn` `random-forest` `gradient-boosting` `classification` `banking` `portfolio`

**Projets connexes :**
| Projet | Lien |
|--------|------|
| Customer Segmentation | [customer-marketing-segmentation](https://github.com/TSAGUE25/customer-marketing-segmentation) |
| Data Quality Audit | [data-quality-audit-framework](https://github.com/TSAGUE25/data-quality-audit-framework) |
| Building Energy Analytics | [building-energy-efficiency-analytics](https://github.com/TSAGUE25/building-energy-efficiency-analytics) |


## Contributors

| Nom | Role | GitHub |
|-----|------|--------|
| **TSAGUE Emmanuel** | Data Scientist - auteur principal | [@TSAGUE25](https://github.com/TSAGUE25) |

---

*Auteur : Emmanuel TSAGUE — Data Scientist / Data Analyst*  
*Formation : DataScientest | Domaines : Finance · Banque · Energie · Commerce*  
*Contact : emmatsague@yahoo.fr*  
*Données : entièrement simulées — aucune donnée réelle ou confidentielle*
