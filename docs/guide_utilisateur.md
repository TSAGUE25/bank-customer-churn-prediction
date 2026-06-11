# Guide utilisateur — Bank Customer Churn Prediction

## Prérequis

- Python 3.9+
- pip

## Installation

```bash
git clone https://github.com/TSAGUE25/bank-customer-churn-prediction
cd bank-customer-churn-prediction
pip install -r requirements.txt
```

## Lancer l'analyse complète

```bash
python notebooks/01_churn_prediction.py
```

Ce script exécute les 9 étapes dans l'ordre :
1. Chargement et aperçu des données
2. EDA — distributions des variables
3. Profil comparatif churners vs non-churners
4. Entraînement des 3 modèles (LR, RF, GB)
5. Métriques détaillées (AUC, F1, précision, rappel)
6. Importance des variables
7. Analyse des faux négatifs
8. Génération des 8 visualisations (dossier `figures/`)
9. Rapport Markdown automatisé

## Utilisation programmatique

### Entraîner et obtenir les métriques

```python
import pandas as pd
from src.churn_predictor import ChurnPredictor

df        = pd.read_csv('data_sample/bank_customers.csv')
predictor = ChurnPredictor(df)
results   = predictor.train_all()

# Résumé global
print(predictor.summary())

# Importance des variables du meilleur modèle
print(predictor.feature_importance())

# Churners non détectés (faux négatifs)
print(predictor.false_negatives())

# Profil moyen churners vs non-churners
print(predictor.churn_profile())
```

### Générer les visualisations

```python
from src.visualization import ChurnVisualizer

viz = ChurnVisualizer(predictor, output_dir='figures')
viz.plot_all()
```

## Fichiers générés

| Fichier | Contenu |
|---------|---------|
| `figures/fig1_distributions.png` | Distributions des variables numériques |
| `figures/fig2_taux_churn_cat.png` | Taux de churn par variable catégorielle |
| `figures/fig3_roc_curves.png` | Courbes ROC des 3 modèles comparés |
| `figures/fig4_confusion_matrix.png` | Matrice de confusion du meilleur modèle |
| `figures/fig5_feature_importance.png` | Importance des variables (RF/GB) |
| `figures/fig6_precision_recall.png` | Courbe précision-rappel |
| `figures/fig7_score_distribution.png` | Distribution des scores de probabilité |
| `figures/fig8_profil_churners.png` | Radar — profil moyen churner vs non-churner |

## Adapter à vos données

Pour utiliser ce pipeline sur un dataset réel, remplacez `bank_customers.csv` par votre fichier
en respectant les mêmes colonnes. Modifiez `FEATURES` et `TARGET` dans `src/churn_predictor.py`
si vos noms de colonnes diffèrent.

## Interpréter les résultats

- **AUC > 0.80** : bon pouvoir discriminant
- **Rappel churners > 0.70** : on détecte 7 churners sur 10
- **Faux négatifs** : churners manqués — analyser leurs caractéristiques pour améliorer le modèle
- **Feature importance** : les variables en haut du classement sont les plus prédictives du départ
