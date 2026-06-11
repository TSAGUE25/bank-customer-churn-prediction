# Méthodologie — Bank Customer Churn Prediction

## 1. Déséquilibre des classes

Le churn est un problème déséquilibré : en banque, le taux réel est de 5–15%. Ce dataset simule ~33% pour faciliter l'apprentissage. En production, des techniques de rééquilibrage seraient nécessaires :
- **Oversampling** : SMOTE (Synthetic Minority Over-sampling Technique)
- **Undersampling** : RandomUnderSampler
- **Ajustement du poids** : `class_weight='balanced'` dans sklearn
- **Changement de seuil** : seuil < 0.50 pour augmenter le rappel sur la classe minoritaire

## 2. Métriques choisies

Pour un problème de churn, le **rappel** (recall) sur la classe 1 est plus important que la précision :
- Un faux négatif (churner non détecté) = client perdu = coût maximal
- Un faux positif (non-churner ciblé) = coût de la campagne de rétention seulement

L'**AUC ROC** est la métrique principale car elle est insensible au seuil de décision et au déséquilibre.

## 3. Validation croisée stratifiée (StratifiedKFold, k=5)

La stratification garantit que chaque fold contient la même proportion de churners (essentiel avec des classes déséquilibrées). Le CV-AUC moyen est plus fiable que l'AUC sur le jeu de test seul.

## 4. Pipeline sklearn

Le pipeline combine StandardScaler + classifier en un seul objet :
- Empêche le data leakage (scaler fité uniquement sur le train)
- Simplifie la mise en production (un seul `predict()` sur les nouvelles données)
- Compatible avec `cross_val_score` et `GridSearchCV`

## 5. Seuil de décision

Le seuil par défaut est 0.50. En production, on peut l'ajuster selon le coût métier :
- Coût de rétention faible → seuil bas (0.30) → maximiser le rappel
- Coût de rétention élevé → seuil haut (0.65) → maximiser la précision
