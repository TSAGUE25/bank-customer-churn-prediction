# Rapport de Prédiction du Churn Bancaire
**Date :** 2024-12-15 | **Dataset :** bank_customers.csv | **Clients :** 150

---

## 1. Résumé exécutif

| Indicateur | Valeur |
|-----------|--------|
| Clients analysés | 150 |
| Taux de churn observé | 33.3% (50 clients) |
| Meilleur modèle | Random Forest |
| AUC ROC (test) | **0.81** |
| CV-AUC (5-fold) | **0.78** |
| F1-score churners | 0.65 |
| Rappel churners | 0.68 |
| Précision churners | 0.62 |

---

## 2. Comparaison des modèles

| Modèle | AUC Test | CV-AUC | AP Score | F1 | Rappel | Précision |
|--------|---------|--------|---------|-----|--------|---------|
| Logistic Regression | 0.72 | 0.70 | 0.55 | 0.58 | 0.60 | 0.56 |
| **Random Forest** | **0.81** | **0.78** | **0.68** | **0.65** | **0.68** | **0.62** |
| Gradient Boosting | 0.79 | 0.76 | 0.65 | 0.62 | 0.64 | 0.60 |

**Modèle sélectionné :** Random Forest (meilleur AUC test et CV-AUC)

---

## 3. Variables les plus prédictives

| Rang | Variable | Importance (%) | Interprétation |
|------|---------|---------------|----------------|
| 1 | `age` | 18.4% | Clients > 50 ans : risque churn x2.1 |
| 2 | `solde` | 15.7% | Solde nul = signal fort de départ imminent |
| 3 | `nb_produits` | 12.3% | 1 seul produit = engagement faible |
| 4 | `score_credit` | 11.8% | Score < 500 corrélé au churn |
| 5 | `membre_actif` | 10.9% | Inactif = risque x1.8 vs actif |
| 6 | `anciennete_annees` | 9.2% | Clients < 2 ans : taux abandon 42% |
| 7 | `salaire_estime` | 8.6% | Faible impact relatif |
| 8 | `pays_enc` | 7.1% | Espagne : taux churn légèrement supérieur |
| 9 | `genre_enc` | 3.5% | Femmes churne légèrement plus (+4 pts) |
| 10 | `carte_credit` | 2.5% | Impact marginal |

---

## 4. Profil type du churner

| Caractéristique | Churners (moy.) | Non-churners (moy.) | Écart |
|----------------|----------------|--------------------|----|
| Age | 44.7 ans | 37.2 ans | +7.5 ans |
| Solde | 91 200 € | 76 400 € | +19% |
| Score crédit | 578 | 652 | -74 pts |
| Nb produits | 1.4 | 1.6 | -0.2 |
| Membre actif | 38% | 52% | -14 pts |
| Ancienneté | 4.8 ans | 5.1 ans | -0.3 ans |

---

## 5. Analyse des faux négatifs (churners non détectés)

Sur 50 churners réels, le modèle en manque ~16 (rappel = 0.68).

**Profil des churners non détectés :**
- Score de crédit plus élevé que la moyenne des churners (≥ 620)
- Solde non nul — départ progressif, non brutal
- Membre actif malgré l'intention de partir
- Ancienneté intermédiaire (3–6 ans)

**Recommandation :** Enrichir les features avec des données comportementales (fréquence de connexion, volume de transactions sur 90 jours) pour capturer ces profils.

---

## 6. Recommandations d'action

### Clients à risque élevé (proba > 0.70)

| Action | Coût estimé | ROI estimé |
|--------|------------|-----------|
| Appel conseiller personnalisé | 25 €/client | 500–1 500 €/client retenu |
| Offre taux préférentiel 6 mois | 80 €/client | 800–2 000 €/client retenu |
| Programme fidélité VIP | 50 €/client | 600–1 200 €/client retenu |

### Clients à risque modéré (proba 0.40–0.70)

| Action | Coût estimé |
|--------|------------|
| Email personnalisé + offre produit | 5 €/client |
| Notification app mobile | 2 €/client |

### Clients à faible risque (proba < 0.40)

- Aucune action de rétention nécessaire
- Campagnes upsell/cross-sell à privilégier

---

## 7. ROI de la campagne de rétention (simulation)

| Paramètre | Valeur |
|-----------|--------|
| Churners détectés (rappel 0.68) | ~34 clients |
| Clients ciblés (proba > 0.50) | ~40 clients |
| Taux de succès rétention estimé | 25% |
| Clients retenus | ~10 clients |
| Revenu moyen annuel / client | 1 500 € |
| Gain préservé | **15 000 €/an** |
| Coût campagne (40 × 35 €) | 1 400 € |
| **ROI** | **+971%** |

---

## 8. Limites et prochaines étapes

| Limite | Action recommandée |
|--------|-------------------|
| 150 clients seulement | Déployer sur 10 000+ clients réels |
| Déséquilibre non traité (33% churn) | Ajouter SMOTE ou class_weight='balanced' |
| Pas d'explication individuelle | Intégrer SHAP pour expliquer chaque prédiction |
| Features statiques uniquement | Ajouter features rolling (30j, 90j) |
| Seuil fixe à 0.50 | Optimiser par courbe coût-bénéfice métier |

---

*Rapport généré automatiquement — données entièrement simulées*
*Auteur : Emmanuel TSAGUE | Formation DataScientest*
