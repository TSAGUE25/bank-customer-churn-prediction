"""
Prédiction du Churn Bancaire — Machine Learning
Cas 11 — Portfolio Data Science | Emmanuel TSAGUE
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.churn_predictor import ChurnPredictor
from src.visualization import ChurnVisualizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'data_sample')
FIG_DIR   = os.path.join(BASE_DIR, 'figures')
REP_DIR   = os.path.join(BASE_DIR, 'reports')
os.makedirs(REP_DIR, exist_ok=True)

# ======================================================================
# 0. CHARGEMENT
# ======================================================================
df = pd.read_csv(os.path.join(DATA_DIR, 'bank_customers.csv'))
print(f"Dataset : {df.shape[0]} clients, {df.shape[1]} colonnes")
print(f"Taux de churn : {df['churn'].mean()*100:.1f}%")
print(f"Valeurs manquantes : {df.isnull().sum().sum()}")

# ======================================================================
# 1. EXPLORATION (EDA)
# ======================================================================
print("\n" + "="*60)
print("1. EXPLORATION")
print("="*60)
print("\nStatistiques descriptives :")
print(df.describe().round(1).to_string())
print("\nChurn par genre :")
print(df.groupby('genre')['churn'].mean().apply(lambda x: f"{x*100:.1f}%"))
print("\nChurn par pays :")
print(df.groupby('pays')['churn'].mean().apply(lambda x: f"{x*100:.1f}%"))
print("\nChurn par nb_produits :")
print(df.groupby('nb_produits')['churn'].mean().apply(lambda x: f"{x*100:.1f}%"))
print("\nChurn par membre_actif :")
print(df.groupby('membre_actif')['churn'].mean().apply(lambda x: f"{x*100:.1f}%"))

# ======================================================================
# 2. PROFIL DES CHURNERS
# ======================================================================
print("\n" + "="*60)
print("2. PROFIL DES CHURNERS")
print("="*60)
predictor = ChurnPredictor(df)
predictor.prepare()
profile = predictor.churn_profile()
print(f"\nComparaison moyennes Churners vs Non-churners :")
print(profile.to_string())

# ======================================================================
# 3. ENTRAINEMENT DES MODELES
# ======================================================================
print("\n" + "="*60)
print("3. ENTRAINEMENT DES 3 MODELES")
print("="*60)
results = predictor.train_all()
print(f"\nRésultats sur le jeu de test :")
print(f"{'Modèle':<25} {'AUC Test':>10} {'CV-AUC':>10} {'AP Score':>10} {'F1-Churn':>10}")
print("-" * 65)
for name, res in results.items():
    f1 = res['report']['1']['f1-score']
    print(f"{name:<25} {res['auc']:>10.3f} {res['cv_auc']:>10.3f} {res['ap']:>10.3f} {f1:>10.3f}")

best = predictor.best_model_name
print(f"\nMeilleur modèle : {best}")

# ======================================================================
# 4. METRIQUES DETAILLEES DU MEILLEUR MODELE
# ======================================================================
print("\n" + "="*60)
print(f"4. METRIQUES DETAILLEES — {best}")
print("="*60)
best_res = results[best]
print(f"\nMatrice de confusion :")
print(best_res['cm'])
print(f"\nRapport de classification :")
rep = best_res['report']
print(f"  Précision (churners)  : {rep['1']['precision']:.3f}")
print(f"  Rappel (churners)     : {rep['1']['recall']:.3f}")
print(f"  F1-score (churners)   : {rep['1']['f1-score']:.3f}")
print(f"  AUC ROC               : {best_res['auc']:.3f}")

# ======================================================================
# 5. IMPORTANCE DES VARIABLES
# ======================================================================
print("\n" + "="*60)
print("5. IMPORTANCE DES VARIABLES")
print("="*60)
importance = predictor.feature_importance()
if not importance.empty:
    print(f"\nTop 10 variables les plus prédictives :")
    print(importance.head(10).to_string(index=False))

# ======================================================================
# 6. ANALYSE DES FAUX NEGATIFS
# ======================================================================
print("\n" + "="*60)
print("6. FAUX NEGATIFS (churners manqués)")
print("="*60)
fn = predictor.false_negatives()
print(f"\n{len(fn)} churners non détectés sur le jeu de test :")
if not fn.empty:
    print(fn[['id_client','age','score_credit','solde','anciennete_annees','proba_churn']].head(10).to_string(index=False))

# ======================================================================
# 7. SYNTHESE
# ======================================================================
print("\n" + "="*60)
print("7. SYNTHESE")
print("="*60)
s = predictor.summary()
print(f"\nClients analysés    : {s['nb_clients']}")
print(f"Taux de churn       : {s['taux_churn']}%")
print(f"Nb churners         : {s['nb_churners']}")
print(f"Meilleur modèle     : {s['meilleur_modele']}")
print(f"AUC ROC (test)      : {s['auc_test']}")
print(f"AUC ROC (CV 5-fold) : {s['cv_auc']}")
print(f"Précision churners  : {s['precision_churners']}")
print(f"Rappel churners     : {s['recall_churners']}")
print(f"F1-score churners   : {s['f1_churners']}")

# ======================================================================
# 8. VISUALISATIONS
# ======================================================================
print("\n" + "="*60)
print("8. GENERATION DES FIGURES")
print("="*60)
viz = ChurnVisualizer(output_dir=FIG_DIR)
viz.plot_variable_distributions(df)
viz.plot_churn_rates(df)
viz.plot_roc_comparison(results, predictor._y_test)
viz.plot_confusion_matrix(best_res['cm'], best)
viz.plot_feature_importance(importance, best)
viz.plot_precision_recall(results, predictor._y_test)
viz.plot_proba_distribution(predictor._y_test.values, best_res['y_proba'], best)
viz.plot_churn_profile(profile)
print("\n8 figures générées dans figures/")

# ======================================================================
# 9. RAPPORT MARKDOWN
# ======================================================================
print("\n" + "="*60)
print("9. RAPPORT MARKDOWN")
print("="*60)

lines = [
    "# Rapport de Prédiction du Churn Bancaire",
    "",
    f"**Date :** 2024-12-05  ",
    f"**Dataset :** {s['nb_clients']} clients | Taux de churn : {s['taux_churn']}%  ",
    f"**Meilleur modèle :** {s['meilleur_modele']}",
    "",
    "> *Données simulées à des fins pédagogiques — aucune donnée réelle.*",
    "",
    "---",
    "",
    "## 1. Synthèse exécutive",
    "",
    "| Indicateur | Valeur |",
    "|-----------|--------|",
    f"| Clients analysés | **{s['nb_clients']}** |",
    f"| Taux de churn | **{s['taux_churn']}%** |",
    f"| Churners identifiés | **{s['nb_churners']}** |",
    f"| Meilleur modèle | **{s['meilleur_modele']}** |",
    f"| AUC ROC (test) | **{s['auc_test']}** |",
    f"| AUC ROC (CV 5-fold) | **{s['cv_auc']}** |",
    f"| F1-score churners | **{s['f1_churners']}** |",
    "",
    "---",
    "",
    "## 2. Comparaison des modèles",
    "",
    "| Modèle | AUC Test | CV-AUC | AP Score | F1-Churn |",
    "|--------|---------|--------|---------|---------|",
]
for name, res in results.items():
    f1 = res['report']['1']['f1-score']
    marker = " **Meilleur**" if name == best else ""
    lines.append(f"| {name}{marker} | {res['auc']:.3f} | {res['cv_auc']:.3f} | {res['ap']:.3f} | {f1:.3f} |")

lines += [
    "",
    "---",
    "",
    "## 3. Variables les plus prédictives",
    "",
    "| Rang | Variable | Importance | Interprétation |",
    "|------|---------|-----------|----------------|",
]
interpretations = {
    'age': 'Les clients plus âgés churne davantage',
    'solde': 'Solde nul = signal fort de départ',
    'nb_produits': '3+ produits = très fort risque',
    'score_credit': 'Score bas corrélé au churn',
    'membre_actif': 'Inactifs = risque x2',
    'anciennete_annees': 'Faible ancienneté = vulnérabilité',
    'salaire_estime': 'Salaire faible légèrement corrélé',
    'genre_enc': 'Femmes churne légèrement plus',
    'pays_enc': 'Clients allemands sur-représentés',
    'carte_credit': 'Carte = légèrement protecteur',
}
if not importance.empty:
    for i, (_, row) in enumerate(importance.head(10).iterrows(), 1):
        interp = interpretations.get(row['feature'], '')
        lines.append(f"| {i} | {row['feature']} | {row['importance']:.3f} | {interp} |")

lines += [
    "",
    "---",
    "",
    "## 4. Recommandations métier",
    "",
    "### Actions immédiates",
    "1. **Alerte précoce** : contacter les clients avec proba_churn > 0.70 dans les 30 jours",
    "2. **Clients à solde nul** : proposer un produit d'épargne adapté (signal le plus fort)",
    "3. **Membres inactifs** : programme de réengagement (offre personnalisée, conseiller dédié)",
    "",
    "### Actions moyen terme",
    "4. **Multi-produit** : réduire le churn en proposant un 2e produit aux clients mono-produit",
    "5. **Scoring automatique** : déployer le modèle en production, scorer hebdomadairement",
    "6. **A/B test** : mesurer l'effet des actions de rétention par segment de risque",
    "",
    "### ROI estimé",
    "Si 20% des churners prédits sont retenus grâce à une offre ciblée (coût moyen : 50€/client) :",
    f"- Churners retenus : ~{int(s['nb_churners']*0.20)} clients",
    "- Valeur annuelle préservée : ~1 500 €/client × clients retenus",
    "",
    "---",
    "",
    "*Rapport généré par Bank Customer Churn Prediction Framework*  ",
    "*Auteur : Emmanuel TSAGUE — Data Scientist / Data Analyst*",
]

report_path = os.path.join(REP_DIR, 'churn_report_sample.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"Rapport : {report_path}")
print("\n=== ANALYSE TERMINÉE ===")
