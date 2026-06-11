# Dictionnaire des données — Bank Customer Churn Prediction

> Données entièrement simulées. Aucune donnée réelle ou confidentielle.

## Fichier : `bank_customers.csv`

| Colonne | Type | Valeurs | Description |
|---------|------|---------|-------------|
| `id_client` | TEXT | CL0001–CL0150 | Identifiant unique fictif |
| `age` | INT | 18–72 | Âge du client en années |
| `genre` | TEXT | F / M | Genre déclaré |
| `pays` | TEXT | France / Espagne | Pays de résidence |
| `score_credit` | INT | 350–850 | Score de solvabilité (type FICO) |
| `solde` | FLOAT | 0–250 000 | Solde du compte courant (€) |
| `nb_produits` | INT | 1–2 | Nombre de produits bancaires souscrits |
| `carte_credit` | INT | 0 / 1 | Possession d'une carte de crédit |
| `membre_actif` | INT | 0 / 1 | Activité récente (connexion, transaction) |
| `salaire_estime` | FLOAT | 11 000–199 000 | Salaire annuel estimé (€) |
| `anciennete_annees` | INT | 0–10 | Ancienneté en tant que client |
| `churn` | INT | 0 / 1 | **Cible** : 1 = client parti, 0 = client resté |

## Variables dérivées (créées dans le code)

| Variable | Origine | Description |
|----------|---------|-------------|
| `genre_enc` | LabelEncoder(genre) | F=0, M=1 |
| `pays_enc` | LabelEncoder(pays) | Espagne=0, France=1 |

## Distribution de la variable cible

| Classe | Valeur | Effectif approx. | % |
|--------|--------|-----------------|---|
| Non-churn | 0 | ~100 | ~67% |
| Churn | 1 | ~50 | ~33% |

> Le taux de churn simulé (33%) est volontairement plus élevé que la réalité bancaire (5–15%)
> pour faciliter l'apprentissage du modèle sur ce petit dataset de 150 clients.

## Signaux de churn identifiés dans les données

| Signal | Variable | Interprétation |
|--------|---------|----------------|
| Solde nul | `solde = 0` | Client qui vide son compte avant de partir |
| Inactivité | `membre_actif = 0` | Absence d'opérations récentes |
| Produit unique | `nb_produits = 1` | Faible engagement envers la banque |
| Age > 50 ans | `age` | Segment plus enclin à changer de banque |
| Score crédit faible | `score_credit < 500` | Relation fragilisée avec l'établissement |
