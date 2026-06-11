# Schéma de référence — Bank Customer Churn Prediction

> Données entièrement simulées à des fins pédagogiques.

## Table : bank_customers.csv (150 clients)

| Colonne | Type | Description | Valeurs |
|---------|------|-------------|---------|
| `id_client` | string | Identifiant unique | CL0001–CL0150 |
| `age` | integer | Âge du client | 24–62 ans |
| `genre` | string | Genre | F, M |
| `pays` | string | Pays de résidence | France, Espagne |
| `score_credit` | integer | Score de solvabilité | 350–850 |
| `solde` | float | Solde du compte (€) | 0 à 250 000 |
| `nb_produits` | integer | Nombre de produits bancaires | 1, 2, 3, 4 |
| `carte_credit` | integer | Possession d'une carte crédit | 0=Non, 1=Oui |
| `membre_actif` | integer | Membre actif (connexions, transactions) | 0=Non, 1=Oui |
| `salaire_estime` | float | Salaire annuel estimé (€) | 10 000–200 000 |
| `anciennete_annees` | integer | Ancienneté en banque | 0–10 ans |
| `churn` | integer | **Variable cible** | 0=Resté, 1=Parti |

## Variable cible

**churn = 1** : le client a quitté la banque dans les 6 prochains mois (simulé).  
**churn = 0** : le client est resté.

**Taux de churn simulé : ~33%** (volontairement élevé pour l'exercice ; en réalité, les taux bancaires sont de 5–15%).

## Variables dérivées (encodage)

| Variable | Encodage | Valeurs |
|----------|---------|---------|
| `genre_enc` | LabelEncoder | F=0, M=1 |
| `pays_enc` | LabelEncoder | Espagne=0, France=1 |

## Signaux de churn les plus forts (basés sur la littérature CRM)

1. **Solde = 0** : le client a vidé son compte → signal fort de départ imminent
2. **Membre inactif** : pas de connexion ni de transaction depuis 3 mois
3. **Âge > 50 ans** : les clients seniors sont plus volatils dans les modèles bancaires
4. **Score crédit faible** : corrélé à une instabilité financière générale
5. **1 seul produit** : faible engagement envers la banque
