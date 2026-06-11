import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class ChurnPredictor:
    """Pipeline de prédiction du churn client bancaire."""

    FEATURES = [
        'age', 'score_credit', 'solde', 'nb_produits',
        'carte_credit', 'membre_actif', 'salaire_estime', 'anciennete_annees',
        'genre_enc', 'pays_enc'
    ]
    TARGET = 'churn'

    def __init__(self, df):
        self.raw = df.copy()
        self._prepared = None
        self.scaler = StandardScaler()
        self.le_genre = LabelEncoder()
        self.le_pays = LabelEncoder()
        self.models = {}
        self.results = {}
        self.best_model_name = None

    # ------------------------------------------------------------------ #
    # 1. Préparation des données
    # ------------------------------------------------------------------ #

    def prepare(self):
        """Encode les variables catégorielles et sépare X/y."""
        df = self.raw.copy()
        df['genre_enc'] = self.le_genre.fit_transform(df['genre'])
        df['pays_enc']  = self.le_pays.fit_transform(df['pays'])
        self._prepared = df
        return df

    def get_prepared(self):
        if self._prepared is None:
            self.prepare()
        return self._prepared

    def split(self, test_size=0.25, random_state=42):
        """Retourne X_train, X_test, y_train, y_test."""
        df = self.get_prepared()
        X = df[self.FEATURES]
        y = df[self.TARGET]
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state, stratify=y)

    # ------------------------------------------------------------------ #
    # 2. Entraînement des modèles
    # ------------------------------------------------------------------ #

    def train_all(self):
        """Entraîne Logistic Regression, Random Forest, Gradient Boosting."""
        X_train, X_test, y_train, y_test = self.split()
        self._X_train = X_train
        self._X_test  = X_test
        self._y_train = y_train
        self._y_test  = y_test

        candidates = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(random_state=42, max_iter=1000, C=0.5))
            ]),
            'Random Forest': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=200, max_depth=8,
                                               random_state=42, n_jobs=-1))
            ]),
            'Gradient Boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                   learning_rate=0.08, random_state=42))
            ]),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, pipeline in candidates.items():
            pipeline.fit(X_train, y_train)
            y_pred   = pipeline.predict(X_test)
            y_proba  = pipeline.predict_proba(X_test)[:, 1]
            cv_auc   = cross_val_score(pipeline, X_train, y_train,
                                       cv=cv, scoring='roc_auc').mean()
            self.models[name]  = pipeline
            self.results[name] = {
                'y_pred':  y_pred,
                'y_proba': y_proba,
                'auc':     roc_auc_score(y_test, y_proba),
                'cv_auc':  cv_auc,
                'report':  classification_report(y_test, y_pred, output_dict=True),
                'cm':      confusion_matrix(y_test, y_pred),
                'ap':      average_precision_score(y_test, y_proba),
            }

        # Sélection du meilleur modèle par AUC
        self.best_model_name = max(self.results, key=lambda k: self.results[k]['auc'])
        return self.results

    # ------------------------------------------------------------------ #
    # 3. Importance des variables
    # ------------------------------------------------------------------ #

    def feature_importance(self, model_name=None):
        """Retourne l'importance des features pour RF ou GB."""
        name = model_name or self.best_model_name
        if name not in self.models:
            return pd.DataFrame()
        clf = self.models[name].named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            imp = pd.DataFrame({
                'feature': self.FEATURES,
                'importance': clf.feature_importances_
            }).sort_values('importance', ascending=False)
            return imp
        elif hasattr(clf, 'coef_'):
            imp = pd.DataFrame({
                'feature': self.FEATURES,
                'importance': abs(clf.coef_[0])
            }).sort_values('importance', ascending=False)
            return imp
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # 4. Analyse des faux négatifs (clients churned non détectés)
    # ------------------------------------------------------------------ #

    def false_negatives(self):
        """Retourne les clients churned non détectés par le meilleur modèle."""
        name = self.best_model_name
        res  = self.results[name]
        idx  = self._X_test.index
        mask = (self._y_test.values == 1) & (res['y_pred'] == 0)
        fn   = self.raw.loc[idx[mask]].copy()
        fn['proba_churn'] = res['y_proba'][mask]
        return fn

    # ------------------------------------------------------------------ #
    # 5. Profil des clients churners
    # ------------------------------------------------------------------ #

    def churn_profile(self):
        """Compare les caractéristiques moyennes churners vs non-churners."""
        df = self.get_prepared()
        num_cols = ['age', 'score_credit', 'solde', 'nb_produits',
                    'salaire_estime', 'anciennete_annees']
        profile = df.groupby('churn')[num_cols].mean().T
        profile.columns = ['Non-churners', 'Churners']
        profile['diff_pct'] = (
            (profile['Churners'] - profile['Non-churners'])
            / profile['Non-churners'] * 100
        ).round(1)
        return profile.round(1)

    # ------------------------------------------------------------------ #
    # 6. Synthèse
    # ------------------------------------------------------------------ #

    def summary(self):
        df = self.raw
        best = self.results.get(self.best_model_name, {})
        report = best.get('report', {})
        return {
            'nb_clients': len(df),
            'taux_churn': round(df['churn'].mean() * 100, 1),
            'nb_churners': int(df['churn'].sum()),
            'meilleur_modele': self.best_model_name,
            'auc_test': round(best.get('auc', 0), 3),
            'cv_auc': round(best.get('cv_auc', 0), 3),
            'precision_churners': round(report.get('1', {}).get('precision', 0), 3),
            'recall_churners': round(report.get('1', {}).get('recall', 0), 3),
            'f1_churners': round(report.get('1', {}).get('f1-score', 0), 3),
            'ap_score': round(best.get('ap', 0), 3),
        }
