# -*- coding: utf-8 -*-
"""XGBoost + optimisation de seuil + SHAP pour la prédiction du churn bancaire."""

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, f1_score,
                              precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les colonnes catégorielles (pays, sexe) en entiers."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def build_xgb_pipeline(scale_pos_weight: float = 5.25) -> Pipeline:
    """XGBoostClassifier avec class_weight implicite via scale_pos_weight."""
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc", random_state=42, n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("XGBoost non installé — utilisation de GradientBoostingClassifier")
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    return Pipeline([("scaler", StandardScaler()), ("model", clf)])


def find_optimal_threshold(y_test, y_proba,
                            beta: float = 1.5) -> float:
    """Trouve le seuil maximisant le F-beta score (beta>1 favorise le recall)."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-9)
    best_idx = np.argmax(fbeta[:-1])
    best_thr = thresholds[best_idx]
    print(f"Seuil optimal (F{beta}) : {best_thr:.3f} | "
          f"Précision={precision[best_idx]:.3f} | Recall={recall[best_idx]:.3f}")
    return float(best_thr)


def evaluate_churn_model(pipeline, X_test, y_test,
                          cost_retention: float = 50,
                          cost_lost_client: float = 500) -> dict:
    """Évalue le modèle de churn : AUC, F1, seuil optimal, coût métier."""
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    best_thr = find_optimal_threshold(y_test, y_proba, beta=1.5)
    y_pred  = (y_proba >= best_thr).astype(int)

    from sklearn.metrics import confusion_matrix
    cm      = confusion_matrix(y_test, y_pred)
    n_fp    = cm[0, 1]   # non-churners contactés inutilement
    n_fn    = cm[1, 0]   # churners manqués
    cout    = n_fp * cost_retention + n_fn * cost_lost_client

    print(f"\nROC-AUC = {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Reste", "Churn"]))
    print(f"Coût estimé : {cout:,.0f} € (FP×{cost_retention}€ + FN×{cost_lost_client}€)")
    return {"roc_auc": auc, "best_threshold": best_thr, "cout_eur": cout}


def plot_shap_churn(model_step, X_test: pd.DataFrame, n_samples: int = 500) -> None:
    """SHAP beeswarm + force plot pour le modèle de churn."""
    try:
        import shap
        X_s = X_test.sample(n=min(n_samples, len(X_test)), random_state=42)
        explainer  = shap.TreeExplainer(model_step)
        shap_vals  = explainer.shap_values(X_s)
        sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        shap.summary_plot(sv, X_s, max_display=10, show=True)
    except ImportError:
        print("shap non installé — pip install shap")
