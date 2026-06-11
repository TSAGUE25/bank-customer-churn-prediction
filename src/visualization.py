import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import roc_curve, precision_recall_curve


class ChurnVisualizer:
    """Visualisations pour la prédiction du churn bancaire."""

    COLORS = {'Logistic Regression': '#2196F3', 'Random Forest': '#4CAF50',
               'Gradient Boosting': '#FF9800'}

    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.rcParams.update({'figure.dpi': 120, 'font.family': 'DejaVu Sans'})

    def _save(self, fig, filename):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure sauvegardée : {path}")
        return path

    # ------------------------------------------------------------------ #
    # Figure 1 : Distribution des variables clés (Churners vs Non)
    # ------------------------------------------------------------------ #

    def plot_variable_distributions(self, df):
        cols = ['age', 'score_credit', 'solde', 'anciennete_annees']
        labels = ['Âge', 'Score crédit', 'Solde (€)', 'Ancienneté (ans)']
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, col, lab in zip(axes.ravel(), cols, labels):
            churners = df[df['churn'] == 1][col]
            non_churners = df[df['churn'] == 0][col]
            ax.hist(non_churners, bins=20, alpha=0.6, color='#4CAF50',
                    label='Non-churner', density=True)
            ax.hist(churners, bins=20, alpha=0.6, color='#f44336',
                    label='Churner', density=True)
            ax.set_title(lab, fontsize=12, fontweight='bold')
            ax.set_ylabel('Densité', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        fig.suptitle('Distribution des variables clés — Churners vs Non-churners',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, 'fig1_distributions.png')

    # ------------------------------------------------------------------ #
    # Figure 2 : Taux de churn par variable catégorielle
    # ------------------------------------------------------------------ #

    def plot_churn_rates(self, df):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, col, title in zip(axes,
                                   ['genre', 'pays', 'nb_produits'],
                                   ['Taux churn par Genre', 'Taux churn par Pays',
                                    'Taux churn par Nb produits']):
            rates = df.groupby(col)['churn'].mean() * 100
            colors = ['#f44336' if r > df['churn'].mean() * 100 else '#4CAF50'
                      for r in rates]
            bars = ax.bar(rates.index.astype(str), rates.values, color=colors,
                          edgecolor='white', width=0.5)
            ax.axhline(df['churn'].mean() * 100, color='navy', linestyle='--',
                       linewidth=1.5, label=f"Moy. {df['churn'].mean()*100:.1f}%")
            for bar, val in zip(bars, rates.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Taux de churn (%)', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        fig.suptitle('Taux de churn par variable catégorielle',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, 'fig2_taux_churn_cat.png')

    # ------------------------------------------------------------------ #
    # Figure 3 : Comparaison AUC des 3 modèles
    # ------------------------------------------------------------------ #

    def plot_roc_comparison(self, results, y_test):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire (AUC=0.50)')
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
            ax.plot(fpr, tpr, linewidth=2,
                    color=self.COLORS.get(name, '#999'),
                    label=f"{name} (AUC={res['auc']:.3f})")
        ax.set_xlabel('Taux de faux positifs (FPR)', fontsize=12)
        ax.set_ylabel('Taux de vrais positifs (TPR)', fontsize=12)
        ax.set_title('Courbes ROC — Comparaison des modèles', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(alpha=0.3)
        return self._save(fig, 'fig3_roc_curves.png')

    # ------------------------------------------------------------------ #
    # Figure 4 : Matrice de confusion du meilleur modèle
    # ------------------------------------------------------------------ #

    def plot_confusion_matrix(self, cm, model_name):
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = [['TN\n(non-churners\ncorrects)', 'FP\n(fausses alarmes)'],
                  ['FN\n(churners\nmanqués)', 'TP\n(churners\ndétectés)']]
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                    xticklabels=['Prédit: 0', 'Prédit: 1'],
                    yticklabels=['Réel: 0', 'Réel: 1'], ax=ax,
                    linewidths=0.5, linecolor='white')
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.4, str(cm[i, j]),
                        ha='center', va='center', fontsize=22, fontweight='bold',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
                ax.text(j + 0.5, i + 0.65, labels[i][j],
                        ha='center', va='center', fontsize=8,
                        color='white' if cm[i, j] > cm.max()/2 else 'gray')
        ax.set_title(f'Matrice de confusion — {model_name}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, 'fig4_confusion_matrix.png')

    # ------------------------------------------------------------------ #
    # Figure 5 : Importance des features
    # ------------------------------------------------------------------ #

    def plot_feature_importance(self, imp_df, model_name):
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ['#f44336' if i < 3 else '#FF9800' if i < 6 else '#4CAF50'
                  for i in range(len(imp_df))]
        bars = ax.barh(imp_df['feature'], imp_df['importance'],
                       color=colors, edgecolor='white', height=0.6)
        for bar, val in zip(bars, imp_df['importance']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Importance des variables — {model_name}',
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        patches = [mpatches.Patch(color='#f44336', label='Top 3 variables'),
                   mpatches.Patch(color='#FF9800', label='Variables moyennes'),
                   mpatches.Patch(color='#4CAF50', label='Variables secondaires')]
        ax.legend(handles=patches, fontsize=9, loc='lower right')
        plt.tight_layout()
        return self._save(fig, 'fig5_feature_importance.png')

    # ------------------------------------------------------------------ #
    # Figure 6 : Courbe Précision-Rappel
    # ------------------------------------------------------------------ #

    def plot_precision_recall(self, results, y_test):
        fig, ax = plt.subplots(figsize=(9, 6))
        baseline = y_test.mean()
        ax.axhline(baseline, color='k', linestyle='--', linewidth=1,
                   label=f'Baseline (AP={baseline:.2f})')
        for name, res in results.items():
            prec, rec, _ = precision_recall_curve(y_test, res['y_proba'])
            ax.plot(rec, prec, linewidth=2,
                    color=self.COLORS.get(name, '#999'),
                    label=f"{name} (AP={res['ap']:.3f})")
        ax.set_xlabel('Rappel (Recall)', fontsize=12)
        ax.set_ylabel('Précision', fontsize=12)
        ax.set_title('Courbes Précision-Rappel', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        return self._save(fig, 'fig6_precision_recall.png')

    # ------------------------------------------------------------------ #
    # Figure 7 : Score de probabilité de churn (histogram)
    # ------------------------------------------------------------------ #

    def plot_proba_distribution(self, y_test, y_proba, model_name):
        fig, ax = plt.subplots(figsize=(9, 5))
        churners = y_proba[y_test == 1]
        non_churners = y_proba[y_test == 0]
        ax.hist(non_churners, bins=20, alpha=0.65, color='#4CAF50',
                label='Non-churners', density=True)
        ax.hist(churners, bins=20, alpha=0.65, color='#f44336',
                label='Churners réels', density=True)
        ax.axvline(0.5, color='navy', linestyle='--', linewidth=2, label='Seuil 0.50')
        ax.set_xlabel('Probabilité prédite de churn', fontsize=12)
        ax.set_ylabel('Densité', fontsize=12)
        ax.set_title(f'Distribution des scores de churn — {model_name}',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        return self._save(fig, 'fig7_score_distribution.png')

    # ------------------------------------------------------------------ #
    # Figure 8 : Profil comparatif Churners vs Non-churners
    # ------------------------------------------------------------------ #

    def plot_churn_profile(self, profile_df):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(profile_df))
        w = 0.35
        bars1 = ax.bar(x - w/2, profile_df['Non-churners'], width=w,
                       label='Non-churners', color='#4CAF50', alpha=0.85)
        bars2 = ax.bar(x + w/2, profile_df['Churners'], width=w,
                       label='Churners', color='#f44336', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(profile_df.index, rotation=20, ha='right', fontsize=10)
        ax.set_title('Profil moyen — Churners vs Non-churners',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        for bars, vals in [(bars1, profile_df['Non-churners']),
                           (bars2, profile_df['Churners'])]:
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(profile_df.max(axis=1)) * 0.01,
                        f'{val:,.0f}', ha='center', fontsize=8)
        plt.tight_layout()
        return self._save(fig, 'fig8_profil_churners.png')
