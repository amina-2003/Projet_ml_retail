"""
train_model.py — Entraînement, évaluation et optimisation des modèles
Étapes :
  1. Chargement des données train/test
  2. Entraînement de 3 modèles supervisés (Logistic Regression, Random Forest, XGBoost)
  3. Analyse KMeans (clustering non supervisé)
  4. Évaluation complète (Accuracy, Precision, Recall, F1, AUC-ROC)
  5. Gestion du déséquilibre (class_weight)
  6. Optimisation ÉTAPE 1 : GridSearchCV  — exploration large, grille fixe
  7. Optimisation ÉTAPE 2 : Optuna        — affinement précis, recherche intelligente
  8. Importance des features
  9. Sauvegarde du meilleur modèle

"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Silencieux sauf erreurs
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
)

from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.utils import load_train_test


# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

RANDOM_STATE = 42
MODELS_DIR = 'models'
CV_FOLDS = 5   # Nombre de folds pour la validation croisée


# ─────────────────────────────────────────────
# ÉTAPE 1 — DÉFINITION DES MODÈLES
# ─────────────────────────────────────────────

def get_base_models() -> dict:
    """
    Retourne un dictionnaire des 3 modèles de base à comparer.

    Pourquoi ces 3 modèles ?
    - LogisticRegression : baseline rapide, interprétable, sensible à l'échelle
    - RandomForest       : robuste, gère les non-linéarités, résistant aux outliers
    - XGBoost            : souvent le meilleur sur données tabulaires, gère les valeurs manquantes
    """
    return {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced',  # Compense le déséquilibre Churn=0/1
            solver='lbfgs',
            C=1.0
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            # scale_pos_weight compense le déséquilibre : n_négatifs / n_positifs
            scale_pos_weight=2,
            use_label_encoder=False,
            verbosity=0
        ),
    }


# ─────────────────────────────────────────────
# ÉTAPE 2 — ENTRAÎNEMENT D'UN SEUL MODÈLE
# ─────────────────────────────────────────────

def train_and_evaluate(model, model_name: str,
                       X_train, y_train,
                       X_test, y_test) -> dict:
    """
    Entraîne un modèle et retourne un dictionnaire de métriques complètes.

    Métriques retournées :
    - accuracy  : proportion de bonnes prédictions (trompeuse si déséquilibre)
    - precision : parmi ceux prédits churners, combien le sont vraiment ?
    - recall    : parmi les vrais churners, combien a-t-on détectés ? (priorité ici)
    - f1        : moyenne harmonique precision/recall
    - roc_auc   : capacité globale à distinguer les deux classes (0.5=aléatoire, 1=parfait)
    - cv_f1     : F1 moyen en validation croisée (plus fiable que le score sur test seul)
    """
    print(f"\n  🏋️  Entraînement : {model_name}...")

    # Entraînement
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    # Métriques
    metrics = {
        'model':     model,
        'name':      model_name,
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc':   round(roc_auc_score(y_test, y_prob), 4),
    }

    # Validation croisée (5-fold stratifiée) sur les données d'entraînement
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    metrics['cv_f1_mean'] = round(cv_scores.mean(), 4)
    metrics['cv_f1_std']  = round(cv_scores.std(), 4)

    # Affichage
    print(f"     Accuracy : {metrics['accuracy']:.4f}")
    print(f"     Precision: {metrics['precision']:.4f}")
    print(f"     Recall   : {metrics['recall']:.4f}  ← priorité Churn")
    print(f"     F1-Score : {metrics['f1']:.4f}")
    print(f"     AUC-ROC  : {metrics['roc_auc']:.4f}")
    print(f"     CV F1    : {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n     Matrice de confusion :")
    print(f"                  Prédit 0   Prédit 1")
    print(f"     Réel 0  :  {cm[0,0]:>8}   {cm[0,1]:>8}")
    print(f"     Réel 1  :  {cm[1,0]:>8}   {cm[1,1]:>8}")
    tn, fp, fn, tp = cm.ravel()
    print(f"     → Vrais Positifs (churners détectés) : {tp}")
    print(f"     → Faux Négatifs (churners manqués)   : {fn}  ← à minimiser !")

    return metrics


# ─────────────────────────────────────────────
# ÉTAPE 3 — COMPARAISON DE TOUS LES MODÈLES
# ─────────────────────────────────────────────

def compare_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    Entraîne et évalue les 3 modèles de base.
    Affiche un tableau comparatif et retourne le nom du meilleur modèle (par F1).
    """
    print("\n" + "=" * 60)
    print("🏆 COMPARAISON DES MODÈLES")
    print("=" * 60)

    models = get_base_models()
    results = []

    for name, model in models.items():
        metrics = train_and_evaluate(model, name, X_train, y_train, X_test, y_test)
        results.append(metrics)

    # Tableau comparatif
    df_results = pd.DataFrame([{
        'Modèle':     r['name'],
        'Accuracy':   r['accuracy'],
        'Precision':  r['precision'],
        'Recall':     r['recall'],
        'F1-Score':   r['f1'],
        'AUC-ROC':    r['roc_auc'],
        'CV F1 (mean)': r['cv_f1_mean'],
        'CV F1 (std)':  r['cv_f1_std'],
    } for r in results])

    print("\n\n📊 TABLEAU COMPARATIF :")
    print(df_results.to_string(index=False))

    # Meilleur modèle selon F1 (métrique principale pour le Churn déséquilibré)
    best_idx = df_results['F1-Score'].idxmax()
    best_name = df_results.loc[best_idx, 'Modèle']
    print(f"\n🥇 Meilleur modèle (F1) : {best_name}")

    # Sauvegarder tous les modèles
    os.makedirs(MODELS_DIR, exist_ok=True)
    for r in results:
        path = os.path.join(MODELS_DIR, f"{r['name']}.pkl")
        joblib.dump(r['model'], path)
        print(f"   💾 {r['name']} sauvegardé → {path}")

    return df_results, results


# ─────────────────────────────────────────────
# ÉTAPE 3.5 — ANALYSE KMEANS (NON SUPERVISÉ)
# ─────────────────────────────────────────────

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KMEANS_FEATURES = [
    'Frequency',
    'MonetaryTotal',
    'AvgDaysBetweenPurchases',
    'SatisfactionScore',
    'SupportTicketsCount',
    'AccountAgeDays',
    'ReturnRatio',
    'EngagementScore',
]

def find_optimal_k(X_km: pd.DataFrame, k_range: range = range(2, 9)) -> int:
    """
    Trouve le nombre optimal de clusters k via Silhouette Score.
    Reçoit déjà les données filtrées + standardisées.
    """
    print("\n  🔎 Recherche du nombre optimal de clusters (k)...")
    print(f"     {'k':<6} {'Inertie':>12} {'Silhouette':>12}")
    print("     " + "-" * 32)

    inertias    = []
    silhouettes = []

    X_sample = X_km.sample(min(2000, len(X_km)), random_state=RANDOM_STATE)

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_sample)

        inertia   = round(km.inertia_, 0)
        sil_score = round(silhouette_score(X_sample, labels,
                                           sample_size=1000,
                                           random_state=RANDOM_STATE), 4)
        inertias.append(inertia)
        silhouettes.append(sil_score)
        print(f"     k={k:<4} {inertia:>12.0f} {sil_score:>12.4f}")

    best_k = list(k_range)[silhouettes.index(max(silhouettes))]
    print(f"\n     ✅ k optimal (meilleur silhouette) : k = {best_k}")
    return best_k


def run_kmeans_analysis(X_train: pd.DataFrame,
                        X_test:  pd.DataFrame,
                        y_train: pd.Series,
                        y_test:  pd.Series,
                        k: int = None) -> dict:

    print(f"\n{'=' * 60}")
    print("🔵 ANALYSE KMEANS (Clustering Non Supervisé)")
    print("=" * 60)
    print("""
  ℹ️  KMeans ≠ classification supervisée.
  Il regroupe les clients en segments SANS connaître le Churn.
  On analyse ensuite quel segment correspond aux clients qui churent.
""")

    # ── Vérification des features disponibles ──
    available = [f for f in KMEANS_FEATURES if f in X_train.columns]
    missing   = [f for f in KMEANS_FEATURES if f not in X_train.columns]

    if missing:
        print(f"  ⚠️  Features absentes (ignorées) : {missing}")
    if len(available) < 3:
        print("  ❌ Pas assez de features KMeans disponibles — abandon")
        return None

    print(f"  → Features utilisées : {available}")

    # ── Affichage stats pour détecter les features dominantes ──
    print("\n  📊 Stats des features KMeans (avant standardisation) :")
    stats = X_train[available].describe().loc[['mean', 'std', 'min', 'max']].round(3)
    print(stats.to_string())

    # ── Standardisation interne au KMeans ──
    # Indépendante du RobustScaler global — chaque feature contribue équitablement
    scaler_km = StandardScaler()
    X_km_train = pd.DataFrame(
        scaler_km.fit_transform(X_train[available]),
        columns=available,
        index=X_train.index
    )
    X_km_test = pd.DataFrame(
        scaler_km.transform(X_test[available]),
        columns=available,
        index=X_test.index
    )

    # ── Partie A : Trouver k optimal ──
    if k is None:
        k = find_optimal_k(X_km_train)

    print(f"\n  🏋️  Entraînement KMeans avec k={k} clusters...")

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    train_clusters = kmeans.fit_predict(X_km_train)
    test_clusters  = kmeans.predict(X_km_test)

    # ── Métriques KMeans ──
    inertia   = round(kmeans.inertia_, 2)
    sil_train = round(silhouette_score(
        X_km_train, train_clusters,
        sample_size=min(2000, len(X_km_train)),
        random_state=RANDOM_STATE
    ), 4)

    print(f"     Inertie          : {inertia}")
    print(f"     Silhouette Score : {sil_train}  (0=mauvais, 1=parfait)")

    # ── Partie B : Analyser la relation clusters ↔ Churn ──
    print(f"\n  📊 Analyse des clusters vs Churn :")
    print(f"     {'Cluster':<10} {'N clients':>10} {'Taux Churn':>12} "
          f"{'Moy Frequency':>14} {'Moy MonetaryTotal':>18} {'Moy SatisfactionScore':>21}")
    print("     " + "-" * 88)

    df_analysis = X_train[available].copy()
    df_analysis['KMeans_Cluster'] = train_clusters
    df_analysis['Churn']          = y_train.values

    cluster_stats = []
    for cluster_id in sorted(df_analysis['KMeans_Cluster'].unique()):
        subset     = df_analysis[df_analysis['KMeans_Cluster'] == cluster_id]
        n          = len(subset)
        churn_rate = round(subset['Churn'].mean() * 100, 1)

        # Profil basé sur Frequency + SatisfactionScore
        avg_freq = subset['Frequency'].mean() if 'Frequency' in subset.columns else None
        avg_sat  = subset['SatisfactionScore'].mean() if 'SatisfactionScore' in subset.columns else None
        avg_mon  = subset['MonetaryTotal'].mean() if 'MonetaryTotal' in subset.columns else None

        if avg_freq is not None and avg_sat is not None:
            if avg_freq > df_analysis['Frequency'].quantile(0.66) and avg_sat > 5:
                profil = "Actif & satisfait"
            elif avg_freq < df_analysis['Frequency'].quantile(0.33):
                profil = "Inactif (risque fort)"
            elif avg_sat < 5:
                profil = "Insatisfait"
            else:
                profil = "Occasionnel"
        else:
            profil = "—"

        cluster_stats.append({
            'cluster': cluster_id, 'n': n,
            'churn_rate': churn_rate, 'profil': profil,
            'avg_frequency': round(avg_freq, 2) if avg_freq else None,
            'avg_monetary':  round(avg_mon, 2)  if avg_mon  else None,
            'avg_sat':       round(avg_sat, 2)  if avg_sat  else None,
        })

        freq_str = f"{avg_freq:.2f}" if avg_freq is not None else "—"
        mon_str  = f"{avg_mon:.2f}"  if avg_mon  is not None else "—"
        sat_str  = f"{avg_sat:.2f}"  if avg_sat  is not None else "—"

        print(f"     Cluster {cluster_id:<3} {n:>10} {churn_rate:>11.1f}%  "
              f"{freq_str:>14} {mon_str:>18} {sat_str:>21}  [{profil}]")

    riskiest = max(cluster_stats, key=lambda x: x['churn_rate'])
    safest   = min(cluster_stats, key=lambda x: x['churn_rate'])
    print(f"\n     ⚠️  Cluster le plus à risque  : Cluster {riskiest['cluster']} "
          f"({riskiest['churn_rate']}% de churn) — {riskiest['profil']}")
    print(f"     ✅ Cluster le plus fidèle    : Cluster {safest['cluster']} "
          f"({safest['churn_rate']}% de churn) — {safest['profil']}")

    # ── Partie C : Ajouter le cluster comme feature ──
    print(f"\n  ➕ Ajout de 'KMeans_Cluster' comme feature dans X_train et X_test...")
    X_train_enriched = X_train.copy()
    X_test_enriched  = X_test.copy()
    X_train_enriched['KMeans_Cluster'] = train_clusters
    X_test_enriched['KMeans_Cluster']  = test_clusters
    print(f"     Shape enrichie : X_train={X_train_enriched.shape} | X_test={X_test_enriched.shape}")

    # ── Sauvegarde ──
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({'kmeans': kmeans, 'scaler': scaler_km, 'features': available},
                os.path.join(MODELS_DIR, 'kmeans.pkl'))
    print(f"     💾 KMeans + scaler sauvegardés → models/kmeans.pkl")

    return {
        'kmeans':           kmeans,
        'scaler_km':        scaler_km,
        'k':                k,
        'inertia':          inertia,
        'silhouette':       sil_train,
        'cluster_stats':    cluster_stats,
        'X_train_enriched': X_train_enriched,
        'X_test_enriched':  X_test_enriched,
        'train_clusters':   train_clusters,
        'test_clusters':    test_clusters,
    }

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────

def optimize_best_model(best_model_name: str,
                        X_train, y_train) -> object:
    """
    Optimise le meilleur modèle avec GridSearchCV.
    Cherche les hyperparamètres qui maximisent le F1-Score
    via validation croisée stratifiée.
    """
    print(f"\n{'=' * 60}")
    print(f"⚙️  OPTIMISATION : {best_model_name} (GridSearchCV)")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Grilles d'hyperparamètres pour chaque modèle
    param_grids = {
        'LogisticRegression': {
            'C':        [0.01, 0.1, 1.0, 10.0],
            'penalty':  ['l2'],
            'solver':   ['lbfgs', 'saga'],
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth':    [None, 10, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2'],
        },
        'XGBoost': {
            'n_estimators':  [100, 200],
            'max_depth':     [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample':     [0.8, 1.0],
        },
    }

    base_models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
        'XGBoost': XGBClassifier(
            random_state=RANDOM_STATE, scale_pos_weight=2,
            eval_metric='logloss', verbosity=0),
    }

    if best_model_name not in param_grids:
        print(f"   ⚠️  Pas de grille définie pour {best_model_name}")
        return base_models.get(best_model_name)

    model   = base_models[best_model_name]
    grid    = param_grids[best_model_name]
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    print(f"   → {n_combos} combinaisons × {CV_FOLDS} folds = {n_combos * CV_FOLDS} fits")

    grid_search = GridSearchCV(
        model, grid,
        cv=cv,
        scoring='f1',          # Optimiser le F1 (Churn déséquilibré)
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    grid_search.fit(X_train, y_train)

    print(f"\n   ✅ Meilleurs hyperparamètres :")
    for param, val in grid_search.best_params_.items():
        print(f"      → {param} = {val}")
    print(f"   → Meilleur F1 CV : {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    # Sauvegarder le modèle optimisé
    path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_model, path)
    print(f"\n   💾 Modèle optimisé sauvegardé → {path}")

    return best_model


# ─────────────────────────────────────────────
# ÉTAPE 7 — OPTIMISATION OPTUNA
# ─────────────────────────────────────────────

def optimize_with_optuna(best_model_name: str,
                         X_train, y_train,
                         grid_best_params: dict,
                         n_trials: int = 80) -> object:
    """
    Affine les hyperparamètres avec Optuna (algorithme TPE — Tree-structured
    Parzen Estimator), en partant des meilleurs paramètres trouvés par GridSearch.

    COMMENT OPTUNA FONCTIONNE :
    ────────────────────────────
    À chaque essai (trial), Optuna :
      1. Regarde les résultats des essais précédents
      2. Construit un modèle probabiliste de la fonction objectif
      3. Choisit les paramètres les plus PROMETTEURS à tester ensuite
         (là où il estime que le score sera le meilleur)
      4. Teste, met à jour son modèle, recommence

    C'est l'opposé de GridSearch qui teste TOUT sans apprendre.

    DIFFÉRENCE CLÉE avec GridSearch :
      GridSearch : learning_rate ∈ {0.05, 0.1, 0.2}  ← 3 valeurs fixes
      Optuna     : learning_rate ∈ [0.01, 0.30]       ← espace continu
                   → peut trouver 0.073, 0.124, etc.

    L'espace de recherche Optuna est CENTRÉ autour des meilleurs
    paramètres GridSearch pour affiner précisément.
    """
    print(f"\n{'=' * 60}")
    print(f"🧠 OPTIMISATION OPTUNA : {best_model_name}")
    print(f"   {n_trials} essais intelligents (algorithme TPE bayésien)")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Définir la fonction objectif qu'Optuna va maximiser ──
    def objective(trial):
        """
        Optuna appelle cette fonction à chaque essai.
        Elle suggère des hyperparamètres, entraîne le modèle,
        et retourne le score F1 en validation croisée.
        """
        if best_model_name == 'XGBoost':
            # Espace de recherche centré autour des meilleurs params GridSearch
            gs_depth = grid_best_params.get('max_depth', 3)
            gs_lr    = grid_best_params.get('learning_rate', 0.1)
            gs_n     = grid_best_params.get('n_estimators', 100)

            params = {
                # trial.suggest_int/float → Optuna choisit intelligemment dans cet espace
                'n_estimators':  trial.suggest_int('n_estimators',
                                    max(50, gs_n - 50), gs_n + 150),
                'max_depth':     trial.suggest_int('max_depth',
                                    max(2, gs_depth - 1), gs_depth + 2),
                'learning_rate': trial.suggest_float('learning_rate',
                                    max(0.01, gs_lr * 0.5), min(0.5, gs_lr * 2),
                                    log=True),   # log=True = exploration log-scale
                'subsample':     trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma':         trial.suggest_float('gamma', 0.0, 0.5),
                'reg_alpha':     trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda':    trial.suggest_float('reg_lambda', 0.5, 2.0),
            }
            model = XGBClassifier(
                **params,
                random_state=RANDOM_STATE,
                scale_pos_weight=2,
                eval_metric='logloss',
                verbosity=0
            )

        elif best_model_name == 'RandomForest':
            gs_n     = grid_best_params.get('n_estimators', 100)
            gs_depth = grid_best_params.get('max_depth', 10) or 20

            params = {
                'n_estimators':      trial.suggest_int('n_estimators',
                                        max(50, gs_n - 50), gs_n + 150),
                'max_depth':         trial.suggest_int('max_depth',
                                        max(3, gs_depth - 5), gs_depth + 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features':      trial.suggest_categorical('max_features',
                                        ['sqrt', 'log2']),
            }
            model = RandomForestClassifier(
                **params,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            )

        elif best_model_name == 'LogisticRegression':
            gs_c = grid_best_params.get('C', 1.0)
            params = {
                'C': trial.suggest_float('C',
                         max(0.001, gs_c * 0.1), gs_c * 10, log=True),
            }
            model = LogisticRegression(
                **params,
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                penalty='l2', solver='lbfgs'
            )
        else:
            return 0.0

        # Validation croisée — score qu'Optuna cherche à maximiser
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring='f1', n_jobs=-1)
        return scores.mean()

    # ── Lancer l'étude Optuna ──
    study = optuna.create_study(
        direction='maximize',   # On veut MAXIMISER le F1
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )

    print(f"   Essai  {'F1 CV':>8}  {'Meilleur':>9}  Paramètres clés")
    print("   " + "-" * 65)

    best_so_far = 0.0
    def print_callback(study, trial):
        nonlocal best_so_far
        val = trial.value or 0.0
        if val > best_so_far:
            best_so_far = val
        # Afficher seulement les multiples de 10 et les améliorations
        if trial.number % 10 == 0 or val == study.best_value:
            key_param = list(trial.params.items())[0] if trial.params else ('—', '—')
            print(f"   #{trial.number:<4} {val:>8.4f}  {study.best_value:>9.4f}  "
                  f"{key_param[0]}={key_param[1]}")

    study.optimize(objective, n_trials=n_trials, callbacks=[print_callback])

    print(f"\n   ✅ Optuna terminé — Meilleur F1 CV : {study.best_value:.4f}")
    print(f"   Meilleurs hyperparamètres trouvés par Optuna :")
    for param, val in study.best_params.items():
        gs_val = grid_best_params.get(param, '—')
        print(f"      → {param:<25} = {str(val):<12}  (GridSearch avait : {gs_val})")

    # ── Entraîner le modèle final avec les meilleurs paramètres Optuna ──
    if best_model_name == 'XGBoost':
        final_model = XGBClassifier(
            **study.best_params,
            random_state=RANDOM_STATE,
            scale_pos_weight=2,
            eval_metric='logloss',
            verbosity=0
        )
    elif best_model_name == 'RandomForest':
        final_model = RandomForestClassifier(
            **study.best_params,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        )
    elif best_model_name == 'LogisticRegression':
        final_model = LogisticRegression(
            **study.best_params,
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            penalty='l2', solver='lbfgs'
        )

    final_model.fit(X_train, y_train)

    # Sauvegarder
    path = os.path.join(MODELS_DIR, 'best_model_optuna.pkl')
    joblib.dump(final_model, path)
    joblib.dump(study, os.path.join(MODELS_DIR, 'optuna_study.pkl'))
    print(f"   💾 Modèle Optuna sauvegardé → {path}")

    return final_model, study
# ─────────────────────────────────────────────

def show_feature_importance(model, feature_names: list, top_n: int = 20) -> None:
    """
    Affiche les N features les plus importantes pour les modèles arborescents.
    Permet de comprendre quels signaux sont les plus prédictifs du churn.
    """
    print(f"\n{'=' * 60}")
    print(f"🔍 TOP {top_n} FEATURES LES PLUS IMPORTANTES")
    print("=" * 60)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        print(f"\n{'Rang':<6} {'Feature':<40} {'Importance':>10}")
        print("-" * 58)
        for rank, idx in enumerate(indices, 1):
            name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            print(f"{rank:<6} {name:<40} {importances[idx]:>10.4f}")

    elif hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_[0])
        indices = np.argsort(coefs)[::-1][:top_n]

        print(f"\n{'Rang':<6} {'Feature':<40} {'|Coefficient|':>14}")
        print("-" * 62)
        for rank, idx in enumerate(indices, 1):
            name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            print(f"{rank:<6} {name:<40} {coefs[idx]:>14.4f}")
    else:
        print("   ⚠️  Ce modèle ne supporte pas l'importance des features.")


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_training(data_dir: str = 'data/train_test',
                 models_dir: str = 'models') -> dict:
    """
    Lance le pipeline complet d'entraînement.
    1. Charge les données préprocessées
    2. Compare 3 modèles
    3. Optimise le meilleur
    4. Affiche l'importance des features
    5. Évaluation finale du modèle optimisé
    """
    global MODELS_DIR
    MODELS_DIR = models_dir

    print("\n" + "=" * 60)
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT")
    print("=" * 60)

    # ── Chargement ──
    X_train, X_test, y_train, y_test = load_train_test(data_dir)
    if X_train is None:
        raise ValueError("Données train/test introuvables. Lance d'abord preprocessing.py.")

    feature_names = X_train.columns.tolist()

    # ── Comparaison des 3 modèles supervisés ──
    df_results, all_results = compare_models(X_train, y_train, X_test, y_test)

    # ── KMeans : Analyse non supervisée ──
    kmeans_results = run_kmeans_analysis(X_train, X_test, y_train, y_test)

    # ── Option : réentraîner le meilleur modèle supervisé avec la feature KMeans ──
    print(f"\n{'=' * 60}")
    print("🔁 TEST : Meilleur modèle supervisé + feature KMeans")
    print("=" * 60)
    print("   → On ajoute 'KMeans_Cluster' aux features et on réévalue...")

    X_train_enr = kmeans_results['X_train_enriched']
    X_test_enr  = kmeans_results['X_test_enriched']

    # Identifier le meilleur modèle supervisé
    best_row  = df_results.loc[df_results['F1-Score'].idxmax()]
    best_name = best_row['Modèle']

    # Réentraîner le meilleur modèle avec les données enrichies
    models_dict = get_base_models()
    best_model_enriched = models_dict[best_name]
    metrics_enriched = train_and_evaluate(
        best_model_enriched,
        f"{best_name} + KMeans_Cluster",
        X_train_enr, y_train,
        X_test_enr,  y_test
    )

    f1_before = best_row['F1-Score']
    f1_after  = metrics_enriched['f1']
    delta     = round(f1_after - f1_before, 4)
    print(f"\n   F1 sans KMeans feature : {f1_before}")
    print(f"   F1 avec KMeans feature : {f1_after}")
    print(f"   Delta                  : {'+' if delta >= 0 else ''}{delta}  "
          f"{'✅ amélioration' if delta > 0 else ('= neutre' if delta == 0 else '⚠️ dégradation')}")

    # On garde la version qui donne le meilleur F1
    if f1_after >= f1_before:
        print(f"\n   → Utilisation des données ENRICHIES pour la suite")
        X_train_final = X_train_enr
        X_test_final  = X_test_enr
        feature_names = X_train_enr.columns.tolist()
    else:
        print(f"\n   → Utilisation des données ORIGINALES (KMeans n'améliore pas)")
        X_train_final = X_train
        X_test_final  = X_test

    # ── Étape 1 : GridSearchCV (exploration large) ──
    model_grid = optimize_best_model(best_name, X_train_final, y_train)
    metrics_grid = train_and_evaluate(
        model_grid, f"{best_name} — GridSearch",
        X_train_final, y_train, X_test_final, y_test
    )

    # Récupérer les meilleurs params GridSearch pour guider Optuna
    gs_path = os.path.join(models_dir, f'{best_name}_gridsearch_params.pkl')
    grid_best_params = {}
    if os.path.exists(gs_path):
        grid_best_params = joblib.load(gs_path)
    elif hasattr(model_grid, 'get_params'):
        grid_best_params = {k: v for k, v in model_grid.get_params().items()
                            if v is not None}

    # ── Étape 2 : Optuna (affinement précis) ──
    model_optuna, study = optimize_with_optuna(
        best_name, X_train_final, y_train,
        grid_best_params=grid_best_params,
        n_trials=50
    )
    metrics_optuna = train_and_evaluate(
        model_optuna, f"{best_name} — Optuna",
        X_train_final, y_train, X_test_final, y_test
    )

    # ── Comparaison GridSearch vs Optuna ──
    print(f"\n{'=' * 60}")
    print("⚖️  COMPARAISON : GridSearchCV  vs  Optuna")
    print("=" * 60)
    print(f"\n   {'Métrique':<15} {'GridSearch':>12} {'Optuna':>12} {'Δ':>8}")
    print("   " + "-" * 50)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        vg = metrics_grid[metric]
        vo = metrics_optuna[metric]
        delta = round(vo - vg, 4)
        arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '=')
        print(f"   {metric:<15} {vg:>12.4f} {vo:>12.4f} {arrow}{abs(delta):>6.4f}")

    # Choisir le meilleur final
    if metrics_optuna['f1'] >= metrics_grid['f1']:
        best_model_optimized = model_optuna
        winner = "Optuna"
    else:
        best_model_optimized = model_grid
        winner = "GridSearchCV"

    print(f"\n   🏆 Vainqueur : {winner} (F1 = {max(metrics_grid['f1'], metrics_optuna['f1'])})")

    # Sauvegarder le meilleur
    joblib.dump(best_model_optimized, os.path.join(models_dir, 'best_model.pkl'))
    print(f"   💾 Meilleur modèle final sauvegardé → models/best_model.pkl")

    # ── Évaluation finale ──
    print(f"\n{'=' * 60}")
    print(f"📊 ÉVALUATION FINALE — {best_name} ({winner})")
    print("=" * 60)
    final_metrics = train_and_evaluate(
        best_model_optimized, f"{best_name} ({winner})",
        X_train_final, y_train, X_test_final, y_test
    )
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    print("\n📈 Courbe Precision-Recall...")

# Probabilités du modèle final
    y_scores = best_model_optimized.predict_proba(X_test)[:, 1]

# Calcul
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    ap_score = average_precision_score(y_test, y_scores)

# Plot
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Modèle Final)')
    plt.legend()
    plt.grid()
    plt.show()

    # ── Importance des features ──
    show_feature_importance(best_model_optimized, feature_names, top_n=20)

    print("\n" + "=" * 60)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print(f"   Modèle final    : {best_name} ({winner})")
    print(f"   AUC-ROC final   : {final_metrics['roc_auc']}")
    print(f"   F1-Score final  : {final_metrics['f1']}")
    print(f"   Recall          : {final_metrics['recall']}")
    print("=" * 60)

    return {
        'best_model':      best_model_optimized,
        'best_model_name': best_name,
        'winner':          winner,
        'metrics':         final_metrics,
        'metrics_grid':    metrics_grid,
        'metrics_optuna':  metrics_optuna,
        'comparison':      df_results,
        'feature_names':   feature_names,
        'kmeans':          kmeans_results,
    }