"""
utils.py — Fonctions utilitaires pour le projet Churn
Chargement des données + affichage des infos de base
"""

import pandas as pd
import numpy as np
import os


# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────

def load_data(file_path: str) -> pd.DataFrame | None:
    """
    Charge le dataset CSV depuis file_path.
    Corrige automatiquement les conflits Git (<<<<<<< HEAD) si présents.
    Retourne un DataFrame propre ou None si erreur.
    """
    try:
        # Vérifier si le fichier contient des marqueurs de conflit Git
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

        if first_line.startswith('<<<<<<<'):
            print("⚠️  Conflit Git détecté dans le fichier CSV — résolution automatique...")
            df = _resolve_git_conflict(file_path)
        else:
            df = pd.read_csv(file_path)

        print(f"✅ Chargement réussi : {df.shape[0]} lignes et {df.shape[1]} colonnes.")
        return df

    except FileNotFoundError:
        print(f"❌ Fichier introuvable : {file_path}")
        return None
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None


def _resolve_git_conflict(file_path: str) -> pd.DataFrame:
    """
    Lit un CSV qui contient des marqueurs de conflit Git.
    Garde uniquement la version HEAD (entre <<<<<<< HEAD et =======).
    """
    import io
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    clean_lines = []
    in_head = False
    in_other = False

    for line in lines:
        if line.startswith('<<<<<<<'):
            in_head = True
            continue
        elif line.startswith('======='):
            in_head = False
            in_other = True
            continue
        elif line.startswith('>>>>>>>'):
            in_other = False
            continue

        if not in_other:
            clean_lines.append(line)

    return pd.read_csv(io.StringIO(''.join(clean_lines)))


# ─────────────────────────────────────────────
# 2. RAPPORT D'EXPLORATION RAPIDE (EDA)
# ─────────────────────────────────────────────

def quick_eda(df: pd.DataFrame, target_col: str = 'Churn') -> None:
    """
    Affiche un rapport d'exploration complet du DataFrame :
    - Dimensions, types, valeurs manquantes
    - Distribution de la cible
    - Colonnes à variance nulle (inutiles)
    - Statistiques des colonnes numériques
    """
    print("=" * 60)
    print("📊 RAPPORT D'EXPLORATION DES DONNÉES")
    print("=" * 60)

    # Dimensions
    print(f"\n📐 Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # Types de colonnes
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
    print(f"   → {len(num_cols)} colonnes numériques")
    print(f"   → {len(cat_cols)} colonnes catégorielles/texte")

    # Valeurs manquantes
    print("\n🔍 Valeurs manquantes :")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("   → Aucune valeur manquante ✅")
    else:
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"   → {col:<30} {count:>5} NaN  ({pct:.1f}%)")

    # Distribution de la cible
    if target_col in df.columns:
        print(f"\n🎯 Distribution de la cible '{target_col}' :")
        vc = df[target_col].value_counts()
        vcp = df[target_col].value_counts(normalize=True)
        for val in vc.index:
            print(f"   → Classe {val} : {vc[val]:>5} ({vcp[val]*100:.1f}%)")
        ratio = vc.max() / vc.min()
        if ratio > 3:
            print(f"   ⚠️  Dataset déséquilibré (ratio {ratio:.1f}:1) — utiliser SMOTE ou class_weight")

    # Colonnes à variance nulle (inutiles)
    print("\n🗑️  Colonnes à variance nulle (à supprimer) :")
    zero_var = [c for c in cat_cols if df[c].nunique() <= 1]
    if zero_var:
        for c in zero_var:
            print(f"   → {c} : valeur unique = '{df[c].iloc[0]}'")
    else:
        print("   → Aucune")

    # Doublons
    dupes = df.duplicated().sum()
    print(f"\n🔁 Lignes dupliquées : {dupes}")

    # Aperçu des colonnes catégorielles
    print("\n📝 Colonnes catégorielles (valeurs uniques) :")
    for c in cat_cols:
        n = df[c].nunique()
        sample = df[c].dropna().unique()[:4]
        print(f"   → {c:<30} {n:>4} valeurs  ex: {list(sample)}")

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────
# 3. SAUVEGARDE / CHARGEMENT UTILITAIRES
# ─────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Sauvegarde un DataFrame en CSV avec confirmation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Sauvegardé : {path}  ({df.shape[0]} lignes × {df.shape[1]} colonnes)")


def load_train_test(data_dir: str = 'data/train_test') -> tuple:
    """
    Charge les fichiers X_train, X_test, y_train, y_test
    depuis le dossier data/train_test/.
    Retourne (X_train, X_test, y_train, y_test).
    """
    paths = {
        'X_train': os.path.join(data_dir, 'X_train.csv'),
        'X_test':  os.path.join(data_dir, 'X_test.csv'),
        'y_train': os.path.join(data_dir, 'y_train.csv'),
        'y_test':  os.path.join(data_dir, 'y_test.csv'),
    }
    try:
        X_train = pd.read_csv(paths['X_train'])
        X_test  = pd.read_csv(paths['X_test'])
        y_train = pd.read_csv(paths['y_train']).squeeze()
        y_test  = pd.read_csv(paths['y_test']).squeeze()
        print(f"✅ Train/test chargés : X_train={X_train.shape}, X_test={X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"❌ Erreur chargement train/test : {e}")
        return None, None, None, None

import pandas as pd

def load_data(file_path):
    """
    Charge le dataset CSV et affiche les dimensions de base.
    """
    try:
        df = pd.read_csv(file_path)
        print(f" Chargement réussi : {df.shape[0]} lignes et {df.shape[1]} colonnes.")
        return df
    except Exception as e:
        print(f" Erreur lors du chargement du fichier : {e}")
        return None
