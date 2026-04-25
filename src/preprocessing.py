"""
preprocessing.py — Pipeline complet de préparation des données
Étapes :
  1. Nettoyage (suppression colonnes inutiles, doublons)
  2. Parsing des dates et des IPs
  3. Feature Engineering (nouvelles features)
  4. Imputation des valeurs manquantes
  5. Encodage des variables catégorielles
  6. Analyse de corrélation et suppression de la multicolinéarité
  7. Séparation Train/Test (stratifiée)
  8. Normalisation (StandardScaler)
 9. ACP — Analyse en Composantes Principales (optionnelle)
 10. Sauvegarde des fichiers
"""

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import category_encoders as ce

from .utils import load_data, save_dataframe


# ─────────────────────────────────────────────
# CONSTANTES DU PROJET
# ─────────────────────────────────────────────

TARGET_COL = 'Churn'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Colonnes à supprimer immédiatement (non prédictives ou redondantes)
COLS_TO_DROP = [
    'CustomerID',           # Identifiant unique → pas prédictif
    'LastLoginIP',          # Remplacé par des features extraites (IP privée/publique)
        # Remplacée par RegYear, RegMonth, RegDay
    'NewsletterSubscribed', # Variance nulle → toujours "Yes"
    'UniqueInvoices',       # Identique à Frequency (même valeurs)
    'CancelledTransactions',# Identique à NegativeQuantityCount
    'Country',
    'PreferredMonth',              # Redondant avec Region (même info géographique)
    #  DATA LEAKAGE — ces colonnes encodent directement le Churn → à supprimer absolument
    'ChurnRiskCategory',    # "Critique" = Churn=1 à 100%, "Faible/Moyen/Élevé" = Churn=0
    'CustomerType',         # "Perdu" = Churn=1 à 100%, "Hyperactif/Occasionnel/Régulier" = Churn=0
    'RFMSegment',
    'AgeCategory',
    'FirstPurchaseDaysAgo',
    'FavoriteSeason',       #  LEAKAGE : Automne = Churn=0 à 98%
    'CustomerTenureDays',
    'LoyaltyLevel',
    'UniqueCountries',
    'Age',
    'Recency',
    
]

# Colonnes catégorielles ordonnées (avec ordre logique)
ORDINAL_MAPPINGS = {
    'SpendingCategory':    ['Low', 'Medium', 'High', 'VIP'],
    'BasketSizeCategory':  ['Petit', 'Moyen', 'Grand'],
    'SatisfactionScore':   None,   # Numérique → pas d'encodage ordinal
}

# Colonnes catégorielles nominales (sans ordre) → One-Hot Encoding
NOMINAL_COLS = [
    'PreferredTimeOfDay', 'WeekendPreference', 'ProductDiversity',
    'Gender', 'AccountStatus', 'RegSeason'
]

NUMERIC_COLS_STORED_AS_OBJECT = [
    'MonetaryTotal', 'MonetaryAvg', 'MonetaryStd',
    'MonetaryMin', 'MonetaryMax', 'AvgQuantityPerTransaction',
    'AvgDaysBetweenPurchases', 'AvgProductsPerTransaction',
    'AvgLinesPerInvoice',
]
EUROPE_REGIONS = {
    'Europe continentale', 'Europe du Nord',
    'Europe du Sud', 'Europe centrale', "Europe de l'Est"
}


# ─────────────────────────────────────────────
# ÉTAPE 1 — NETTOYAGE DE BASE
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes inutiles, les doublons,
    et corrige les valeurs aberrantes évidentes.
    """
    print("\n[1/8] 🧹 Nettoyage de base...")
    df = df.copy()

    # Supprimer doublons
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df.drop_duplicates(inplace=True)
        print(f"   → {n_dupes} doublons supprimés")

    # Supprimer colonnes inutiles (seulement celles présentes dans le df)
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df.drop(columns=cols_present, inplace=True)
    print(f"   → Colonnes supprimées : {cols_present}")

    # SupportTicketsCount : valeurs négatives (-1) → remplacer par 0
    if 'SupportTicketsCount' in df.columns:
        neg_count = (df['SupportTicketsCount'] < 0).sum()
        df['SupportTicketsCount'] = df['SupportTicketsCount'].clip(lower=0)
        if neg_count > 0:
            print(f"   → SupportTicketsCount : {neg_count} valeurs négatives remplacées par 0")

    # SatisfactionScore : valeurs > 10 → aberrantes (échelle 0-10), clip
    if 'SatisfactionScore' in df.columns:
        weird = (df['SatisfactionScore'] > 10).sum()
        df['SatisfactionScore'] = df['SatisfactionScore'].clip(lower=0, upper=10)
        if weird > 0:
            print(f"   → SatisfactionScore : {weird} valeurs > 10 clippées à 10")


    # Forcer la conversion numérique (errors='coerce' → NaN si invalide)
    for col in NUMERIC_COLS_STORED_AS_OBJECT:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            n_coerced = df[col].isna().sum()
            if n_coerced > 0:
                print(f"   → {col} : {n_coerced} valeurs invalides → NaN (seront imputées)")


    print(f"   → Shape après nettoyage : {df.shape}")
    return df
from imblearn.over_sampling import SMOTE

def balance_classes(X_train, y_train):
    print("\n[SMOTE] ⚖️  Rééquilibrage des classes...")
    print(f"   → Avant : {y_train.value_counts().to_dict()}")
    
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    print(f"   → Après : {pd.Series(y_res).value_counts().to_dict()}")
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)
# ─────────────────────────────────────────────
# ÉTAPE 2 — PARSING DES DATES
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# ÉTAPE 2 — PARSING DES DATES
# ─────────────────────────────────────────────

import re

def _parse_registration_date(series: pd.Series) -> pd.Series:
    """
    Parse RegistrationDate avec détection de format par règle explicite.

    Problème de l'approche naïve (dayfirst=True) :
      → pd.to_datetime applique dayfirst=True aux formats slash (/),
        MAIS ignore ce paramètre pour les formats ISO (YYYY-MM-DD).
      → Résultat : '2010-10-04' parsé en 2010-10-04 (correct),
                   mais '05/10/2010' parsé en 2010-05-10 alors qu'on
                   voulait 2010-10-05 (UK convention).
      → 529 dates incorrectes sur 4372 avec l'approche naïve.

    Formats présents dans le dataset :
    ┌──────────────────┬──────────────┬────────────────────────────────────┐
    │ Format           │ Exemple      │ Détection                          │
    ├──────────────────┼──────────────┼────────────────────────────────────┤
    │ YYYY-MM-DD (ISO) │ 2010-10-04   │ contient des tirets                │
    │ DD/MM/YYYY       │ 17/09/2010   │ part1 > 12 → jour forcément        │
    │ MM/DD/YYYY       │ 10/18/2010   │ part2 > 12 → mois US forcément     │
    │ Ambigu /YYYY     │ 05/10/2010   │ ≤ 12 des deux côtés → UK → DD/MM   │
    │ DD/MM/YY         │ 17/07/10     │ part1 > 12 → jour forcément        │
    │ MM/DD/YY         │ 09/23/10     │ part2 > 12 → mois US forcément     │
    │ Ambigu /YY       │ 02/11/10     │ ≤ 12 des deux côtés → UK → DD/MM   │
    └──────────────────┴──────────────┴────────────────────────────────────┘

    Règle pour les cas ambigus :
      90% des clients sont UK → convention européenne DD/MM prioritaire.
    """
    def _parse_one(s):
        s = str(s).strip()

        # 1. ISO YYYY-MM-DD — non ambigu
        if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
            return pd.to_datetime(s, format='%Y-%m-%d')

        # 2. Slash + année 4 chiffres
        if re.match(r'^\d{2}/\d{2}/\d{4}$', s):
            p1, p2, _ = s.split('/')
            if int(p1) > 12:       # jour > 12 → DD/MM/YYYY
                return pd.to_datetime(s, format='%d/%m/%Y')
            elif int(p2) > 12:     # mois > 12 → format US MM/DD/YYYY
                return pd.to_datetime(s, format='%m/%d/%Y')
            else:                  # ambigu → contexte UK → DD/MM/YYYY
                return pd.to_datetime(s, format='%d/%m/%Y')

        # 3. Slash + année 2 chiffres
        if re.match(r'^\d{2}/\d{2}/\d{2}$', s):
            p1, p2, _ = s.split('/')
            if int(p1) > 12:       # DD/MM/YY
                return pd.to_datetime(s, format='%d/%m/%y')
            elif int(p2) > 12:     # format US MM/DD/YY
                return pd.to_datetime(s, format='%m/%d/%y')
            else:                  # ambigu → DD/MM/YY
                return pd.to_datetime(s, format='%d/%m/%y')

        return pd.NaT  # format inconnu → sera imputé

    return series.apply(_parse_one)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse RegistrationDate et extrait des features temporelles.
    """
    print("\n[2/8] 📅 Parsing des dates...")
    df = df.copy()

    if 'RegistrationDate' not in df.columns:
        print("   → RegistrationDate déjà traitée, skip.")
        return df

    df['RegistrationDate'] = _parse_registration_date(df['RegistrationDate'])

    n_nat = df['RegistrationDate'].isna().sum()
    if n_nat > 0:
        print(f"   ⚠️  {n_nat} dates non parsables (NaT) — seront imputées")
    else:
        print(f"   ✅ 0 NaT — tous les formats reconnus")

    # Features temporelles extraites
    df['RegYear']      = df['RegistrationDate'].dt.year
    df['RegMonth']     = df['RegistrationDate'].dt.month
    df['RegDay']       = df['RegistrationDate'].dt.day
    df['RegWeekday']   = df['RegistrationDate'].dt.weekday     # 0=Lundi
    df['RegIsWeekend'] = (df['RegWeekday'] >= 5).astype(int)

    # Âge du compte en jours (ref = date max du dataset)
    ref_date = df['RegistrationDate'].max()
    df['AccountAgeDays'] = (ref_date - df['RegistrationDate']).dt.days
    df['RegSeason'] = df['RegMonth'].map({
    12: 'hiver', 1: 'hiver', 2: 'hiver',
     3: 'printemps', 4: 'printemps', 5: 'printemps',
     6: 'été', 7: 'été', 8: 'été',
     9: 'automne', 10: 'automne', 11: 'automne'})
    # Nouvelle feature binaire : nouveau client (< 1 an)
    ref_date = df['RegistrationDate'].max()  # déjà calculé avant ce point
    df['IsNewClient'] = (df['AccountAgeDays'] < 365).astype(int)

    df.drop(columns=['RegistrationDate'], inplace=True)
    df.drop(columns=['RegYear'], inplace=True)
    df.drop(columns=['RegDay'], inplace=True) # RegDay n'apporte pas d'information pertinente pour le churn (pas de pattern cyclique évident)
    print(f"   → Features créées :  RegMonth,RegWeekday, RegIsWeekend, AccountAgeDays")
    return df


# ─────────────────────────────────────────────
# ÉTAPE 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée de nouvelles features à partir des features existantes
    pour capturer des patterns comportementaux supplémentaires.
    """
    print("\n[3/8] ⚙️  Feature Engineering...")
    df = df.copy()
    new_features = []

    # --- Features RFM enrichies ---
    # Dépenses normalisées par la récence (client actif qui dépense beaucoup = faible churn)
    # if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
    #     df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    #     new_features.append('MonetaryPerDay')

    # if 'Frequency' in df.columns and 'Recency' in df.columns:
    #     df['FrequencyPerDay'] = df['Frequency'] / (df['Recency'] + 1)
    #     new_features.append('FrequencyPerDay')
    # Valeur moyenne du panier
    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)
        new_features.append('AvgBasketValue')

    # Fréquence relative (transactions par jour de tenure)
    if 'TotalTransactions' in df.columns and 'CustomerTenureDays' in df.columns:
        df['TransactionsPerDay'] = df['TotalTransactions'] / (df['CustomerTenureDays'] + 1)
        new_features.append('TransactionsPerDay')

    # --- Comportement d'achat ---
    # Taux d'annulation (plus le taux est élevé, plus le client est insatisfait)
    if 'NegativeQuantityCount' in df.columns and 'TotalTransactions' in df.columns:
        df['CancellationRate'] = df['NegativeQuantityCount'] / (df['TotalTransactions'] + 1)
        new_features.append('CancellationRate')
        
    # 0=Autre, 1=Europe, 2=UK  → ordinal, pas de dummy trap, pas de multicolinéarité
    # if 'Region' in df.columns:
    #     def encode_region(r):
    #         if r == 'UK':
    #             return 2
    #         elif r in EUROPE_REGIONS:
    #             return 1
    #         else:
    #             return 0
    #     df['RegionGroup'] = df['Region'].map(encode_region)
    #     df.drop(columns=['Region'], inplace=True)
    #     new_features.append('RegionGroup')
    # Volatilité des dépenses (clients erratiques = instables)
    if 'MonetaryStd' in df.columns and 'MonetaryAvg' in df.columns:
        df['SpendingVolatility'] = df['MonetaryStd'] / (abs(df['MonetaryAvg']) + 1)
        new_features.append('SpendingVolatility')

    # --- Engagement client ---
    # Score d'engagement combiné (normalisation simple)
    if all(c in df.columns for c in ['SupportTicketsCount', 'SatisfactionScore']):
        # Beaucoup de tickets + satisfaction basse = fort risque de churn
        df['EngagementScore'] = df['SatisfactionScore'] / (df['SupportTicketsCount'] + 1)
        new_features.append('EngagementScore')

    # Diversité produit normalisée par le nombre de transactions
    if 'UniqueProducts' in df.columns and 'TotalTransactions' in df.columns:
        df['ProductsPerTransaction'] = df['UniqueProducts'] / (df['TotalTransactions'] + 1)
        new_features.append('ProductsPerTransaction')
    
    
    print(f"   → {len(new_features)} nouvelles features créées : {new_features}")
    return df

# ─────────────────────────────────────────────
# ÉTAPE 4 — IMPUTATION DES VALEURS MANQUANTES
# ─────────────────────────────────────────────

def impute_missing(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Impute les valeurs manquantes séparément sur X_train et X_test.
    IMPORTANT : fit sur X_train uniquement, transform sur les deux.
    Retourne (X_train_imputed, X_test_imputed, imputers_dict).
    """
    print("\n[4/8] 🩹 Imputation des valeurs manquantes...")
    X_train = X_train.copy()
    X_test  = X_test.copy()
    imputers = {}

    # Colonnes numériques → médiane (robuste aux outliers)
    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    num_missing = [c for c in num_cols if X_train[c].isnull().any()]

    if num_missing:
        imp_median = SimpleImputer(strategy='median')
        X_train[num_missing] = imp_median.fit_transform(X_train[num_missing])
        X_test[num_missing]  = imp_median.transform(X_test[num_missing])
        imputers['numeric'] = imp_median
        print(f"   → Médiane appliquée sur : {num_missing}")

    # Colonnes catégorielles → mode (valeur la plus fréquente)
    cat_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    cat_missing = [c for c in cat_cols if X_train[c].isnull().any()]

    if cat_missing:
        imp_mode = SimpleImputer(strategy='most_frequent')
        X_train[cat_missing] = imp_mode.fit_transform(X_train[cat_missing])
        X_test[cat_missing]  = imp_mode.transform(X_test[cat_missing])
        imputers['categorical'] = imp_mode
        print(f"   → Mode appliqué sur : {cat_missing}")

    remaining = X_train.isnull().sum().sum()
    print(f"   → Valeurs manquantes restantes : {remaining}")
    # Dans impute_missing(), ajouter pour les colonnes critiques :
    critical_cols = ['SatisfactionScore', 'AvgDaysBetweenPurchases']
    critical_missing = [c for c in critical_cols if c in X_train.columns 
                        and X_train[c].isnull().any()]

    if critical_missing:
        knn_imp = KNNImputer(n_neighbors=5)
        X_train[critical_missing] = knn_imp.fit_transform(X_train[critical_missing])
        X_test[critical_missing]  = knn_imp.transform(X_test[critical_missing])
        imputers['knn'] = knn_imp
        print(f"   → KNN Imputer appliqué sur : {critical_missing}")

    return X_train, X_test, imputers


# ─────────────────────────────────────────────
# ÉTAPE 5 — ENCODAGE DES VARIABLES CATÉGORIELLES
# ─────────────────────────────────────────────

def encode_categoricals(X_train: pd.DataFrame,
                        X_test: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Encode les variables catégorielles :
    - Ordinal Encoding pour les colonnes avec ordre logique
    - One-Hot Encoding pour les colonnes nominales (sans ordre)
    """
    print("\n[5/8] 🔤 Encodage des variables catégorielles...")
    X_train = X_train.copy()
    X_test  = X_test.copy()
    encoders = {}

    # --- Ordinal Encoding (colonnes avec ordre logique) ---
    ordinal_cols_present = {
        col: mapping
        for col, mapping in ORDINAL_MAPPINGS.items()
        if col in X_train.columns and mapping is not None
    }

    if ordinal_cols_present:
        for col, order in ordinal_cols_present.items():
            enc = OrdinalEncoder(
                categories=[order],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            X_train[[col]] = enc.fit_transform(X_train[[col]])
            X_test[[col]]  = enc.transform(X_test[[col]])
            encoders[f'ordinal_{col}'] = enc
        print(f"   → Ordinal Encoding : {list(ordinal_cols_present.keys())}")
    if 'Region' in X_train.columns:
        te = ce.TargetEncoder(cols=['Region'], smoothing=1.0)
        # fit UNIQUEMENT sur X_train + y_train → pas de leakage
        X_train['Region'] = te.fit_transform(X_train['Region'], y_train)
        X_test['Region']  = te.transform(X_test['Region'])
        encoders['target_encoder_region'] = te
        print("   → Target Encoding appliqué sur : Region")

    # --- One-Hot Encoding (colonnes nominales) ---
    nominal_present = [c for c in NOMINAL_COLS if c in X_train.columns]

    if nominal_present:
        X_train = pd.get_dummies(X_train, columns=nominal_present, drop_first=False, dtype=int)
        X_test  = pd.get_dummies(X_test,  columns=nominal_present, drop_first=False, dtype=int)

        # Aligner les colonnes (X_test peut manquer certaines colonnes après get_dummies)
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
        print(f"   → One-Hot Encoding : {nominal_present}")
        encoders['ohe_columns'] = nominal_present

    print(f"   → Shape après encodage : X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, encoders


# ─────────────────────────────────────────────
# ÉTAPE 6 — SUPPRESSION DE LA MULTICOLINÉARITÉ
# ─────────────────────────────────────────────

def remove_multicollinearity(X_train: pd.DataFrame,
                             X_test: pd.DataFrame,
                             threshold: float = 0.85) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Supprime les features trop corrélées entre elles (|r| > threshold).
    Garde celle qui a le plus de variance (std la plus élevée).
    Retourne (X_train, X_test, liste des colonnes supprimées).
 
    ALGORITHME :
    ─────────────
    1. Matrice de corrélation absolue sur les colonnes numériques
    2. Triangle supérieur STRICT (k=1) → chaque paire (A,B) apparaît une seule fois
    3. stack() pour extraire les paires réelles (supprime les NaN automatiquement)
    4. Pour chaque paire avec |r| > threshold :
       → Supprimer celle avec la std la plus faible (moins informative)
 
    POURQUOI NE PAS UTILISER upper[col] ?
    ───────────────────────────────────────
    upper[col] lit la COLONNE de 'col' dans le triangle supérieur.
    Or dans triu(k=1), pour la colonne 'col' à position i :
      - Les lignes j < i  → valeur réelle (triangle sup)
      - Les lignes j >= i → NaN (triangle inf + diagonale)
    La condition NaN > threshold = False en pandas,
    MAIS any(Series_avec_NaN) peut retourner True si NaN est présent.
    Résultat : faux positifs type 'MonetaryStd corrélé avec [MonetaryStd]'.
    La solution : stack() supprime les NaN et donne exactement les paires.
    """
    print(f"\n[6/8] 🔗 Suppression de la multicolinéarité (seuil |r| > {threshold})...")
 
    num_cols = X_train.select_dtypes(include='number').columns.tolist()
 
    if len(num_cols) < 2:
        print("   → Moins de 2 colonnes numériques, skip.")
        return X_train, X_test, []
 
    # ── Matrice de corrélation absolue ──
    corr_matrix = X_train[num_cols].corr().abs()
 
    # ── Triangle supérieur STRICT (k=1 exclut la diagonale) ──
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    upper = corr_matrix.where(mask)
 
    # ── Extraire les paires avec |r| > threshold via stack() ──
    # stack() supprime automatiquement les NaN → uniquement les vraies paires
    high_corr_pairs = (
        upper.stack()
             .reset_index()
             .rename(columns={'level_0': 'feature_A', 'level_1': 'feature_B', 0: 'correlation'})
             .query('correlation > @threshold')
             .sort_values('correlation', ascending=False)
    )
 
    if high_corr_pairs.empty:
        print("   → Aucune multicolinéarité détectée au seuil choisi ✅")
        return X_train, X_test, []
 
    print(f"   → {len(high_corr_pairs)} paires avec |r| > {threshold} :")
 
    # ── Pour chaque paire, supprimer la feature avec la std la plus faible ──
    to_drop = set()
 
    for _, row in high_corr_pairs.iterrows():
        feat_a = row['feature_A']
        feat_b = row['feature_B']
        corr   = row['correlation']
 
        # Si les deux sont déjà marquées pour suppression, rien à faire
        if feat_a in to_drop and feat_b in to_drop:
            continue
        # Si l'une est déjà supprimée, l'autre est gardée → rien à faire
        if feat_a in to_drop or feat_b in to_drop:
            continue
 
        # Supprimer celle avec la std la plus faible (moins de variance = moins informative)
        std_a = X_train[feat_a].std()
        std_b = X_train[feat_b].std()
        col_to_remove = feat_a if std_a <= std_b else feat_b
        col_to_keep   = feat_b if std_a <= std_b else feat_a
 
        to_drop.add(col_to_remove)
        print(f"   → |r|={corr:.3f} : '{feat_a}' ↔ '{feat_b}'"
              f" → suppression de '{col_to_remove}' (std={X_train[col_to_remove].std():.3f})")
 
    to_drop_list = list(to_drop)
    X_train.drop(columns=to_drop_list, inplace=True, errors='ignore')
    X_test.drop( columns=to_drop_list, inplace=True, errors='ignore')
 
    print(f"   → {len(to_drop_list)} colonnes supprimées : {to_drop_list}")
    return X_train, X_test, to_drop_list


# ─────────────────────────────────────────────
# ÉTAPE 7 — NORMALISATION (RobustScaler)
# ─────────────────────────────────────────────

def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Applique RobustScaler sur les colonnes numériques.
    - fit_transform sur X_train UNIQUEMENT
    - transform sur X_test (pour éviter le data leakage)
    Les colonnes binaires (0/1) issues du One-Hot sont exclues du scaling.
    """
    print("\n[7/8] 📏 Normalisation (RobustScaler)...")

    # Identifier les colonnes numériques continues (pas les binaires 0/1)
    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    binary_cols = [c for c in num_cols if X_train[c].nunique() <= 2]
    cols_to_scale = [c for c in num_cols if c not in binary_cols]

    scaler = RobustScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

    print(f"   → {len(cols_to_scale)} colonnes normalisées")
    print(f"   → {len(binary_cols)} colonnes binaires conservées telles quelles")
    return X_train, X_test, scaler
# ─────────────────────────────────────────────
# ÉTAPE 8 — ACP (PCA) — OPTIONNEL
# ─────────────────────────────────────────────
def apply_pca(X_train: pd.DataFrame,
              X_test: pd.DataFrame,
              y_train: pd.Series,
              variance_threshold: float = 0.95,
              replace_features: bool = False,
              models_dir: str = 'models') -> tuple:
    """
    Applique l'ACP (PCA) sur les données normalisées.
 
    POURQUOI L'ACP ?
    ─────────────────
    On a 109 features après encodage. Certaines sont redondantes ou bruitées.
    L'ACP projette ces 109 dimensions sur un nombre réduit de "composantes
    principales" qui capturent le maximum de variance (= d'information).
 
    COMMENT ÇA MARCHE ?
    ────────────────────
    1. On calcule la matrice de covariance de toutes les features
    2. On trouve les axes (vecteurs propres) qui maximisent la variance
    3. PC1 = l'axe qui explique le plus de variance
       PC2 = l'axe suivant, perpendiculaire à PC1
       ... etc.
    4. On garde les N premières composantes qui expliquent 95% de la variance totale
 
    DEUX MODES :
    ─────────────
    - replace_features=False (défaut) :
        On calcule l'ACP pour ANALYSER et VISUALISER uniquement.
        Les vraies features ne sont pas remplacées → les modèles XGBoost/RF
        gardent leurs features originales (plus interprétables).
 
    - replace_features=True :
        Les features originales sont REMPLACÉES par les composantes PCA.
        Utile pour la régression logistique ou les SVM (sensibles à la dimension).
        Attention : on perd l'interprétabilité (PC1, PC2 ne veulent rien dire
        métier, contrairement à Recency, Frequency, etc.)
 
    RÈGLE D'OR : fit sur X_train UNIQUEMENT, transform sur les deux.
 
    Retourne :
        (X_train_out, X_test_out, pca_model, n_components, variance_explained)
        Si replace_features=False → X_train_out == X_train (inchangé)
    """
    print(f"\n[9/10] 🔬 ACP — Analyse en Composantes Principales...")
    print(f"   → Variance à conserver : {variance_threshold*100:.0f}%")
    print(f"   → Mode : {'Remplacement des features' if replace_features else 'Analyse seulement'}")
 
    # Travailler uniquement sur les colonnes numériques
    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    n_features_avant = len(num_cols)
 
    # ── Phase 1 : Analyser la variance expliquée pour chaque composante ──
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_train[num_cols])
 
    # Variance cumulée
    variance_cumulative = np.cumsum(pca_full.explained_variance_ratio_)
 
    # Trouver combien de composantes sont nécessaires pour atteindre le seuil
    n_components = int(np.argmax(variance_cumulative >= variance_threshold) + 1)
 
    print(f"\n   📊 Analyse de la variance expliquée :")
    print(f"   {'Composantes':<15} {'Variance cumulée':>18}")
    print(f"   " + "-" * 35)
    checkpoints = [1, 2, 3, 5, 10, 20, n_components]
    seen = set()
    for n in sorted(set(checkpoints)):
        if n <= len(variance_cumulative) and n not in seen:
            seen.add(n)
            var = variance_cumulative[n - 1] * 100
            marker = " ← seuil atteint" if n == n_components else ""
            print(f"   PC1 à PC{n:<8} {var:>17.1f}%{marker}")
 
    print(f"\n   ✅ {n_components} composantes suffisent pour expliquer "
          f"{variance_threshold*100:.0f}% de la variance")
    print(f"   → Réduction : {n_features_avant} features → {n_components} composantes "
          f"({100*(1 - n_components/n_features_avant):.0f}% de réduction)")
 
    # ── Phase 2 : Entraîner la PCA finale avec n_components ──
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(X_train[num_cols])   # fit sur X_train UNIQUEMENT
 
    # ── Phase 3 : Afficher les features les plus influentes sur PC1 et PC2 ──
    print(f"\n   🔍 Features les plus influentes sur les 2 premières composantes :")
    for pc_idx, pc_name in enumerate(['PC1', 'PC2']):
        if pc_idx >= n_components:
            break
        loadings = pd.Series(
            np.abs(pca.components_[pc_idx]),
            index=num_cols
        ).sort_values(ascending=False).head(5)
        print(f"\n   {pc_name} (explique {pca.explained_variance_ratio_[pc_idx]*100:.1f}% de variance) :")
        for feat, load in loadings.items():
            print(f"      → {feat:<35} loading={load:.4f}")
 
    # ── Phase 4 : Décision selon le mode ──
    if replace_features:
        # Remplacer les features numériques par les composantes PCA
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
 
        X_train_pca = pca.transform(X_train[num_cols])
        X_test_pca  = pca.transform(X_test[num_cols])
 
        # Colonnes non numériques (binaires OHE) qu'on conserve
        non_num_cols = [c for c in X_train.columns if c not in num_cols]
 
        X_train_out = pd.concat([
            pd.DataFrame(X_train_pca, columns=pca_cols),
            X_train[non_num_cols].reset_index(drop=True)
        ], axis=1)
        X_test_out = pd.concat([
            pd.DataFrame(X_test_pca,  columns=pca_cols),
            X_test[non_num_cols].reset_index(drop=True)
        ], axis=1)
 
        print(f"\n   → Features REMPLACÉES : X_train={X_train_out.shape} | X_test={X_test_out.shape}")
        print(f"   ⚠️  Interprétabilité réduite (PC1, PC2 ≠ features métier)")
 
    else:
        # Mode analyse seulement — données inchangées
        X_train_out = X_train
        X_test_out  = X_test
        print(f"\n   → Mode analyse : données originales conservées ✅")
        print(f"   → PCA sauvegardée pour visualisation dans les notebooks")
 
    # ── Sauvegarde ──
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pca, os.path.join(models_dir, 'pca.pkl'))
    joblib.dump({
        'n_components':          n_components,
        'variance_threshold':    variance_threshold,
        'variance_ratio':        pca.explained_variance_ratio_.tolist(),
        'variance_cumulative':   variance_cumulative.tolist(),
        'feature_names':         num_cols,
        'replace_features':      replace_features,
    }, os.path.join(models_dir, 'pca_metadata.pkl'))
    print(f"   💾 PCA sauvegardée → models/pca.pkl")
 
    return X_train_out, X_test_out, pca, n_components, pca.explained_variance_ratio_

# ─────────────────────────────────────────────
# PIPELINE COMPLET
# ─────────────────────────────────────────────

def run_preprocessing(
    data_filename: str,
    raw_dir: str = 'data/raw',
    processed_dir: str = 'data/processed',
    output_dir: str = 'data/train_test',
    models_dir: str = 'models',
    use_pca: bool = False,
    pca_variance: float = 0.95,
    pca_replace: bool = False
) -> dict:

    print("\n" + "=" * 60)
    print("🚀 DÉMARRAGE DU PIPELINE DE PRÉPROCESSING")
    print("=" * 60)

    # ── Création des dossiers ──
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ── Chargement RAW (NE JAMAIS MODIFIER CE FICHIER) ──
    raw_path = os.path.join(raw_dir, data_filename)
    df = load_data(raw_path)

    if df is None:
        raise ValueError(f"Impossible de charger les données : {raw_path}")

    print(f"\n📋 Shape initiale (RAW) : {df.shape}")

    # ── PIPELINE DE TRANSFORMATION ──
    df = clean_data(df)
    df = parse_dates(df)
    df = feature_engineering(df)
    


    # ── Sauvegarde dataset processed ──
    processed_path = os.path.join(processed_dir, "dataset_processed.csv")
    df.to_csv(processed_path, index=False)

    print(f"\n💾 Dataset processed sauvegardé : {processed_path}")

    # ── Séparation X / y ──
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"\n📐 Features : {X.shape[1]} | Target distribution : {y.value_counts().to_dict()}")

    # ── Train/Test Split ──
    print("\n✂️  Train/Test split (stratifié 80/20)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"   → X_train : {X_train.shape} | X_test : {X_test.shape}")

    # ── PREPROCESSING ──
    
    X_train, X_test, imputers = impute_missing(X_train, X_test)
    X_train, X_test, encoders = encode_categoricals(X_train, X_test, y_train)
    X_train, X_test, dropped_cols = remove_multicollinearity(X_train, X_test)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    X_train, y_train = balance_classes(X_train, y_train)
    # ── Étape 9 : ACP (optionnelle) ──
    pca_model = None
    n_components = None
    if use_pca:
        X_train, X_test, pca_model, n_components, _ = apply_pca(
            X_train, X_test, y_train,
            variance_threshold=pca_variance,
            replace_features=pca_replace,
            models_dir=models_dir
        )
    else:
        print("\n[9/10] 🔬 ACP — désactivée (use_pca=False)")
        print(f"   → Pour activer : run_preprocessing(..., use_pca=True)")
        print(f"   → Pour analyse seule (sans remplacer les features) : pca_replace=False")
        print(f"   → Pour remplacer les features : pca_replace=True")

    # ── SAUVEGARDE TRAIN/TEST ──
    print("\n💾 Sauvegarde train/test...")

    save_dataframe(X_train, os.path.join(output_dir, 'X_train.csv'))
    save_dataframe(X_test, os.path.join(output_dir, 'X_test.csv'))
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    # ── SAUVEGARDE MODELS ──
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(imputers, os.path.join(models_dir, 'imputers.pkl'))
    joblib.dump(encoders, os.path.join(models_dir, 'encoders.pkl'))

    joblib.dump(
        {
            'dropped_cols': dropped_cols,
            'feature_names': X_train.columns.tolist()
        },
        os.path.join(models_dir, 'metadata.pkl')
    )

    print("\n" + "=" * 60)
    print("✅ PREPROCESSING TERMINÉ AVEC SUCCÈS")
    print(f"📊 Final shape : X_train={X_train.shape} | X_test={X_test.shape}")
    print(f"📁 RAW inchangé : {raw_dir}")
    print(f"📁 PROCESSED : {processed_dir}")
    print(f"📁 TRAIN/TEST : {output_dir}")
    if pca_model and pca_replace:
        print(f"   ACP activée  : {n_components} composantes (variance ≥ {pca_variance*100:.0f}%)")
    print("=" * 60)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'imputers': imputers,
        'encoders': encoders,
    }