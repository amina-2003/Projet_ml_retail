"""
predict.py — API Flask pour servir le modèle de prédiction du Churn
=======================================================================
PIPELINE DE PRÉTRAITEMENT EN PRODUCTION — miroir exact de preprocessing.py
 
Ordre des transformations appliqué lors de l'entraînement :
  1. Nettoyage de base (clip valeurs aberrantes)
  2. Parsing de la date d'inscription → features temporelles
  3. Feature Engineering (AvgBasketValue, CancellationRate, SpendingVolatility,
                          EngagementScore, ProductsPerTransaction)
     ⚠️  TransactionsPerDay utilise CustomerTenureDays (supprimé avant train) → ignoré
     ⚠️  MonetaryPerDay utilise Recency (supprimé avant train) → ignoré
  4. Imputation des valeurs manquantes (mediane, via imputers.pkl)
  5. Encodage catégoriel (OrdinalEncoder, TargetEncoder, OHE via encoders.pkl)
  6. Suppression multicolinéarité (colonnes listées dans metadata.pkl)
  7. Normalisation RobustScaler (via scaler.pkl)
  8. Alignement sur les features attendues (metadata['feature_names'])
 
Endpoints :
  GET  /health          → vérifier que l'API fonctionne
  POST /predict         → prédire le churn pour un client
  POST /predict/batch   → prédire pour plusieurs clients à la fois
  GET  /model/info      → informations sur le modèle chargé
"""
 
import os
import re
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES — doivent être identiques à preprocessing.py
# ─────────────────────────────────────────────────────────────────────────────
 
# Colonnes supprimées lors du nettoyage (ne doivent PAS arriver dans la requête)
COLS_TO_DROP = [
    'CustomerID', 'LastLoginIP', 'NewsletterSubscribed', 'UniqueInvoices',
    'CancelledTransactions', 'Country', 'PreferredMonth',
    'ChurnRiskCategory', 'CustomerType', 'RFMSegment', 'Recency',
    'AgeCategory', 'FirstPurchaseDaysAgo', 'FavoriteSeason',
    'CustomerTenureDays', 'LoyaltyLevel', 'UniqueCountries', 'Age',
]
 
# Mappings ordinaux (identiques à ORDINAL_MAPPINGS dans preprocessing.py)
ORDINAL_MAPPINGS = {
    'SpendingCategory':   ['Low', 'Medium', 'High', 'VIP'],
    'BasketSizeCategory': ['Petit', 'Moyen', 'Grand'],
}
 
# Colonnes One-Hot (identiques à NOMINAL_COLS dans preprocessing.py)
NOMINAL_COLS = [
    'PreferredTimeOfDay', 'WeekendPreference', 'ProductDiversity',
    'Gender', 'AccountStatus', 'RegSeason',
]
 
# Colonnes numériques stockées en string dans le dataset brut
NUMERIC_COLS_STORED_AS_OBJECT = [
    'MonetaryTotal', 'MonetaryAvg', 'MonetaryStd',
    'MonetaryMin', 'MonetaryMax', 'AvgQuantityPerTransaction',
    'AvgDaysBetweenPurchases', 'AvgProductsPerTransaction',
    'AvgLinesPerInvoice',
]
 
# Date de référence utilisée pendant l'entraînement pour calculer AccountAgeDays
# ⚠️  IMPORTANT : cette date doit être la même que celle vue lors du fit du scaler.
#     Elle correspond à df['RegistrationDate'].max() sur le dataset complet.
#     À ajuster si vous réentraînez sur de nouvelles données.
TRAINING_REF_DATE = datetime(2011, 12, 9)   # date max du dataset UCI Retail
 
# ─────────────────────────────────────────────────────────────────────────────
# INITIALISATION DE L'APP FLASK
# ─────────────────────────────────────────────────────────────────────────────
 
app = Flask(__name__)
MODELS_DIR = 'models'
 
 
def load_artifacts() -> dict:
    """
    Charge le modèle et tous les transformers depuis le dossier models/.
    Retourne un dictionnaire avec tous les objets nécessaires à la prédiction.
    """
    artifacts = {}
    files = {
        'model':    'best_model.pkl',
        'scaler':   'scaler.pkl',
        'imputers': 'imputers.pkl',
        'encoders': 'encoders.pkl',
        'metadata': 'metadata.pkl',
    }
    for key, filename in files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
            print(f"✅ Chargé : {filename}")
        else:
            print(f"⚠️  Introuvable : {filename}")
            artifacts[key] = None
 
    return artifacts
 
 
print("🔄 Chargement du modèle et des transformers...")
ARTIFACTS = load_artifacts()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — NETTOYAGE DE BASE
# ─────────────────────────────────────────────────────────────────────────────
 
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Miroir de clean_data() dans preprocessing.py :
    - Supprime les colonnes non-prédictives si elles arrivent dans la requête
    - Clippe SatisfactionScore à [0, 10]
    - Clippe SupportTicketsCount à [0, +∞]
    - Convertit les colonnes numériques stockées en string
    """
    df = df.copy()
 
    # Supprimer colonnes inutiles (si présentes dans la requête)
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
 
    # Correction valeurs aberrantes
    if 'SupportTicketsCount' in df.columns:
        df['SupportTicketsCount'] = df['SupportTicketsCount'].clip(lower=0)
 
    if 'SatisfactionScore' in df.columns:
        df['SatisfactionScore'] = df['SatisfactionScore'].clip(lower=0, upper=10)
 
    # Conversion numérique des colonnes qui peuvent arriver en string
    for col in NUMERIC_COLS_STORED_AS_OBJECT:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
 
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — PARSING DE LA DATE D'INSCRIPTION
# ─────────────────────────────────────────────────────────────────────────────
 
def _parse_one_date(s: str) -> pd.Timestamp:
    """
    Miroir de _parse_one() dans preprocessing.py.
    Gère les formats ISO, DD/MM/YYYY, MM/DD/YYYY, et les variantes YY.
    Convention UK (DD/MM) pour les cas ambigus.
    """
    s = str(s).strip()
 
    # ISO YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        return pd.to_datetime(s, format='%Y-%m-%d')
 
    # Slash + 4 chiffres
    if re.match(r'^\d{2}/\d{2}/\d{4}$', s):
        p1, p2, _ = s.split('/')
        if int(p1) > 12:
            return pd.to_datetime(s, format='%d/%m/%Y')
        elif int(p2) > 12:
            return pd.to_datetime(s, format='%m/%d/%Y')
        else:
            return pd.to_datetime(s, format='%d/%m/%Y')  # convention UK
 
    # Slash + 2 chiffres
    if re.match(r'^\d{2}/\d{2}/\d{2}$', s):
        p1, p2, _ = s.split('/')
        if int(p1) > 12:
            return pd.to_datetime(s, format='%d/%m/%y')
        elif int(p2) > 12:
            return pd.to_datetime(s, format='%m/%d/%y')
        else:
            return pd.to_datetime(s, format='%d/%m/%y')  # convention UK
 
    return pd.NaT
 
 
def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Miroir de parse_dates() dans preprocessing.py.
    Parse RegistrationDate et crée les features temporelles.
    Si la date est absente, toutes les features sont imputées à 0 (NaN → imputer).
    """
    df = df.copy()
 
    if 'RegistrationDate' not in df.columns:
        # Date absente → créer les features avec NaN (seront imputées)
        for feat in ['RegYear', 'RegMonth', 'RegWeekday', 'RegIsWeekend',
                     'AccountAgeDays', 'IsNewClient']:
            df[feat] = np.nan
        # RegSeason absent → sera géré comme NaN catégoriel
        df['RegSeason'] = np.nan
        return df
 
    # Parser la date
    df['RegistrationDate'] = df['RegistrationDate'].apply(_parse_one_date)
 
    ref_date = pd.Timestamp(TRAINING_REF_DATE)
 
    df['RegYear']      = df['RegistrationDate'].dt.year
    df['RegMonth']     = df['RegistrationDate'].dt.month
    # RegDay est supprimé dans preprocessing.py après création → on ne le crée pas
    df['RegWeekday']   = df['RegistrationDate'].dt.weekday
    df['RegIsWeekend'] = (df['RegWeekday'] >= 5).astype(int)
    df['AccountAgeDays'] = (ref_date - df['RegistrationDate']).dt.days
 
    df['RegSeason'] = df['RegMonth'].map({
        12: 'hiver',    1: 'hiver',    2: 'hiver',
         3: 'printemps', 4: 'printemps', 5: 'printemps',
         6: 'été',       7: 'été',       8: 'été',
         9: 'automne',  10: 'automne',  11: 'automne',
    })
 
    df['IsNewClient'] = (df['AccountAgeDays'] < 365).astype(int)
 
    df.drop(columns=['RegistrationDate'], inplace=True)
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
 
def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Miroir de feature_engineering() dans preprocessing.py.
 
    Features créées lors de l'entraînement :
      ✅ AvgBasketValue        → MonetaryTotal / (Frequency + 1)
      ✅ CancellationRate      → NegativeQuantityCount / (TotalTransactions + 1)
      ✅ SpendingVolatility    → MonetaryStd / (|MonetaryAvg| + 1)
      ✅ EngagementScore       → SatisfactionScore / (SupportTicketsCount + 1)
      ✅ ProductsPerTransaction → UniqueProducts / (TotalTransactions + 1)
 
      ❌ MonetaryPerDay       → nécessite Recency (colonne supprimée = data leakage)
      ❌ TransactionsPerDay   → nécessite CustomerTenureDays (colonne supprimée)
         Ces deux features ne sont donc PAS présentes dans les données d'entraînement
         et ne doivent PAS être créées ici.
    """
    df = df.copy()
 
    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)
 
    if 'NegativeQuantityCount' in df.columns and 'TotalTransactions' in df.columns:
        df['CancellationRate'] = df['NegativeQuantityCount'] / (df['TotalTransactions'] + 1)
 
    if 'MonetaryStd' in df.columns and 'MonetaryAvg' in df.columns:
        df['SpendingVolatility'] = df['MonetaryStd'] / (df['MonetaryAvg'].abs() + 1)
 
    if 'SatisfactionScore' in df.columns and 'SupportTicketsCount' in df.columns:
        df['EngagementScore'] = df['SatisfactionScore'] / (df['SupportTicketsCount'] + 1)
 
    if 'UniqueProducts' in df.columns and 'TotalTransactions' in df.columns:
        df['ProductsPerTransaction'] = df['UniqueProducts'] / (df['TotalTransactions'] + 1)
 
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
 
def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les imputers sauvegardés (fit sur X_train uniquement).
    imputers.pkl contient un dict avec clés : 'numeric', 'categorical', 'knn' (optionnel).
    """
    imputers = ARTIFACTS.get('imputers')
    if imputers is None:
        # Fallback : médiane locale si pas d'imputer chargé
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        return df
 
    df = df.copy()
 
    # Imputation numérique (médiane)
    if 'numeric' in imputers:
        imp = imputers['numeric']
        # Appliquer seulement sur les colonnes que l'imputer connaît
        cols = [c for c in imp.feature_names_in_ if c in df.columns]
        if cols:
            df[cols] = imp.transform(df[cols])
 
    # Imputation catégorielle (mode)
    if 'categorical' in imputers:
        imp = imputers['categorical']
        cols = [c for c in imp.feature_names_in_ if c in df.columns]
        if cols:
            df[cols] = imp.transform(df[cols])
 
    # KNN Imputer sur colonnes critiques
    if 'knn' in imputers:
        imp = imputers['knn']
        cols = [c for c in imp.feature_names_in_ if c in df.columns]
        if cols:
            df[cols] = imp.transform(df[cols])
 
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — ENCODAGE CATÉGORIEL
# ─────────────────────────────────────────────────────────────────────────────
 
def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Miroir de encode_categoricals() dans preprocessing.py.
    Utilise les encoders sauvegardés dans encoders.pkl.
 
    encoders.pkl contient un dict avec :
      - 'ordinal_SpendingCategory'   : OrdinalEncoder
      - 'ordinal_BasketSizeCategory' : OrdinalEncoder
      - 'target_encoder_region'      : TargetEncoder (category_encoders)
      - 'ohe_columns'                : list des colonnes OHE (pd.get_dummies)
    """
    encoders = ARTIFACTS.get('encoders')
    if encoders is None:
        return df
 
    df = df.copy()
 
    # --- Ordinal Encoding ---
    for col in list(ORDINAL_MAPPINGS.keys()):
        key = f'ordinal_{col}'
        if key in encoders and col in df.columns:
            enc = encoders[key]
            df[[col]] = enc.transform(df[[col]])
 
    # --- Target Encoding pour Region ---
    if 'target_encoder_region' in encoders and 'Region' in df.columns:
        te = encoders['target_encoder_region']
        df['Region'] = te.transform(df['Region'])
 
    # --- One-Hot Encoding (pd.get_dummies) ---
    # On reproduit get_dummies sur les colonnes présentes, puis on aligne
    # avec les colonnes créées lors de l'entraînement (stockées dans metadata).
    nominal_present = [c for c in NOMINAL_COLS if c in df.columns]
    if nominal_present:
        df = pd.get_dummies(df, columns=nominal_present, drop_first=False, dtype=int)
 
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 + 7 — SUPPRESSION MULTICOLINÉARITÉ + NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
 
def _drop_multicol_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Supprime les colonnes retirées pour multicolinéarité (metadata['dropped_cols'])
    - Applique le RobustScaler sauvegardé sur les colonnes continues
 
    Le scaler a été fit uniquement sur X_train → on applique transform ici.
    """
    metadata = ARTIFACTS.get('metadata')
    scaler   = ARTIFACTS.get('scaler')
 
    df = df.copy()
 
    # Supprimer les colonnes multicolinéaires
    if metadata and 'dropped_cols' in metadata:
        cols_to_drop = [c for c in metadata['dropped_cols'] if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
 
    # Normalisation RobustScaler
    if scaler is not None:
        # Le scaler ne connaît que les colonnes numériques continues (pas les binaires)
        scale_cols = [c for c in scaler.feature_names_in_ if c in df.columns]
        if scale_cols:
            df[scale_cols] = scaler.transform(df[scale_cols])
 
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 8 — ALIGNEMENT FINAL SUR LES FEATURES DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
 
def _align_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligne le DataFrame sur les features exactes attendues par le modèle
    (dans le bon ordre), telles que sauvegardées dans metadata['feature_names'].
 
    - Colonnes manquantes → ajoutées avec 0 (ex : catégories OHE absentes)
    - Colonnes en trop → ignorées
    """
    metadata = ARTIFACTS.get('metadata')
    if metadata is None or 'feature_names' not in metadata:
        return df
 
    expected = metadata['feature_names']
    for feat in expected:
        if feat not in df.columns:
            df[feat] = 0
 
    return df[expected]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLET DE PRÉTRAITEMENT
# ─────────────────────────────────────────────────────────────────────────────
 
def preprocess_input(client_data: dict) -> np.ndarray:
    """
    Applique le pipeline complet de prétraitement sur un dictionnaire client.
    Respecte l'ordre exact des transformations de preprocessing.py.
 
    Args:
        client_data : dict contenant les champs bruts du client
 
    Returns:
        np.ndarray prêt pour model.predict()
    """
    df = pd.DataFrame([client_data])
 
    df = _clean(df)               # Étape 1 : nettoyage
    df = _parse_dates(df)         # Étape 2 : dates → features temporelles
    df = _feature_engineering(df) # Étape 3 : nouvelles features comportementales
    df = _impute(df)              # Étape 4 : imputation (transformers sauvegardés)
    df = _encode(df)              # Étape 5 : encodage catégoriel
    df = _drop_multicol_and_scale(df)  # Étapes 6+7 : multicolinéarité + scaling
    df = _align_features(df)      # Étape 8 : alignement final
 
    return df.values
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS DE L'API
# ─────────────────────────────────────────────────────────────────────────────
 
@app.route('/health', methods=['GET'])
def health():
    """
    Vérifie que l'API est opérationnelle.
    Retourne 'ok' si le modèle est chargé, 'degraded' sinon.
    """
    model_loaded = ARTIFACTS.get('model') is not None
    return jsonify({
        'status':       'ok' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'message':      'API de prédiction Churn opérationnelle'
                        if model_loaded else 'Modèle non chargé',
    }), 200
 
 
@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Retourne des informations sur le modèle actuellement chargé.
    """
    model    = ARTIFACTS.get('model')
    metadata = ARTIFACTS.get('metadata')
 
    if model is None:
        return jsonify({'error': 'Aucun modèle chargé'}), 503
 
    n_features    = len(metadata['feature_names']) if metadata else 'inconnu'
    dropped_cols  = metadata.get('dropped_cols', []) if metadata else []
 
    return jsonify({
        'model_type':    type(model).__name__,
        'n_features':    n_features,
        'dropped_cols':  dropped_cols,
        'target':        'Churn (0=Non, 1=Oui)',
        'threshold':     0.285,   # seuil optimal identifié sur la courbe PR
        'description':   'Prédiction du churn client — XGBoost optimisé Optuna',
    }), 200
 
 
@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédit le churn pour UN seul client.
 
    Body JSON attendu (champs bruts, comme dans le dataset) :
    {
        "Frequency": 12,
        "MonetaryTotal": 1500.0,
        "MonetaryAvg": 125.0,
        "MonetaryStd": 45.0,
        "TotalTransactions": 12,
        "NegativeQuantityCount": 1,
        "SatisfactionScore": 7,
        "SupportTicketsCount": 2,
        "AvgDaysBetweenPurchases": 22.5,
        "UniqueProducts": 8,
        "AvgLinesPerInvoice": 3.2,
        "ReturnRatio": 0.08,
        "SpendingCategory": "High",
        "BasketSizeCategory": "Moyen",
        "Region": "UK",
        "PreferredTimeOfDay": "Matin",
        "WeekendPreference": "Semaine",
        "ProductDiversity": "Explorateur",
        "Gender": "M",
        "AccountStatus": "Active",
        "RegistrationDate": "15/03/2015",
        ...
    }
 
    Réponse :
    {
        "churn_prediction":  1,
        "churn_label":       "Oui - Client à risque",
        "churn_probability": 0.8247,
        "risk_level":        "Critique"
    }
    """
    model = ARTIFACTS.get('model')
    if model is None:
        return jsonify({'error': "Modèle non disponible. Lancez d'abord train_model.py."}), 503
 
    if not request.is_json:
        return jsonify({'error': 'Content-Type doit être application/json'}), 400
 
    client_data = request.get_json()
    if not client_data:
        return jsonify({'error': 'Body JSON vide ou invalide'}), 400
 
    try:
        X = preprocess_input(client_data)
 
        prediction  = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
 
        # Niveaux de risque calibrés sur les seuils identifiés lors de l'évaluation
        if probability >= 0.75:
            risk_level = "Critique"
        elif probability >= 0.50:
            risk_level = "Élevé"
        elif probability >= 0.25:
            risk_level = "Moyen"
        else:
            risk_level = "Faible"
 
        # Prédiction avec le seuil optimal (0.285) identifié sur la courbe PR
        prediction_optimal = int(probability >= 0.285)
 
        return jsonify({
            'churn_prediction':         prediction,          # seuil standard 0.5
            'churn_prediction_optimal': prediction_optimal,  # seuil optimal 0.285
            'churn_label':   'Oui - Client à risque' if prediction == 1 else 'Non - Client fidèle',
            'churn_probability': round(probability, 4),
            'risk_level':        risk_level,
        }), 200
 
    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prédiction : {str(e)}'}), 500
 
 
@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Prédit le churn pour PLUSIEURS clients à la fois.
 
    Body JSON attendu :
    {
        "clients": [
            { "Frequency": 12, "MonetaryTotal": 1500.0, ... },
            { "Frequency": 2,  "MonetaryTotal": 80.0,  ... }
        ]
    }
 
    Réponse :
    {
        "predictions": [
            { "client_index": 0, "churn_prediction": 0, "churn_probability": 0.12,
              "risk_level": "Faible" },
            { "client_index": 1, "churn_prediction": 1, "churn_probability": 0.87,
              "risk_level": "Critique" }
        ],
        "n_clients":  2,
        "n_churners": 1,
        "churn_rate": 0.5
    }
    """
    model = ARTIFACTS.get('model')
    if model is None:
        return jsonify({'error': 'Modèle non disponible.'}), 503
 
    if not request.is_json:
        return jsonify({'error': 'Content-Type doit être application/json'}), 400
 
    data    = request.get_json()
    clients = data.get('clients', [])
 
    if not clients:
        return jsonify({'error': 'La liste "clients" est vide ou manquante'}), 400
 
    try:
        predictions_list = []
        for i, client in enumerate(clients):
            X    = preprocess_input(client)
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0][1])
 
            if prob >= 0.75:
                risk = "Critique"
            elif prob >= 0.50:
                risk = "Élevé"
            elif prob >= 0.25:
                risk = "Moyen"
            else:
                risk = "Faible"
 
            predictions_list.append({
                'client_index':             i,
                'churn_prediction':         pred,
                'churn_prediction_optimal': int(prob >= 0.285),
                'churn_probability':        round(prob, 4),
                'risk_level':               risk,
            })
 
        n_churners = sum(1 for p in predictions_list if p['churn_prediction'] == 1)
 
        return jsonify({
            'predictions': predictions_list,
            'n_clients':   len(clients),
            'n_churners':  n_churners,
            'churn_rate':  round(n_churners / len(clients), 4),
        }), 200
 
    except Exception as e:
        return jsonify({'error': f'Erreur batch : {str(e)}'}), 500
 
 
# ─────────────────────────────────────────────
# TEST MANUEL DES PRÉDICTIONS
# ─────────────────────────────────────────────
 
def tester_predictions():
    """
    Teste le pipeline de prédiction sur 3 profils clients typiques.
    Appeler directement : python predict.py
    """
 
    clients = {
        "Client VIP fidèle": {
            "Frequency": 35,
            "MonetaryTotal": 5288.63,
            "MonetaryAvg": 151.1,
            "MonetaryStd": 98.4,
            "MonetaryMin": 12.5,
            "MonetaryMax": 420.0,
            "TotalTransactions": 35,
            "NegativeQuantityCount": 2,
            "ZeroPriceCount": 0,
            "ReturnRatio": 0.057,
            "AvgQuantityPerTransaction": 5.4,
            "AvgDaysBetweenPurchases": 8.2,
            "AvgLinesPerInvoice": 9.1,
            "AvgProductsPerTransaction": 4.8,
            "UniqueProducts": 24,
            "SatisfactionScore": 9.0,
            "SupportTicketsCount": 1,
            "PreferredDayOfWeek": 3,
            "PreferredHour": 10,
            "SpendingCategory": "VIP",
            "BasketSizeCategory": "Grand",
            "Region": "UK",
            "PreferredTimeOfDay": "Matin",
            "WeekendPreference": "Semaine",
            "ProductDiversity": "Explorateur",
            "Gender": "M",
            "AccountStatus": "Active",
            "RegistrationDate": "17/07/2010",
        },
        "Client à risque de churn": {
            "Frequency": 2,
            "MonetaryTotal": 12.5,
            "MonetaryAvg": 21.25,
            "MonetaryStd": 12.3,
            "MonetaryMin": 30.0,
            "MonetaryMax": 58.5,
            "TotalTransactions": 2,
            "NegativeQuantityCount": 4,
            "ZeroPriceCount": 1,
            "ReturnRatio": 0.67,
            "AvgQuantityPerTransaction": 1.5,
            "AvgDaysBetweenPurchases": 180.0,
            "AvgLinesPerInvoice": 1.2,
            "AvgProductsPerTransaction": 1.0,
            "UniqueProducts": 2,
            "SatisfactionScore": 2.0,
            "SupportTicketsCount": 8,
            "PreferredDayOfWeek": 6,
            "PreferredHour": 20,
            "SpendingCategory": "Low",
            "BasketSizeCategory": "Petit",
            "Region": "Autre",
            "PreferredTimeOfDay": "Soir",
            "WeekendPreference": "Weekend",
            "ProductDiversity": "Spécialiste",
            "Gender": "F",
            "AccountStatus": "Suspended",
            "RegistrationDate": "",
            "IsNewClient": False,
        },
        "Client moyen occasionnel": {
            "Frequency": 7,
            "MonetaryTotal": 420.0,
            "MonetaryAvg": 60.0,
            "MonetaryStd": 22.0,
            "MonetaryMin": 20.0,
            "MonetaryMax": 110.0,
            "TotalTransactions": 7,
            "NegativeQuantityCount": 1,
            "ZeroPriceCount": 0,
            "ReturnRatio": 0.14,
            "AvgQuantityPerTransaction": 3.1,
            "AvgDaysBetweenPurchases": 45.0,
            "AvgLinesPerInvoice": 4.2,
            "AvgProductsPerTransaction": 2.8,
            "UniqueProducts": 9,
            "SatisfactionScore": 6.0,
            "SupportTicketsCount": 3,
            "PreferredDayOfWeek": 2,
            "PreferredHour": 14,
            "SpendingCategory": "Medium",
            "BasketSizeCategory": "Moyen",
            "Region": "UK",
            "PreferredTimeOfDay": "Après-midi",
            "WeekendPreference": "Semaine",
            "ProductDiversity": "Généraliste",
            "Gender": "F",
            "AccountStatus": "Active",
            "RegistrationDate": "2009-05-9",
        },
    }
 
    model = ARTIFACTS.get('model')
    if model is None:
        print("❌ Modèle non chargé — lancez d'abord main.py")
        return
 
    print("\n" + "=" * 60)
    print("🧪 TEST MANUEL DES PRÉDICTIONS")
    print("=" * 60)
 
    for nom, client in clients.items():
        print(f"\n👤 {nom}")
        print("-" * 40)
        try:
            X = preprocess_input(client)
            prediction  = int(model.predict(X)[0])
            probability = float(model.predict_proba(X)[0][1])
            pred_optimal = int(probability >= 0.285)
 
            if probability >= 0.75:   risk = "Critique 🔴"
            elif probability >= 0.50: risk = "Élevé 🟠"
            elif probability >= 0.25: risk = "Moyen 🟡"
            else:                     risk = "Faible 🟢"
 
            print(f"  Churn (seuil 0.5)   : {'OUI ⚠️' if prediction   == 1 else 'NON ✅'}")
            print(f"  Churn (seuil 0.285) : {'OUI ⚠️' if pred_optimal == 1 else 'NON ✅'}")
            print(f"  Probabilité         : {probability:.2%}")
            print(f"  Niveau de risque    : {risk}")
 
        except Exception as e:
            print(f"  ❌ Erreur : {e}")
 
    print("\n" + "=" * 60)
 
 
    
# ─────────────────────────────────────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tester_predictions()
    print("\n" + "=" * 60)
    print("🚀 DÉMARRAGE DE L'API FLASK — PRÉDICTION CHURN")
    print("=" * 60)
    print("   Endpoints disponibles :")
    print("   → GET  http://localhost:5000/health")
    print("   → GET  http://localhost:5000/model/info")
    print("   → POST http://localhost:5000/predict")
    print("   → POST http://localhost:5000/predict/batch")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)