"""
app/app.py — Application Flask principale
Interface web pour prédire le churn d'un client.

Routes :
  GET  /              → Page d'accueil + formulaire de prédiction
  POST /predict       → Résultat de prédiction pour un client
  GET  /dashboard     → Tableau de bord des statistiques globales
  GET  /api/predict   → API JSON (pour intégration externe)
  GET  /api/health    → Health check
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Ajouter la racine du projet au path pour importer src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

# ─────────────────────────────────────────────
# CHARGEMENT DES ARTEFACTS ML
# ─────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_artifacts():
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
        else:
            artifacts[key] = None
    return artifacts

ARTIFACTS = load_artifacts()


def predict_client(form_data: dict) -> dict:
    """
    Prend les données brutes du formulaire, les prépare
    et retourne la prédiction + probabilité.
    """
    model    = ARTIFACTS.get('model')
    metadata = ARTIFACTS.get('metadata')

    if model is None:
        return {'error': 'Modèle non chargé. Lance main.py d\'abord.'}

    # Construire le DataFrame avec les features attendues
    feature_names = metadata['feature_names'] if metadata else []
    row = {feat: 0 for feat in feature_names}

    # Remplir avec les valeurs du formulaire
    for key, val in form_data.items():
        if key in row:
            try:
                row[key] = float(val)
            except (ValueError, TypeError):
                row[key] = 0

    X = pd.DataFrame([row])

    # Prédiction
    prediction  = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    # Niveau de risque
    if probability >= 0.75:
        risk_level = "Critique"
        risk_color = "#ef4444"
    elif probability >= 0.50:
        risk_level = "Élevé"
        risk_color = "#f97316"
    elif probability >= 0.25:
        risk_level = "Moyen"
        risk_color = "#eab308"
    else:
        risk_level = "Faible"
        risk_color = "#22c55e"

    return {
        'prediction':   prediction,
        'label':        'Oui — Client à risque' if prediction == 1 else 'Non — Client fidèle',
        'probability':  round(probability * 100, 1),
        'risk_level':   risk_level,
        'risk_color':   risk_color,
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    """Page d'accueil avec le formulaire de prédiction."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Reçoit le formulaire, calcule la prédiction, affiche le résultat."""
    if request.is_json:
        form_data = request.get_json()
    else:
        form_data = request.form.to_dict()
    result = predict_client(form_data)
    return jsonify(result)


@app.route('/dashboard')
def dashboard():
    """Tableau de bord : statistiques globales sur les données."""
    stats = {}
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw',
                             'retail_customers_COMPLETE_CATEGORICAL.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Nettoyage des marqueurs Git si présents
        if df.columns[0].startswith('<'):
            df = pd.read_csv(data_path, skiprows=1)

        stats = {
            'total_clients':   len(df),
            'churn_count':     int(df['Churn'].sum()) if 'Churn' in df.columns else 0,
            'churn_rate':      round(df['Churn'].mean() * 100, 1) if 'Churn' in df.columns else 0,
            'avg_recency':     round(df['Recency'].mean(), 1) if 'Recency' in df.columns else 0,
            'avg_frequency':   round(df['Frequency'].mean(), 1) if 'Frequency' in df.columns else 0,
            'avg_monetary':    round(df['MonetaryTotal'].mean(), 0) if 'MonetaryTotal' in df.columns else 0,
        }
    model = ARTIFACTS.get('model')
    stats['model_type'] = type(model).__name__ if model else 'Non chargé'
    return render_template('dashboard.html', stats=stats)


# ── API JSON (pour appels externes ou tests) ──

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'status': 'ok',
        'model_loaded': ARTIFACTS.get('model') is not None
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not request.is_json:
        return jsonify({'error': 'Content-Type: application/json requis'}), 400
    data   = request.get_json()
    result = predict_client(data)
    return jsonify(result)


# ─────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("\n🚀 Application Flask — Prédiction Churn")
    print("   http://localhost:5000")
    print("   http://localhost:5000/dashboard\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
