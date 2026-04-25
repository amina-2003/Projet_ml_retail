"""
main.py — Point d'entrée du projet
Lance le pipeline complet : preprocessing → training → évaluation
"""


from src.preprocessing import run_preprocessing
from src.train_model import run_training

DATA_PATH   = 'retail_customers_COMPLETE_CATEGORICAL.csv'
DATA_DIR    = 'data/raw'
PROCESSED_DIR = 'data/processed'
OUTPUT_DIR  = 'data/train_test'
MODELS_DIR  = 'models'

if __name__ == '__main__':
    # Étape 1 : Préprocessing
    run_preprocessing(DATA_PATH, DATA_DIR, PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR)

    # Étape 2 : Entraînement
    results = run_training(OUTPUT_DIR, MODELS_DIR)

    print(f"\n🏆 Pipeline terminé. Modèle final : {results['best_model_name']}")
    print(f"   AUC-ROC : {results['metrics']['roc_auc']}")
    print(f"   F1      : {results['metrics']['f1']}")
