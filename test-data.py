from src.utils import load_data

# Chemin vers votre fichier (assurez-vous qu'il est dans data/raw/)
DATA_PATH = "data/raw/retail_customers_COMPLETE_CATEGORICAL.csv"

# 1. Chargement
df = load_data(DATA_PATH)

if df is not None:
    # 2. Aperçu des premières lignes
    print("\n--- Aperçu des 5 premières lignes ---")
    print(df.head())

    # 3. Statistiques descriptives pour les 52 features
    print("\n--- Statistiques descriptives (numériques) ---")
    print(df.describe())

    # 4. Analyse des valeurs manquantes (Point critique du projet)
    print("\n--- Analyse des valeurs manquantes (NaN) ---")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0]) # Affiche seulement les colonnes avec des vides

    # 5. Vérification du déséquilibre de la classe 'Churn'
    if 'Churn' in df.columns:
        print("\n--- Répartition de la cible (Churn) ---")
        print(df['Churn'].value_counts(normalize=True))