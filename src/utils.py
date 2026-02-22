import pandas as pd

def load_data(file_path):
    """
    Charge le dataset CSV et affiche les dimensions de base.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Chargement réussi : {df.shape[0]} lignes et {df.shape[1]} colonnes.")
        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement du fichier : {e}")
        return None