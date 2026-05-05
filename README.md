# ChurnSight — Prédiction du Churn Client

Projet de Machine Learning pour prédire si un client va churner (quitter) ou non,
à partir de ses données comportementales (RFM, ancienneté, satisfaction...).

## Arborescence

```
projet_ml_retail/
├── data/
│   ├── raw/                  ← CSV original (ne pas modifier)
│   ├── processed/            ← Données nettoyées intermédiaires
│   └── train_test/           ← X_train, X_test, y_train, y_test (généré)
├── notebooks/                ← Exploration Jupyter (EDA visuelle)
├── src/                      ← Scripts de production
│   ├── utils.py              ← Chargement + rapport EDA
│   ├── preprocessing.py      ← Pipeline de préparation (8 étapes)
│   ├── train_model.py        ← Entraînement + évaluation + GridSearch
│   └── predict.py            ← API JSON Flask (endpoints REST)
├── app/                      ← Interface web Flask
│   ├── app.py                ← Routes web (formulaire + dashboard)
│   └── templates/            ← Pages HTML (index, result, dashboard)
├── models/                   ← Modèles .pkl sauvegardés (généré)
├── reports/                  ← Visualisations et rapports
├── main.py                   ← Point d'entrée (lance tout)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Lancer le pipeline complet (preprocessing + training)
```bash
python main.py
```

### 2. Démarrer l'API Flask
```bash
python app/app.py
# → http://localhost:5000/health
# → http://localhost:5000/predict
```

## Pipeline de preprocessing

| Étape | Description |
|-------|-------------|
| 1. Nettoyage | Suppression des colonnes inutiles et data leakage, correction des valeurs aberrantes |
| 2. Parsing dates | `RegistrationDate` → `RegMonth`, `RegWeekday`, `RegIsWeekend`, `AccountAgeDays`, `IsNewClient`, `RegSeason` |
| 3. Feature Engineering | 5 nouvelles features : `AvgBasketValue`, `CancellationRate`, `SpendingVolatility`, `EngagementScore`, `ProductsPerTransaction` |
| 4. Imputation | Médiane (numériques), mode (catégorielles), KNN pour `SatisfactionScore` et `AvgDaysBetweenPurchases` |
| 5. Encodage | Ordinal (`SpendingCategory`, `BasketSizeCategory`) + Target Encoding (`Region`) + One-Hot (`Gender`, `AccountStatus`, `PreferredTimeOfDay`, `WeekendPreference`, `ProductDiversity`, `RegSeason`) |
| 6. Multicolinéarité | Suppression des features avec \|r\| > 0.85 (garde celle avec la plus grande variance) |
| 7. Normalisation | RobustScaler sur les features numériques continues (colonnes binaires exclues) |
| 8. Rééquilibrage | SMOTE sur X_train pour corriger le déséquilibre des classes |
| 9. Sauvegarde | CSV dans `data/train_test/`, transformers dans `models/` |

## Colonnes supprimées (data leakage / redondance)

Les colonnes suivantes sont retirées avant l'entraînement car elles encodent directement
ou indirectement la variable cible, ou sont redondantes :

`CustomerID`, `LastLoginIP`, `NewsletterSubscribed`, `UniqueInvoices`,
`CancelledTransactions`, `Country`, `PreferredMonth`, `ChurnRiskCategory`,
`CustomerType`, `RFMSegment`, `AgeCategory`, `FirstPurchaseDaysAgo`,
`FavoriteSeason`, `CustomerTenureDays`, `LoyaltyLevel`, `UniqueCountries`,
`Age`, `Recency`

## Résultats

| Modèle | F1-Score | AUC-ROC | CV F1 |Recall|
|--------|----------|---------|-------|-------|
| Logistic Regression | 0.6649  | 0.8453 | 0.8257 |0.8866|
| Random Forest | 0.6843 | 0.8738 | 0.8529 |0.8900|
| **XGBoost** ← meilleur | **0.7512** | **0.8947** | **0.8651** |**0.8500**|

Le modèle final est sélectionné par comparaison entre la version GridSearchCV et la version optimisée Optuna (TPE Bayesian), en retenant celle avec le meilleur AUC-ROC.

## API Flask — Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | Vérifier que l'API est opérationnelle |
| `GET` | `/model/info` | Informations sur le modèle chargé |
| `POST` | `/predict` | Prédiction pour un client |
| `POST` | `/predict/batch` | Prédiction pour plusieurs clients |

Le seuil de classification optimal est **0.285** (identifié sur la courbe Precision-Recall).

## Technologies

- Python 3.11+
- scikit-learn, XGBoost, imbalanced-learn (SMOTE), category_encoders
- Optuna (optimisation bayésienne TPE)
- Flask (API REST)
- joblib (sauvegarde modèles)
- matplotlip (Visualisation des données et des performances du
modèle)
-category-encoders (Encodage des variables catégorielles)
