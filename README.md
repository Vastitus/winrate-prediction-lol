# League of Legends Win Rate Prediction

Bachelorarbeit Projekt zur Vorhersage der Gewinnwahrscheinlichkeit in League of Legends Matches.

## Projektübersicht

Dieses Projekt entwickelt Machine Learning Modelle zur Vorhersage der Gewinnwahrscheinlichkeit von League of Legends Matches basierend auf:
- Champion-Auswahl (Champ Select)

## Technischer Stack

- **API**: Riot Games API
- **ML-Modelle**: Random Forest, XGBoost, Neural Network
- **Sprache**: Python
- **Daten**: ~60k Matches von Top-Spielern (Master+)

## Projektstruktur

```
lol-winrate-prediction/
├── data/
│   ├── raw/              # Rohe API-Daten von Matches 
│   └── datasets/         # Finale Datasets (CSV)
├── src/
│   ├── data_collection/  # Skripte für API-Abfragen
│   ├── preprocessing/    # Datenverarbeitung
│   └──models/           # ML-Modelle
├── notebooks/            # Jupyter Notebooks für Graphen, Heatmaps und Tabellen
├── config/               # Konfigurationsdateien, eigentlich nur für die Riot API
└── README.md
```

## Setup

### 1. Repository klonen

```bash
git clone <repository-url>
cd lol-winrate-prediction
```

### 2. Python Dependencies installieren

```bash
pip install pandas numpy scikit-learn xgboost joblib requests python-dotenv matplotlib seaborn
```

### 3. Jupyter Notebook installieren (für Visualisierung)

```bash
pip install jupyter
# oder
pip install jupyterlab
```

### 4. Datasets herunterladen

Die Datasets (59334 Matches) sind zu groß für GitHub. Die sind im Google Drive. Davor runterladen und dann hier in den Richtigen Ordner ziehen (datasets).
- `59334_filtered_augmented_dataset.csv`
- `59334_with_winrates.csv`
- `59334_champselect_features.csv`

## Verwendung

### Modelle trainieren
```bash
python src/models/train_random_forest.py
python src/models/train_xgboost.py
python src/models/train_neural_network.py
```

### Modelle vergleichen
```bash
python src/models/compare_models.py
```

### Visualisierung (Jupyter Notebook)
```bash
jupyter notebook notebooks/lol_champselect_eda_and_model_eval.ipynb
```

