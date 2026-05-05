# League of Legends Win Rate Prediction

Bachelorprojekt. Winrate calculator basierend auf gewählten Champions.

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
│   ├── raw/              # Rohe API-Daten (JSON)
│   └── datasets/         # Verarbeitete Datasets (CSV)
├── src/
│   ├── data_collection/  # API-Wrapper und Match-Sammlung
│   ├── preprocessing/    # Feature Engineering (Winrates, Augmentation)
│   └── models/           # ML-Modelle und Preprocessing
├── scripts/              # Ausführbare Skripte (Daten sammeln, Sanity Checks)
├── models/               # Gespeicherte Modelle (.pkl) und Features
├── notebooks/            # Jupyter Notebooks für Visualisierung
└── config/               # Riot API Konfiguration
```

## Setup

### 1. Repository klonen

```bash
git clone <repository-url>
cd lol-winrate-prediction
```

### 2. Python Dependencies installieren

**Alle Dependencies auf einmal installieren:**

```bash
pip install pandas numpy scikit-learn xgboost joblib requests python-dotenv matplotlib seaborn notebook ipykernel
```

**Oder einzeln installieren:**

**Machine Learning:**
- `pandas` - Datenverarbeitung
- `numpy` - Numerische Operationen
- `scikit-learn` - ML-Modelle (Random Forest, Neural Network)
- `xgboost` - XGBoost Modell
- `joblib` - Modell-Speicherung

**API & Daten-Sammlung:**
- `requests` - HTTP Requests für Riot API
- `python-dotenv` - Umgebungsvariablen (.env Datei)

**Visualisierung:**
- `matplotlib` - Plots
- `seaborn` - Statistische Visualisierungen

**Jupyter Notebook:**
- `notebook` - Jupyter Notebook
- `ipykernel` - Python Kernel für Jupyter

### 4. Datasets herunterladen

Die Datasets sind zu groß für GitHub. Du kannst sie jetzt direkt per Script laden:

```bash
# Direkte URL
python scripts/download_dataset.py --url "<DOWNLOAD_URL>" --output data/datasets/new_matches_dataset.csv

# Oder Google Drive File-ID
python scripts/download_dataset.py --gdrive-id "<FILE_ID>" --output data/datasets/new_matches_dataset.csv
```

### 5. Preprocessing-Pipeline mit neuem Dataset

Alle Verarbeitungsskripte unterstützen jetzt flexible Input/Output-Pfade:

```bash
# 1) Filtern
python scripts/filter_dataset.py --input data/datasets/new_matches_dataset.csv --output data/datasets/new_filtered_dataset.csv --min-matches 50

# 2) Augmentieren
python src/preprocessing/augment_dataset.py --input data/datasets/new_filtered_dataset.csv --output data/datasets/new_filtered_augmented_dataset.csv

# 3) Winrate-Features
python src/preprocessing/add_champion_winrates.py --input data/datasets/new_filtered_augmented_dataset.csv --output data/datasets/new_with_winrates.csv

# 4) Champselect-Features
python src/preprocessing/add_champselect_features.py --input data/datasets/new_filtered_augmented_dataset.csv --output data/datasets/new_champselect_features.csv
```

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

### Sanity Check (Pipeline-Validierung)
```bash
python scripts/sanity_check_synthetic_labels.py
```

### Visualisierung (Jupyter Notebook)
```bash
jupyter notebook notebooks/lol_champselect_eda_and_model_eval.ipynb
```

## V2 - BA-Experimente

Der neue Stand für die Bachelorarbeit liegt unter `v2/`. Legacy-Code in `src/` und `scripts/` bleibt zur Referenz unverändert (siehe `legacy/README.md`).

### Dataset bauen (High-Elo, Queue 420, mit Timeline-Diffs @ Min 7 / Min 15)

```bash
python v2/scripts/build_match_dataset_v2.py --target-matches 5000 --output data/datasets/v2_match_dataset_5000.csv
```

Failsafes: Checkpoints (CSV-Append), Resume, Cooldown bei leeren Regionen, no-progress timeout.

### Experimente (Pregame vs. Min 7 vs. Min 15)

```bash
jupyter notebook v2/notebooks/model_experiments.ipynb
```

Trainiert pro Condition Random Forest, XGBoost und MLP und schreibt die Plots nach `v2/notebooks/charts/`.
