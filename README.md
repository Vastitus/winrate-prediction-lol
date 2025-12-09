# League of Legends Win Rate Prediction

Bachelorprojekt. Winrate calculator basierend auf gewählten Champions.

## Projektübersicht

Dieses Projekt entwickelt Machine Learning Modelle zur Vorhersage der Gewinnwahrscheinlichkeit von League of Legends Matches basierend auf Champions picked.

## Technischer Stack

- **API**: Riot Games API
- **ML-Modelle**: Random Forest, XGBoost, Neural Network?
- **Sprache**: Python
- **Daten**: ~60k Matches von Spielern in Master+ ELO.

## Projektstruktur

```
lol-winrate-prediction/
├── data/
│   ├── raw/              # Rohe API-Daten
│   ├── processed/        # Verarbeitete Datensätze
│   └── datasets/         # Finale Datasets (CSV, Parquet, etc.)
├── src/
│   ├── data_collection/  # Skripte für API-Abfragen
│   ├── preprocessing/    # Datenverarbeitung
│   ├── models/           # ML-Modelle
│   └── visualization/    # Visualisierungen
├── notebooks/            # Jupyter Notebooks für Experimente
├── config/               # Konfigurationsdateien
├── requirements.txt      # Python Dependencies
└── README.md
```

## Setup

### 1. Repository klonen

```bash
git clone <repository-url>
cd lol-winrate-prediction
```

### 2. Virtual Environment erstellen

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 4. Riot API Key konfigurieren

Erstelle eine `.env` Datei im Root-Verzeichnis:

```
RIOT_API_KEY=dein_api_key_hier
```

**Wichtig**: Die `.env` Datei ist in `.gitignore` und wird nicht ins Repository hochgeladen!

## Ressourcen

- [Riot Games API Documentation](https://developer.riotgames.com/)
- [League of Legends Wiki](https://leagueoflegends.fandom.com/)

