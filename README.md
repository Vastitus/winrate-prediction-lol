# League of Legends Win Rate Prediction

Bachelorarbeit Projekt zur Vorhersage der Gewinnwahrscheinlichkeit in League of Legends Matches.

## Projektübersicht

Dieses Projekt entwickelt Machine Learning Modelle zur Vorhersage der Gewinnwahrscheinlichkeit von League of Legends Matches basierend auf:
- Champion-Auswahl (Champ Select)
- Optional: In-Game Parameter nach 10 Minuten (Gold-Differenz, Objectives, etc.)

## Technischer Stack

- **API**: Riot Games API
- **ML-Modelle**: Random Forest, XGBoost, LSTM
- **Sprache**: Python
- **Daten**: ~200k Matches von Top-Spielern

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

## Nächste Schritte

- [ ] Riot API Key beschaffen
- [ ] Daten-Sammlung implementieren
- [ ] Dataset aufbauen (~200k Matches)
- [ ] Preprocessing Pipeline entwickeln
- [ ] ML-Modelle implementieren und vergleichen
- [ ] Web-Interface für Visualisierung (optional)

## Ressourcen

- [Riot Games API Documentation](https://developer.riotgames.com/)
- [League of Legends Wiki](https://leagueoflegends.fandom.com/)

