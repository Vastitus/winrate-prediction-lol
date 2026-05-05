# V2 Arbeitsbereich

Aktueller Stand für die Bachelorarbeit. Legacy-Dateien in `src/` und `scripts/` bleiben unverändert.

## Struktur

- `v2/scripts/` - ausführbare Skripte (Datensammlung, Pipeline)
- `v2/src/` - neuer Python-Code (Module)
- `v2/notebooks/` - Experiment-Notebook + gespeicherte Plots (`charts/`)

## Dataset bauen

High-Elo (Challenger / GM / Master), Queue 420, faire Verteilung über alle Regionen,
Match-Details + Timeline (Gold/Kills/Tower/Dragon-Diff @ Min 7 und Min 15):

```bash
python v2/scripts/build_match_dataset_v2.py --target-matches 5000 --output data/datasets/v2_match_dataset_5000.csv
```

Wichtige Optionen:

- `--resume` - bestehende CSV weiterführen statt neu starten
- `--checkpoint-every N` - alle N Zeilen zwischenspeichern (default 25, append-only)
- `--no-progress-timeout-seconds N` - Abbruch wenn N Sekunden lang keine neue Zeile (default 900)
- `--parallel` - Routing-Threads (weniger fair bei kleinem n, schneller bei großem n)

## Experimente

Drei Conditions (Pregame / Pregame + Min 7 / Pregame + Min 15) x drei Modelle (RF / XGB / NN):

```bash
jupyter notebook v2/notebooks/model_experiments.ipynb
```

Plots werden automatisch nach `v2/notebooks/charts/` als PNG gespeichert.
