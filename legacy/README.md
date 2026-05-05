# Legacy Reference (Read-Only)

Dieser Ordner markiert den alten Projektstand als Referenz.
Die eigentlichen Legacy-Dateien bleiben an ihren aktuellen Pfaden und sollen
ab jetzt nicht mehr verändert werden.

## Legacy Python-Dateien (nicht mehr bearbeiten)

- `src/data_collection/collect_matches.py`
- `src/data_collection/riot_api.py`
- `src/preprocessing/augment_dataset.py`
- `src/preprocessing/add_champion_winrates.py`
- `src/preprocessing/add_champselect_features.py`
- `src/models/train_random_forest.py`
- `src/models/train_xgboost.py`
- `src/models/train_neural_network.py`
- `src/models/compare_models.py`
- `scripts/collect_250_matches.py`
- `scripts/collect_600_matches.py`
- `scripts/collect_5k_matches.py`
- `scripts/collect_50k_matches.py`
- `scripts/filter_dataset.py`
- `scripts/augment_dataset.py`
- `scripts/download_dataset.py`

## Legacy Notebook

- `notebooks/lol_champselect_eda_and_model_eval.ipynb`

## Ab jetzt für neue Entwicklung

Nutze nur:

- `v2/src/`
- `v2/scripts/`

So bleiben Legacy und neue Version sauber getrennt im selben Repository.
