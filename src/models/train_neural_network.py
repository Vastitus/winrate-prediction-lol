"""Trainiert Neural Network (MLP) für Win-Rate Vorhersage."""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

# Ensure project root is on sys.path so `import src...` works when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib

from src.models.preprocessing import infer_feature_spec, make_preprocessor


def load_data():
    """
    Lädt das Dataset (augmentiert falls vorhanden, sonst normal).
    
    Data Augmentation verdoppelt die Datenmenge durch Team-Spiegelung,
    was zu besserer Generalisierung führt.
    """
    project_root = Path(__file__).parent.parent.parent
    
    # Standard: Dataset mit Winrates (liefert in deinem Setup aktuell die beste Accuracy)
    # Optional: Champ-Select Feature-Dataset (Winrates + Matchups + Synergy) ist experimentell.
    champselect_path = project_root / "data" / "datasets" / "59334_champselect_features.csv"
    winrates_path = project_root / "data" / "datasets" / "59334_with_winrates.csv"
    augmented_path = project_root / "data" / "datasets" / "59334_filtered_augmented_dataset.csv"
    if champselect_path.exists():
        return pd.read_csv(champselect_path)
    elif winrates_path.exists():
        return pd.read_csv(winrates_path)
    elif augmented_path.exists():
        return pd.read_csv(augmented_path)
    else:
        raise FileNotFoundError("Kein Dataset gefunden!")


def prepare_features(df, exclude_bans=True):
    """
    Bereitet Features für Training vor.
    
    Neural Networks profitieren von normalisierten Features.
    
    Args:
        exclude_bans: Wenn True, werden Ban-Features ausgeschlossen (nur Champion-Picks)
    
    Returns:
        X: Features
        y: Target-Variable
        feature_cols: Liste der Feature-Spalten
        groups: Gruppen für Group-basiertes Splitting (Basis-Match-IDs)
    """
    exclude_cols = [
        "match_id",
        "game_version",
        "game_mode",
        "queue_id",           # Immer gleiche ID, nicht relevant
        "game_duration",      # Data Leakage! Spiellänge ist erst NACH dem Spiel bekannt
        "target",
        "team1_win",
        "team2_win",
        "has_complete_positions"
    ]
    
    # Wenn exclude_bans=True, füge alle Ban-Spalten hinzu
    if exclude_bans:
        ban_cols = [col for col in df.columns if 'ban' in col]
        exclude_cols.extend(ban_cols)
        print(f"[INFO] Bans ausgeschlossen: {len(ban_cols)} Ban-Features entfernt")

    # Synergy-Features sind bewusst ausgeschlossen (zu komplex/zu noisy).
    exclude_cols.extend([col for col in df.columns if "synergy" in col.lower()])

    spec = infer_feature_spec(df, exclude_cols=exclude_cols)
    feature_cols = spec.feature_cols
    X = df[feature_cols].copy()
    y = df["target"].astype(int)
    
    # Erstelle Gruppen für Group-basiertes Splitting
    match_ids = df["match_id"].astype(str)
    groups = match_ids.str.replace('_mirrored', '', regex=False)
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Eindeutige Match-Gruppen: {groups.nunique()}")
    
    return X, y, feature_cols, spec, groups


def split_data(X, y, groups):
    """
    Teilt Daten in Trainings-, Validierungs- und Test-Sets.
    
    WICHTIG: Verwendet Group-basiertes Splitting, damit Original und
    gespiegelte Version eines Matches immer im gleichen Set bleiben.
    Das verhindert Data Leakage.
    """
    # Zuerst: Training + Validation vs. Test (70% vs. 15%)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train_val = X.iloc[train_val_idx]
    X_test = X.iloc[test_idx]
    y_train_val = y.iloc[train_val_idx]
    y_test = y.iloc[test_idx]
    groups_train_val = groups.iloc[train_val_idx]
    
    # Dann: Training vs. Validation (70% vs. 15%)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=42)
    train_idx, val_idx = next(gss2.split(X_train_val, y_train_val, groups_train_val))
    
    X_train = X_train_val.iloc[train_idx]
    X_val = X_train_val.iloc[val_idx]
    y_train = y_train_val.iloc[train_idx]
    y_val = y_train_val.iloc[val_idx]
    
    print(f"\nDaten-Aufteilung:")
    print(f"  Training:   {len(X_train)} Matches (70%)")
    print(f"  Validation: {len(X_val)} Matches (15%)")
    print(f"  Test:       {len(X_test)} Matches (15%)")
    print(f"  (Group-basiert: Original + gespiegelte Version bleiben zusammen)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def _safe_auc(y_true, proba_pos):
    try:
        return float(roc_auc_score(y_true, proba_pos))
    except Exception:
        return float("nan")


def _safe_logloss(y_true, proba):
    try:
        return float(log_loss(y_true, proba, labels=[0, 1]))
    except Exception:
        return float("nan")


def _eval_metrics(model, X, y):
    pred = model.predict(X)
    acc = float(accuracy_score(y, pred))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        ll = _safe_logloss(y, proba)
        auc = _safe_auc(y, proba[:, 1])
    else:
        ll, auc = float("nan"), float("nan")
    return {"acc": acc, "logloss": ll, "auc": auc}


def _build_model(spec, mlp_params: dict):
    pre = make_preprocessor(spec)
    mlp = MLPClassifier(**mlp_params)
    return Pipeline(steps=[("preprocess", pre), ("model", mlp)])


def train_model(X_train, y_train, X_val, y_val, spec, tune: bool = False, trials: int = 10, seed: int = 42):
    """
    Trainiert Neural Network Modell.
    
    MLPClassifier (Multi-Layer Perceptron):
    - hidden_layer_sizes=(100, 50): 2 Hidden Layers mit 100 und 50 Neuronen
    - max_iter=500: Maximale Training-Iterationen (Epochs)
    - learning_rate_init=0.001: Anfängliche Lernrate
    - early_stopping: Stoppt wenn Validation nicht besser wird
    """
    print("\n" + "=" * 60)
    print("TRAINING NEURAL NETWORK")
    print("=" * 60)

    base_params = dict(
        hidden_layer_sizes=(100, 50),
        max_iter=800,
        learning_rate_init=0.001,
        alpha=1e-3,  # a bit stronger regularization than default
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
    )

    if not tune:
        model = _build_model(spec, base_params)
        print("Trainiere Modell...")
        print("  (Dies kann einige Minuten dauern)")
        model.fit(X_train, y_train)
        print("[OK] Training abgeschlossen")

        m_train = _eval_metrics(model, X_train, y_train)
        m_val = _eval_metrics(model, X_val, y_val)

        print(f"\nErgebnisse:")
        print(f"  Training Accuracy:   {m_train['acc']:.4f} ({m_train['acc']*100:.2f}%)")
        print(f"  Validation Accuracy: {m_val['acc']:.4f} ({m_val['acc']*100:.2f}%)")
        print(f"  Val AUC:             {m_val['auc']:.4f}")
        print(f"  Val LogLoss:         {m_val['logloss']:.4f}")
        try:
            inner = model.named_steps["model"]
            print(f"  Iterationen: {inner.n_iter_} von max. {inner.max_iter}")
        except Exception:
            pass
        return model

    rng = np.random.RandomState(seed)
    space = {
        "hidden_layer_sizes": [(64,), (128,), (128, 64), (64, 32), (100, 50)],
        "alpha": [1e-5, 1e-4, 1e-3, 5e-3, 1e-2],
        "learning_rate_init": [5e-4, 1e-3, 2e-3],
        "activation": ["relu", "tanh"],
        "batch_size": ["auto", 256, 512],
        "max_iter": [400, 800, 1200],
    }
    keys = list(space.keys())
    best = None
    best_params = None
    best_score = float("inf")

    print(f"[TUNE] MLPClassifier: {trials} trials (score=val logloss)")
    try:
        for i in range(int(trials)):
            params = dict(base_params)
            for k in keys:
                params[k] = space[k][int(rng.randint(0, len(space[k])))]
            model = _build_model(spec, params)
            print(f"[TUNE] Trial {i+1:02d}/{trials} train...")
            model.fit(X_train, y_train)
            m_val = _eval_metrics(model, X_val, y_val)
            score = m_val["logloss"]
            print(f"[TUNE] {i+1:02d}/{trials}  val_acc={m_val['acc']:.4f}  val_auc={m_val['auc']:.4f}  val_logloss={m_val['logloss']:.4f}  params={{'layers':{params['hidden_layer_sizes']}, 'alpha':{params['alpha']}, 'lr':{params['learning_rate_init']}, 'act':{params.get('activation')}, 'bs':{params.get('batch_size')}, 'max_iter':{params.get('max_iter')}}}")
            if np.isfinite(score) and score < best_score:
                best_score = score
                best = model
                best_params = params
    except KeyboardInterrupt:
        print("\n[TUNE] Abgebrochen (Ctrl+C). Verwende bestes Modell 'so far'...\n")

    if best is None:
        print("[TUNE] Fallback auf Basis-Parameter (kein gültiger LogLoss).")
        best = _build_model(spec, base_params)
        best.fit(X_train, y_train)
        return best

    m_train = _eval_metrics(best, X_train, y_train)
    m_val = _eval_metrics(best, X_val, y_val)
    print("[TUNE] Best params:", best_params)
    print(f"[TUNE] Best  Train acc={m_train['acc']:.4f} | Val acc={m_val['acc']:.4f} | Val AUC={m_val['auc']:.4f} | Val LogLoss={m_val['logloss']:.4f}")
    try:
        inner = best.named_steps["model"]
        print(f"[TUNE] Iterationen: {inner.n_iter_} von max. {inner.max_iter}")
    except Exception:
        pass
    return best


def evaluate_model(model, X_test, y_test):
    """Evaluiert Modell auf Test-Set."""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        print(f"Test AUC: {_safe_auc(y_test, proba[:, 1]):.4f}")
        print(f"Test LogLoss: {_safe_logloss(y_test, proba):.4f}")
    
    return test_acc


def save_model(model, feature_cols):
    """Speichert trainiertes Modell (inkl. Preprocessing Pipeline)."""
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "neural_network.pkl"
    joblib.dump(model, model_path)
    print(f"\n[OK] Modell gespeichert: {model_path}")

    features_path = models_dir / "neural_network_features.txt"
    with open(features_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"[OK] Features gespeichert: {features_path}")

    try:
        feat_out = model.named_steps["preprocess"].get_feature_names_out()
        (models_dir / "neural_network_features_transformed.txt").write_text("\n".join(map(str, feat_out)))
    except Exception:
        pass


def main(exclude_bans=True, tune: bool = False, trials: int = 15):
    """
    Hauptfunktion: Lädt Daten, trainiert Modell, evaluiert.
    
    Args:
        exclude_bans: Wenn True, werden Bans ausgeschlossen (nur Champion-Picks)
    """
    print("=" * 60)
    print("NEURAL NETWORK TRAINING")
    if exclude_bans:
        print("(OHNE BANS - nur Champion-Picks)")
    else:
        print("(MIT BANS)")
    print("=" * 60)
    
    df = load_data()
    X, y, feature_cols, spec, groups = prepare_features(df, exclude_bans=exclude_bans)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, groups)

    model = train_model(X_train, y_train, X_val, y_val, spec, tune=tune, trials=trials)
    test_acc = evaluate_model(model, X_test, y_test)
    save_model(model, feature_cols)
    
    print("\n" + "=" * 60)
    print("FERTIG!")
    print("=" * 60)
    print(f"Finale Test Accuracy: {test_acc*100:.2f}%")
    
    return model, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bans", "-n", action="store_true", help="Trainiere ohne Bans (Default)")
    parser.add_argument("--with-bans", action="store_true", help="Trainiere mit Bans")
    parser.add_argument("--tune", action="store_true", help="Kleine Hyperparameter-Suche (auf Validation Split)")
    parser.add_argument("--trials", type=int, default=10, help="Anzahl Trials für --tune (Default: 10)")
    args = parser.parse_args()

    exclude_bans = True
    if args.with_bans:
        exclude_bans = False
    elif args.no_bans:
        exclude_bans = True

    main(exclude_bans=exclude_bans, tune=args.tune, trials=args.trials)
