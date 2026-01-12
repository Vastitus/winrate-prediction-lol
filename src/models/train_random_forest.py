"""Trainiert Random Forest Modell für Win-Rate Vorhersage."""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib

from src.models.preprocessing import infer_feature_spec, make_preprocessor


def load_data():
    """Lädt Dataset (priorisiert champselect > winrates > augmented)."""
    project_root = Path(__file__).parent.parent.parent
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


def prepare_features(df, exclude_bans=False):
    """Bereitet Features vor: entfernt Metadaten, One-Hot encodiert Champion-IDs."""
    exclude_cols = [
        "match_id", "game_version", "game_mode", "queue_id", "game_duration",
        "target", "team1_win", "team2_win", "has_complete_positions"
    ]
    if exclude_bans:
        exclude_cols.extend([col for col in df.columns if 'ban' in col])

    # Synergy-Features sind bewusst ausgeschlossen (zu komplex/zu noisy).
    exclude_cols.extend([col for col in df.columns if "synergy" in col.lower()])

    spec = infer_feature_spec(df, exclude_cols=exclude_cols)
    X = df[spec.feature_cols].copy()
    y = df["target"].astype(int)
    match_ids = df["match_id"].astype(str)
    groups = match_ids.str.replace('_mirrored', '', regex=False)

    return X, y, spec.feature_cols, spec, groups


def split_data(X, y, groups):
    """Teilt Daten in Train/Val/Test (70%/15%/15%) mit Group-basiertem Splitting."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train_val = X.iloc[train_val_idx]
    X_test = X.iloc[test_idx]
    y_train_val = y.iloc[train_val_idx]
    y_test = y.iloc[test_idx]
    groups_train_val = groups.iloc[train_val_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=42)
    train_idx, val_idx = next(gss2.split(X_train_val, y_train_val, groups_train_val))
    
    return (X_train_val.iloc[train_idx], X_train_val.iloc[val_idx], X_test,
            y_train_val.iloc[train_idx], y_train_val.iloc[val_idx], y_test)

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


def _build_model(spec, rf_params: dict):
    pre = make_preprocessor(spec)
    rf = RandomForestClassifier(**rf_params)
    return Pipeline(steps=[("preprocess", pre), ("model", rf)])


def train_model(X_train, y_train, X_val, y_val, spec, tune: bool = False, trials: int = 10, seed: int = 42):
    """Trainiert Random Forest Modell (optional: kleine Hyperparameter-Suche)."""
    base_params = dict(
        n_estimators=400,
        max_depth=10,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
        class_weight=None,
    )

    if not tune:
        model = _build_model(spec, base_params)
        model.fit(X_train, y_train)
        m_train = _eval_metrics(model, X_train, y_train)
        m_val = _eval_metrics(model, X_val, y_val)
        print(f"Train: {m_train['acc']:.2%} | Val: {m_val['acc']:.2%} | Val AUC: {m_val['auc']:.4f} | Val LogLoss: {m_val['logloss']:.4f}")
        return model

    rng = np.random.RandomState(seed)
    space = {
        "n_estimators": [250, 400, 600, 900],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_split": [10, 15, 25, 40],
        "min_samples_leaf": [5, 8, 12, 20],
        "max_features": ["sqrt", 0.3, 0.5],
        "class_weight": [None, "balanced"],
    }
    keys = list(space.keys())

    best = None
    best_params = None
    best_score = float("inf")  # logloss lower is better

    print(f"[TUNE] RandomForest: {trials} trials (score=val logloss)")
    try:
        for i in range(int(trials)):
            params = dict(base_params)
            for k in keys:
                params[k] = space[k][int(rng.randint(0, len(space[k])))]
            model = _build_model(spec, params)
            model.fit(X_train, y_train)
            m_val = _eval_metrics(model, X_val, y_val)
            score = m_val["logloss"]
            print(f"[TUNE] {i+1:02d}/{trials}  val_acc={m_val['acc']:.4f}  val_auc={m_val['auc']:.4f}  val_logloss={m_val['logloss']:.4f}  params={{'depth':{params['max_depth']}, 'n':{params['n_estimators']}, 'leaf':{params['min_samples_leaf']}, 'split':{params['min_samples_split']}, 'maxfeat':{params['max_features']}, 'cw':{params['class_weight']}}}")
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
    return best


def evaluate_model(model, X_test, y_test):
    """Evaluiert Modell auf Test-Set."""
    pred = model.predict(X_test)
    test_acc = float(accuracy_score(y_test, pred))
    print(f"Test Accuracy: {test_acc:.2%}")
    print(classification_report(y_test, pred))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        print(f"Test AUC: {_safe_auc(y_test, proba[:, 1]):.4f}")
        print(f"Test LogLoss: {_safe_logloss(y_test, proba):.4f}")
    return test_acc


def save_model(model, feature_cols):
    """Speichert Modell und Features."""
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, models_dir / "random_forest.pkl")
    (models_dir / "random_forest_features.txt").write_text("\n".join(feature_cols))

    # Save transformed feature names (after OneHotEncoding) for interpretability.
    try:
        feat_out = model.named_steps["preprocess"].get_feature_names_out()
        (models_dir / "random_forest_features_transformed.txt").write_text("\n".join(map(str, feat_out)))
    except Exception:
        pass


def main(exclude_bans=False, tune: bool = False, trials: int = 20):
    """Hauptfunktion: lädt Daten, trainiert Modell, evaluiert."""
    df = load_data()
    X, y, feature_cols, spec, groups = prepare_features(df, exclude_bans=exclude_bans)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, groups)
    
    model = train_model(X_train, y_train, X_val, y_val, spec, tune=tune, trials=trials)
    test_acc = evaluate_model(model, X_test, y_test)
    save_model(model, feature_cols)
    
    return model, test_acc


def compare_with_without_bans():
    """Vergleicht Modell mit und ohne Bans."""
    model_with_bans, acc_with_bans = main(exclude_bans=False)
    model_without_bans, acc_without_bans = main(exclude_bans=True)
    print(f"Mit Bans: {acc_with_bans:.2%} | Ohne Bans: {acc_without_bans:.2%}")
    return model_with_bans, model_without_bans, acc_with_bans, acc_without_bans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", "-c", action="store_true", help="Vergleicht mit/ohne Bans (2 Trainingsläufe)")
    parser.add_argument("--no-bans", "-n", action="store_true", help="Trainiere ohne Bans (Default)")
    parser.add_argument("--with-bans", action="store_true", help="Trainiere mit Bans")
    parser.add_argument("--tune", action="store_true", help="Kleine Hyperparameter-Suche (auf Validation Split)")
    parser.add_argument("--trials", type=int, default=10, help="Anzahl Trials für --tune (Default: 10)")
    args = parser.parse_args()

    if args.compare:
        compare_with_without_bans()
    else:
        exclude_bans = True
        if args.with_bans:
            exclude_bans = False
        elif args.no_bans:
            exclude_bans = True
        main(exclude_bans=exclude_bans, tune=args.tune, trials=args.trials)
