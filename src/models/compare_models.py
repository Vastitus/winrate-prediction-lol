"""Vergleicht alle trainierten Modelle auf dem gleichen Test-Set."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so this script works when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import joblib


@dataclass
class ModelEval:
    name: str
    accuracy: float
    auc: Optional[float] = None
    logloss: Optional[float] = None
    feature_importances: Optional[List[Tuple[str, float]]] = None  # (feature, importance)


def load_data():
    """
    Lädt das Dataset (Winrates-Dataset falls vorhanden, sonst augmentiert/normal).
    """
    project_root = Path(__file__).parent.parent.parent
    
    # Für den Vergleich: lieber das "reichste" Dataset laden, damit alle Modelle ihre erwarteten Features finden.
    # (Ein Winrates-only Modell funktioniert auch, wenn zusätzliche Spalten vorhanden sind.)
    champselect_path = project_root / "data" / "datasets" / "59334_champselect_features.csv"
    # Danach: Dataset mit Winrates (nur Champion-Stärke)
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


def load_model(model_name):
    """
    Lädt ein trainiertes Modell und die gespeicherten Features.
    
    Args:
        model_name: Name des Modells ("random_forest", "xgboost", "neural_network")
    
    Returns:
        Modell (sklearn Pipeline), feature_cols (Liste der Input-Features)
    """
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"
    
    model_path = models_dir / f"{model_name}.pkl"
    
    if not model_path.exists():
        return None, None, None
    
    model = joblib.load(model_path)
    
    # Lade gespeicherte Features (wichtig für Feature-Matching!)
    features_path = models_dir / f"{model_name}_features.txt"
    feature_cols = None
    if features_path.exists():
        with open(features_path, "r") as f:
            feature_cols = [line.strip() for line in f.readlines() if line.strip()]
    
    return model, feature_cols


def _safe_auc(y_true, proba_pos):
    try:
        return float(roc_auc_score(y_true, proba_pos))
    except Exception:
        return None


def _safe_logloss(y_true, proba):
    try:
        return float(log_loss(y_true, proba, labels=[0, 1]))
    except Exception:
        return None


def evaluate_model(model, X_test, y_test, expected_features):
    """
    Evaluiert ein Modell auf Test-Set.
    
    Args:
        expected_features: Liste der Features, die das Modell erwartet
    
    Returns:
        Test Accuracy
    """
    # Stelle sicher, dass X_test die richtigen Features hat
    if expected_features:
        # Prüfe ob Features übereinstimmen
        missing_features = set(expected_features) - set(X_test.columns)
        extra_features = set(X_test.columns) - set(expected_features)
        
        if missing_features or extra_features:
            # Verwende nur die erwarteten Features
            X_test = X_test[[col for col in expected_features if col in X_test.columns]]

    # Modelle sind als sklearn Pipeline gespeichert (inkl. OneHotEncoding etc.)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))

    auc = None
    ll = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            ll = _safe_logloss(y_test, proba)
            auc = _safe_auc(y_test, proba[:, 1])
        except Exception:
            pass

    return accuracy, auc, ll


def _ascii_bar(value: float, min_value: float, max_value: float, width: int = 30) -> str:
    if max_value <= min_value:
        filled = width // 2
    else:
        filled = int(round((value - min_value) / (max_value - min_value) * width))
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _print_ascii_accuracy_chart(evals: List[ModelEval]) -> None:
    if not evals:
        return

    accs = [e.accuracy for e in evals]
    min_a = min(accs + [0.50])
    max_a = max(accs + [0.55])

    print("Accuracy Chart (ASCII):")
    for e in evals:
        bar = _ascii_bar(e.accuracy, min_a, max_a, width=34)
        print(f"  {e.name:14s} {bar}  {e.accuracy*100:5.2f}%")
    print()


def _try_save_plots(output_dir: Path, evals: List[ModelEval]) -> None:
    """
    Speichert optional PNGs. Im Terminal können wir keine echten Bilder rendern,
    aber wir können sie als Datei ablegen.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("[WARN] matplotlib nicht verfügbar - überspringe PNG-Plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Model comparison bar chart
    names = [e.name for e in evals]
    accs = [e.accuracy for e in evals]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(names, accs)
    ax.set_ylim(0.45, 0.60)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison (Test Accuracy)")
    for i, a in enumerate(accs):
        ax.text(i, a + 0.002, f"{a*100:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path = output_dir / "model_comparison.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] Plot gespeichert: {out_path}")

    # 2) Feature importances (falls verfügbar)
    for e in evals:
        if not e.feature_importances:
            continue
        top = e.feature_importances[:20]
        feats = [f for f, _ in top][::-1]
        vals = [v for _, v in top][::-1]

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.barh(feats, vals)
        ax.set_title(f"Top Feature Importances - {e.name}")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fpath = output_dir / f"feature_importance_{e.name.lower()}.png"
        fig.savefig(fpath, dpi=160)
        plt.close(fig)
        print(f"[OK] Plot gespeichert: {fpath}")


def main(argv: Optional[List[str]] = None):
    """Hauptfunktion: Lädt alle Modelle und vergleicht sie."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Speichert PNG Plots unter reports/")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("MODELL-VERGLEICH")
    print("=" * 60)
    print()
    
    # 1. Daten laden
    print("Lade Daten...")
    df = load_data()
    
    # 2. Test-Set erstellen (gleiche Aufteilung wie beim Training)
    # Erstelle Gruppen für Group-basiertes Splitting
    match_ids = df["match_id"].astype(str)
    groups = match_ids.str.replace('_mirrored', '', regex=False)
    
    # WICHTIG: Group-basiertes Splitting wie in Trainings-Scripts
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss.split(df, df['target'], groups))
    
    df_test = df.iloc[test_idx].copy()
    y_test = df_test['target']
    
    print(f"Test-Set: {len(df_test)} Matches")
    print()
    
    # 3. Lade alle Modelle und evaluiere sie
    models_to_compare = [
        "random_forest",
        "xgboost",
        "neural_network"
    ]
    
    evals: list[ModelEval] = []
    
    print("Lade und evaluiere Modelle...")
    print("-" * 60)
    
    for model_name in models_to_compare:
        model, expected_features = load_model(model_name)
        
        if model is None:
            print(f"[WARN] {model_name.upper()}: Modell nicht gefunden (noch nicht trainiert)")
            continue
        
        if expected_features is None:
            print(f"[WARN] {model_name.upper()}: Feature-Liste nicht gefunden")
            continue
        
        # Bereite Features für dieses spezifische Modell vor
        X_test = df_test[expected_features].copy()

        accuracy, auc, ll = evaluate_model(model, X_test, y_test, expected_features)

        # Feature importance, falls verfügbar
        importances: list[tuple[str, float]] | None = None
        if hasattr(model, "named_steps"):
            try:
                inner = model.named_steps.get("model")
                pre = model.named_steps.get("preprocess")
                if inner is not None and hasattr(inner, "feature_importances_") and pre is not None:
                    feats = list(map(str, pre.get_feature_names_out()))
                    vals = list(getattr(inner, "feature_importances_"))
                    importances = sorted(zip(feats, vals), key=lambda x: x[1], reverse=True)
            except Exception:
                importances = None

        pretty_name = {
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
            "neural_network": "Neural Network",
        }.get(model_name, model_name)

        evals.append(ModelEval(name=pretty_name, accuracy=accuracy, auc=auc, logloss=ll, feature_importances=importances))
        extra = ""
        if auc is not None:
            extra += f" | AUC {auc:.4f}"
        if ll is not None:
            extra += f" | LogLoss {ll:.4f}"
        print(f"[OK] {model_name.upper()}: {accuracy*100:.2f}% Accuracy{extra}")
    
    print("-" * 60)
    print()
    
    # 4. Zeige Vergleich
    if not evals:
        print("Keine Modelle gefunden. Trainiere zuerst die Modelle:")
        print("  python src/models/train_random_forest.py")
        print("  python src/models/train_xgboost.py")
        print("  python src/models/train_neural_network.py")
        return
    
    print("=" * 60)
    print("VERGLEICHS-ERGEBNISSE")
    print("=" * 60)
    print()
    
    # Sortiere nach Accuracy (beste zuerst)
    sorted_results = sorted(((e.name, e.accuracy) for e in evals), key=lambda x: x[1], reverse=True)
    sorted_evals = sorted(evals, key=lambda e: e.accuracy, reverse=True)

    _print_ascii_accuracy_chart(sorted_evals)
    
    print("Ranking (beste zuerst):")
    print("-" * 60)
    for rank, e in enumerate(sorted_evals, 1):
        extras = ""
        if e.auc is not None:
            extras += f" | AUC {e.auc:.4f}"
        if e.logloss is not None:
            extras += f" | LogLoss {e.logloss:.4f}"
        print(f"{rank}. {e.name:20s}: {e.accuracy*100:6.2f}% Accuracy{extras}")
    
    print("-" * 60)
    print()
    
    # Zeige Unterschiede
    if len(sorted_results) > 1:
        best_acc = sorted_results[0][1]
        best_model = sorted_results[0][0]
        
        print(f"Bester Model: {best_model.upper()} mit {best_acc*100:.2f}%")
        print()
        print("Unterschiede zum besten Modell:")
        for model_name, accuracy in sorted_results[1:]:
            diff = (best_acc - accuracy) * 100
            model_display = {
                "random_forest": "Random Forest",
                "xgboost": "XGBoost",
                "neural_network": "Neural Network"
            }.get(model_name, model_name)
            print(f"  {model_display:20s}: -{diff:.2f} Prozentpunkte")
    
    print()
    print("=" * 60)

    if args.plot:
        reports_dir = Path(__file__).parent.parent.parent / "reports"
        _try_save_plots(reports_dir, sorted_evals)


if __name__ == "__main__":
    main()
