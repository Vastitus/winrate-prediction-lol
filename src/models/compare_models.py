"""
Vergleicht alle trainierten Modelle und zeigt deren Performance.

Dieses Script lädt alle gespeicherten Modelle und evaluiert sie
auf dem gleichen Test-Set, um einen fairen Vergleich zu ermöglichen.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_data():
    """
    Lädt das Dataset (augmentiert falls vorhanden, sonst normal).
    """
    project_root = Path(__file__).parent.parent.parent
    
    # Versuche zuerst augmentiertes Dataset
    augmented_path = project_root / "data" / "datasets" / "59334_filtered_augmented_dataset.csv"
    normal_path = project_root / "data" / "datasets" / "29668_filtered_dataset.csv"
    
    if augmented_path.exists():
        dataset_path = augmented_path
    elif normal_path.exists():
        dataset_path = normal_path
    else:
        raise FileNotFoundError(f"Kein Dataset gefunden! Erwartet: {normal_path}")
    
    df = pd.read_csv(dataset_path)
    return df


def prepare_features(df):
    """Bereitet Features vor (gleiche Funktion wie in Trainings-Scripts)."""
    exclude_cols = [
        "match_id",
        "game_version",
        "game_mode",
        "target",
        "team1_win",
        "team2_win",
        "has_complete_positions"
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].fillna(-1)
    y = df["target"]
    
    # Erstelle Gruppen für Group-basiertes Splitting
    match_ids = df["match_id"].astype(str)
    groups = match_ids.str.replace('_mirrored', '', regex=False)
    
    return X, y, feature_cols, groups


def load_model(model_name, feature_cols):
    """
    Lädt ein trainiertes Modell.
    
    Args:
        model_name: Name des Modells ("random_forest", "xgboost", "neural_network")
        feature_cols: Liste der Feature-Spalten (für Neural Network Scaler)
    
    Returns:
        Modell (und Scaler für Neural Network)
    """
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"
    
    model_path = models_dir / f"{model_name}.pkl"
    
    if not model_path.exists():
        return None, None
    
    model = joblib.load(model_path)
    
    # Für Neural Network: Lade auch Scaler
    scaler = None
    if model_name == "neural_network":
        scaler_path = models_dir / "neural_network_scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
    
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test, model_name):
    """
    Evaluiert ein Modell auf Test-Set.
    
    Returns:
        Test Accuracy
    """
    # Für Neural Network: Features normalisieren
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
    else:
        predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy


def main():
    """Hauptfunktion: Lädt alle Modelle und vergleicht sie."""
    print("=" * 60)
    print("MODELL-VERGLEICH")
    print("=" * 60)
    print()
    
    # 1. Daten laden und vorbereiten
    print("Lade Daten...")
    df = load_data()
    X, y, feature_cols, groups = prepare_features(df)
    
    # 2. Test-Set erstellen (gleiche Aufteilung wie beim Training)
    # WICHTIG: Group-basiertes Splitting wie in Trainings-Scripts
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss.split(X, y, groups))
    
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    
    print(f"Test-Set: {len(X_test)} Matches")
    print()
    
    # 3. Lade alle Modelle und evaluiere sie
    models_to_compare = [
        "random_forest",
        "xgboost",
        "neural_network"
    ]
    
    results = {}
    
    print("Lade und evaluiere Modelle...")
    print("-" * 60)
    
    for model_name in models_to_compare:
        model, scaler = load_model(model_name, feature_cols)
        
        if model is None:
            print(f"⚠ {model_name.upper()}: Modell nicht gefunden (noch nicht trainiert)")
            continue
        
        accuracy = evaluate_model(model, scaler, X_test, y_test, model_name)
        results[model_name] = accuracy
        
        print(f"✓ {model_name.upper()}: {accuracy*100:.2f}% Accuracy")
    
    print("-" * 60)
    print()
    
    # 4. Zeige Vergleich
    if not results:
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
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("Ranking (beste zuerst):")
    print("-" * 60)
    for rank, (model_name, accuracy) in enumerate(sorted_results, 1):
        model_display = {
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
            "neural_network": "Neural Network"
        }.get(model_name, model_name)
        
        print(f"{rank}. {model_display:20s}: {accuracy*100:6.2f}% Accuracy")
    
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


if __name__ == "__main__":
    main()
