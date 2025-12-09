"""
Trainiert ein Neural Network (Multi-Layer Perceptron) für Win-Rate Vorhersage.

Neural Networks lernen komplexe Muster durch mehrere Schichten von Neuronen.
Jede Schicht transformiert die Eingabe und lernt abstraktere Features.

Vorteile:
- Kann komplexe, nicht-lineare Muster lernen
- Gut für Interaktionen zwischen Features (z.B. Champion-Synergien)
- Flexible Architektur
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib


def load_data():
    """
    Lädt das Dataset (augmentiert falls vorhanden, sonst normal).
    
    Data Augmentation verdoppelt die Datenmenge durch Team-Spiegelung,
    was zu besserer Generalisierung führt.
    """
    project_root = Path(__file__).parent.parent.parent
    
    # Versuche zuerst augmentiertes Dataset
    augmented_path = project_root / "data" / "datasets" / "59334_filtered_augmented_dataset.csv"
    normal_path = project_root / "data" / "datasets" / "29668_filtered_dataset.csv"
    
    if augmented_path.exists():
        dataset_path = augmented_path
        print(f"✓ Nutze augmentiertes Dataset (verdoppelte Datenmenge)")
    elif normal_path.exists():
        dataset_path = normal_path
        print(f"ℹ Nutze normales Dataset (für Augmentation: python src/preprocessing/augment_dataset.py)")
    else:
        raise FileNotFoundError(f"Kein Dataset gefunden! Erwartet: {normal_path}")
    
    print(f"Lade Dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Geladen: {len(df)} Matches")
    
    return df


def prepare_features(df):
    """
    Bereitet Features für Training vor.
    
    Neural Networks profitieren von normalisierten Features.
    
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
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Eindeutige Match-Gruppen: {groups.nunique()}")
    
    return X, y, feature_cols, groups


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


def normalize_features(X_train, X_val, X_test):
    """
    Normalisiert Features (wichtig für Neural Networks).
    
    Neural Networks lernen besser, wenn Features auf ähnlicher Skala sind.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Features normalisiert")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_model(X_train, y_train, X_val, y_val):
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
    
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # 2 Hidden Layers: 100 → 50 Neuronen
        max_iter=500,                   # Max. Training-Iterationen
        learning_rate_init=0.001,       # Lernrate
        random_state=42,
        early_stopping=True,            # Stoppt bei fehlender Verbesserung
        validation_fraction=0.1         # 10% für Early Stopping
    )
    
    print("Trainiere Modell...")
    print("  (Dies kann einige Minuten dauern)")
    model.fit(X_train, y_train)
    print("✓ Training abgeschlossen")
    
    # Evaluation
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"\nErgebnisse:")
    print(f"  Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Iterationen: {model.n_iter_} von max. {model.max_iter}")
    
    return model


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
    
    return test_acc


def save_model(model, scaler, feature_cols):
    """Speichert trainiertes Modell und Scaler."""
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "neural_network.pkl"
    joblib.dump(model, model_path)
    print(f"\n✓ Modell gespeichert: {model_path}")
    
    scaler_path = models_dir / "neural_network_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler gespeichert: {scaler_path}")
    
    features_path = models_dir / "neural_network_features.txt"
    with open(features_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"✓ Features gespeichert: {features_path}")


def main():
    """Hauptfunktion: Lädt Daten, trainiert Modell, evaluiert."""
    print("=" * 60)
    print("NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    df = load_data()
    X, y, feature_cols, groups = prepare_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, groups)
    
    # Normalisiere Features (wichtig für Neural Networks)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_features(
        X_train, X_val, X_test
    )
    
    model = train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    test_acc = evaluate_model(model, X_test_scaled, y_test)
    save_model(model, scaler, feature_cols)
    
    print("\n" + "=" * 60)
    print("FERTIG!")
    print("=" * 60)
    print(f"Finale Test Accuracy: {test_acc*100:.2f}%")
    
    return model, test_acc


if __name__ == "__main__":
    main()
