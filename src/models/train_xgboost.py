"""
Trainiert ein XGBoost Modell für Win-Rate Vorhersage.

XGBoost (Extreme Gradient Boosting) ist ein Gradient Boosting Modell.
Es baut sequenziell Entscheidungsbäume, wobei jeder neue Baum die Fehler
des vorherigen korrigiert. Das führt zu sehr genauen Vorhersagen.

Vorteile:
- Sehr hohe Accuracy
- Robust gegen Overfitting (durch Regularisierung)
- Zeigt Feature Importance
- Funktioniert gut mit strukturierten Daten
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    
    Entfernt nicht-feature Spalten (IDs, Metadaten, Target).
    Behält nur die Features, die das Modell lernen soll.
    
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
    print(f"Feature-Spalten: {feature_cols}")
    print(f"Eindeutige Match-Gruppen: {groups.nunique()}")
    
    return X, y, feature_cols, groups


def split_data(X, y, groups):
    """
    Teilt Daten in Trainings-, Validierungs- und Test-Sets.
    
    WICHTIG: Verwendet Group-basiertes Splitting, damit Original und
    gespiegelte Version eines Matches immer im gleichen Set bleiben.
    Das verhindert Data Leakage.
    
    - Training (70%): Modell lernt darauf
    - Validation (15%): Tuning und Überprüfung während Training
    - Test (15%): Finale Evaluation (wird NUR am Ende verwendet)
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


def train_model(X_train, y_train, X_val, y_val):
    """
    Trainiert XGBoost Modell.
    
    XGBoost:
    - Baut Bäume sequenziell (jeder korrigiert Fehler des vorherigen)
    - n_estimators=100: Anzahl Bäume
    - max_depth=10: Maximale Tiefe
    - learning_rate=0.1: Wie stark jeder neue Baum die Vorhersage ändert
    - Regularisierung verhindert Overfitting
    """
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)
    
    model = xgb.XGBClassifier(
        n_estimators=100,      # Anzahl Bäume
        max_depth=10,          # Maximale Tiefe
        learning_rate=0.1,     # Lernrate
        random_state=42,       # Für Reproduzierbarkeit
        eval_metric="logloss"  # Metrik für Evaluation
    )
    
    print("Trainiere Modell...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  # Evaluierung während Training
        verbose=False
    )
    print("✓ Training abgeschlossen")
    
    # Evaluation
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"\nErgebnisse:")
    print(f"  Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 wichtigste Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluiert Modell auf Test-Set (finale Evaluation)."""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))
    
    return test_acc


def save_model(model, feature_cols):
    """Speichert trainiertes Modell."""
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "xgboost.pkl"
    joblib.dump(model, model_path)
    print(f"\n✓ Modell gespeichert: {model_path}")
    
    features_path = models_dir / "xgboost_features.txt"
    with open(features_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"✓ Features gespeichert: {features_path}")


def main():
    """Hauptfunktion: Lädt Daten, trainiert Modell, evaluiert."""
    print("=" * 60)
    print("XGBOOST TRAINING")
    print("=" * 60)
    
    df = load_data()
    X, y, feature_cols = prepare_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    model = train_model(X_train, y_train, X_val, y_val)
    test_acc = evaluate_model(model, X_test, y_test)
    save_model(model, feature_cols)
    
    print("\n" + "=" * 60)
    print("FERTIG!")
    print("=" * 60)
    print(f"Finale Test Accuracy: {test_acc*100:.2f}%")
    
    return model, test_acc


if __name__ == "__main__":
    main()
