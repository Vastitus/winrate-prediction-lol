"""Trainiert XGBoost Modell für Win-Rate Vorhersage."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib


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


def prepare_features(df, exclude_bans=True, use_categorical=True):
    """
    Bereitet Features für Training vor.
    
    Entfernt nicht-feature Spalten (IDs, Metadaten, Target).
    Behält nur die Features, die das Modell lernen soll.
    
    Args:
        exclude_bans: Wenn True, werden Ban-Features ausgeschlossen (nur Champion-Picks)
        use_categorical: Wenn True, werden Champion-IDs als kategorische Features behandelt
    
    Returns:
        X: Features
        y: Target-Variable
        feature_cols: Liste der Feature-Spalten
        groups: Gruppen für Group-basiertes Splitting (Basis-Match-IDs)
        categorical_indices: Indizes der kategorischen Features (für XGBoost)
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
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    
    # XGBoost kann echte categorical splits, wenn dtype='category' und enable_categorical=True.
    categorical_cols = []
    if use_categorical:
        # Champion-ID Spalten (NICHT abgeleitete numeric Features)
        champion_cols = [
            col
            for col in feature_cols
            if any(pos in col for pos in ["top", "jungle", "mid", "adc", "support"])
            and ("winrate" not in col)
            and ("matchup" not in col)
            and ("synergy" not in col)
            and ("diff" not in col)
        ]
        for col in champion_cols:
            X[col] = X[col].fillna(-1).astype(int).astype("category")
        categorical_cols = champion_cols

        # Alle numeric feature-spalten sauber casten
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]
        for col in numeric_cols:
            X[col] = X[col].fillna(0.0).astype(float)

        print(f"[INFO] {len(categorical_cols)} Champion-Features als pandas category (enable_categorical)")
    else:
        X = X.fillna(-1).astype(float)
    
    y = df["target"]
    
    # Erstelle Gruppen für Group-basiertes Splitting
    match_ids = df["match_id"].astype(str)
    groups = match_ids.str.replace('_mirrored', '', regex=False)
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Feature-Spalten: {feature_cols}")
    print(f"Eindeutige Match-Gruppen: {groups.nunique()}")
    
    return X, y, feature_cols, groups, categorical_cols


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


def train_model(X_train, y_train, X_val, y_val, categorical_cols=None):
    """
    Trainiert XGBoost Modell.
    
    XGBoost (Aggressive Regularisierung gegen Overfitting):
    - Baut Bäume sequenziell (jeder korrigiert Fehler des vorherigen)
    - n_estimators=300: Mehr Bäume = mehr "Training" (ähnlich wie Epochen)
    - max_depth=6: Flachere Bäume (reduziert Overfitting stark)
    - learning_rate=0.05: Niedrigere Lernrate (bessere Generalisierung)
    - subsample=0.7: Nutze nur 70% der Daten pro Baum (reduziert Overfitting)
    - colsample_bytree=0.7: Nutze nur 70% der Features pro Baum (reduziert Overfitting)
    - min_child_weight=5: Mindestens 5 Samples pro Blatt (reduziert Overfitting)
    - reg_alpha=0.1, reg_lambda=1.0: L1/L2 Regularisierung (reduziert Overfitting)
    - Ziel: Training und Test Accuracy sollten näher zusammen sein (weniger Overfitting)
    """
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)
    
    # XGBoost mit echten categorical splits (pandas category + enable_categorical=True)
    # Aggressive Regularisierung gegen Overfitting
    model = xgb.XGBClassifier(
        n_estimators=300,      # Mehr Bäume = mehr "Training" (ähnlich wie Epochen)
        max_depth=6,           # Flachere Bäume (reduziert Overfitting stark)
        learning_rate=0.05,    # Niedrigere Lernrate (bessere Generalisierung)
        subsample=0.7,         # Nutze nur 70% der Daten pro Baum (reduziert Overfitting)
        colsample_bytree=0.7, # Nutze nur 70% der Features pro Baum (reduziert Overfitting)
        min_child_weight=5,    # Mindestens 5 Samples pro Blatt (reduziert Overfitting)
        reg_alpha=0.1,         # L1 Regularisierung (reduziert Overfitting)
        reg_lambda=1.0,        # L2 Regularisierung (reduziert Overfitting)
        random_state=42,       # Für Reproduzierbarkeit
        eval_metric="logloss",  # Metrik für Evaluation
        tree_method="hist",     # Schnellere Methode
        enable_categorical=True
    )
    
    print("Trainiere Modell...")
    # Wichtig: Für enable_categorical muss DataFrame (mit dtype category) durchgereicht werden.
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print("[OK] Training abgeschlossen")
    
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
    print(f"\n[OK] Modell gespeichert: {model_path}")
    
    features_path = models_dir / "xgboost_features.txt"
    with open(features_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"[OK] Features gespeichert: {features_path}")


def main(exclude_bans=True):
    """
    Hauptfunktion: Lädt Daten, trainiert Modell, evaluiert.
    
    Args:
        exclude_bans: Wenn True, werden Bans ausgeschlossen (nur Champion-Picks)
    """
    print("=" * 60)
    print("XGBOOST TRAINING")
    if exclude_bans:
        print("(OHNE BANS - nur Champion-Picks)")
    else:
        print("(MIT BANS)")
    print("=" * 60)
    
    df = load_data()
    X, y, feature_cols, groups, categorical_cols = prepare_features(df, exclude_bans=exclude_bans)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, groups)
    
    model = train_model(X_train, y_train, X_val, y_val, categorical_cols)
    test_acc = evaluate_model(model, X_test, y_test)
    save_model(model, feature_cols)
    
    print("\n" + "=" * 60)
    print("FERTIG!")
    print("=" * 60)
    print(f"Finale Test Accuracy: {test_acc*100:.2f}%")
    
    return model, test_acc


if __name__ == "__main__":
    main()
