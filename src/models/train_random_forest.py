"""Trainiert Random Forest Modell für Win-Rate Vorhersage."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


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
    """Bereitet Features vor: entfernt Metadaten, behandelt Champion-IDs als int."""
    exclude_cols = [
        "match_id", "game_version", "game_mode", "queue_id", "game_duration",
        "target", "team1_win", "team2_win", "has_complete_positions"
    ]
    if exclude_bans:
        exclude_cols.extend([col for col in df.columns if 'ban' in col])
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy().fillna(-1)
    
    for col in X.columns:
        if any(tok in col for tok in ["team1_", "team2_"]) and not any(x in col for x in ["winrate", "synergy", "diff", "matchup"]):
            try:
                X[col] = X[col].astype(int)
            except Exception:
                pass
    
    y = df["target"]
    match_ids = df["match_id"].astype(str)
    groups = match_ids.str.replace('_mirrored', '', regex=False)
    
    return X, y, feature_cols, groups


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


def train_model(X_train, y_train, X_val, y_val):
    """Trainiert Random Forest Modell."""
    model = RandomForestClassifier(
        n_estimators=250, max_depth=10, min_samples_split=15,
        min_samples_leaf=8, max_features='sqrt', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"Train: {train_acc:.2%} | Val: {val_acc:.2%}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluiert Modell auf Test-Set."""
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test Accuracy: {test_acc:.2%}")
    print(classification_report(y_test, model.predict(X_test)))
    return test_acc


def save_model(model, feature_cols):
    """Speichert Modell und Features."""
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, models_dir / "random_forest.pkl")
    (models_dir / "random_forest_features.txt").write_text("\n".join(feature_cols))


def main(exclude_bans=False):
    """Hauptfunktion: lädt Daten, trainiert Modell, evaluiert."""
    df = load_data()
    X, y, feature_cols, groups = prepare_features(df, exclude_bans=exclude_bans)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, groups)
    
    model = train_model(X_train, y_train, X_val, y_val)
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
    import sys
    
    # Wenn "--compare" als Argument, vergleiche beide Varianten
    if "--compare" in sys.argv or "-c" in sys.argv:
        compare_with_without_bans()
    # Wenn "--no-bans" als Argument, trainiere ohne Bans
    elif "--no-bans" in sys.argv or "-n" in sys.argv:
        main(exclude_bans=True)
    # Standard: Ohne Bans trainieren (nur Champion-Picks)
    else:
        main(exclude_bans=True)
