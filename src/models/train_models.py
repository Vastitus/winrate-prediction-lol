"""
Training verschiedener ML-Modelle für Win-Rate Prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Lädt das Dataset."""
    if dataset_path.suffix == ".parquet":
        return pd.read_parquet(dataset_path)
    elif dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unbekanntes Dateiformat: {dataset_path.suffix}")


def prepare_features(df: pd.DataFrame):
    """Bereitet Features für Training vor."""
    # Entferne nicht-feature Spalten
    exclude_cols = ["match_id", "game_version", "game_mode", "target", "team1_win", "team2_win"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].fillna(-1)
    y = df["target"]
    
    return X, y, feature_cols


def train_random_forest(X_train, y_train, X_val, y_val):
    """Trainiert Random Forest Modell."""
    print("Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"Random Forest - Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_pred))
    
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Trainiert XGBoost Modell."""
    print("Training XGBoost...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluation
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"XGBoost - Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_pred))
    
    return model


def compare_models(df: pd.DataFrame, output_dir: Path):
    """Vergleicht verschiedene Modelle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Daten vorbereiten
    X, y, feature_cols = prepare_features(df)
    
    # Train/Val/Test Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Trainiere Modelle
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Test Evaluation
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    
    rf_test_pred = rf_model.predict(X_test)
    xgb_test_pred = xgb_model.predict(X_test)
    
    rf_test_acc = accuracy_score(y_test, rf_test_pred)
    xgb_test_acc = accuracy_score(y_test, xgb_test_pred)
    
    print(f"\nRandom Forest Test Accuracy: {rf_test_acc:.4f}")
    print(f"XGBoost Test Accuracy: {xgb_test_acc:.4f}")
    
    # Speichere Modelle
    joblib.dump(rf_model, output_dir / "random_forest.pkl")
    joblib.dump(xgb_model, output_dir / "xgboost.pkl")
    
    print(f"\nModelle gespeichert in: {output_dir}")
    
    return {
        "random_forest": rf_model,
        "xgboost": xgb_model
    }


if __name__ == "__main__":
    # Pfade relativ zum Projekt-Root
    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / "data" / "datasets" / "lol_matches.parquet"
    models_dir = project_root / "models"
    
    df = load_dataset(dataset_path)
    compare_models(df, models_dir)

