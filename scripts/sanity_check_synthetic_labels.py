"""
Sanity check: train on an artificially easy target so models MUST reach ~100% accuracy.

Idea (from professor):
- Create a synthetic label that depends on a simple, learnable rule over champselect features,
  e.g. "Team1 wins iff it picked champion A AND champion B".
- Train models on that synthetic label and evaluate on a held-out test split.

This does NOT prove real predictive power on the real target; it only validates that
the data pipeline + encoding + models can learn a strong signal when it exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None


# Ensure project root on sys.path so `import src...` works when running directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.preprocessing import infer_feature_spec, make_preprocessor  # noqa: E402


POSITIONS = ["top", "jungle", "mid", "adc", "support"]


def make_groups(df: pd.DataFrame) -> pd.Series:
    match_ids = df["match_id"].astype(str)
    return match_ids.str.replace("_mirrored", "", regex=False)


def pick_default_champs(df: pd.DataFrame) -> tuple[int, int]:
    """Pick two frequent champs so the synthetic positive class isn't vanishingly small."""
    pick_cols = [f"team1_{p}" for p in POSITIONS] + [f"team2_{p}" for p in POSITIONS]
    vals = df[pick_cols].fillna(-1).astype(int).values.reshape(-1)
    vals = vals[vals != -1]
    uniq, cnt = np.unique(vals, return_counts=True)
    order = np.argsort(-cnt)
    if len(order) < 2:
        raise RuntimeError("Not enough champion IDs found to choose defaults.")
    return int(uniq[order[0]]), int(uniq[order[1]])


def synth_label_team1_has_both(df: pd.DataFrame, champ_a: int, champ_b: int) -> np.ndarray:
    """y=1 iff Team1 has both champs (in any roles)."""
    t1 = df[[f"team1_{p}" for p in POSITIONS]].fillna(-1).astype(int).values
    has_a = (t1 == int(champ_a)).any(axis=1)
    has_b = (t1 == int(champ_b)).any(axis=1)
    return (has_a & has_b).astype(int)


def flip_noise(y: np.ndarray, noise: float, seed: int) -> np.ndarray:
    if noise <= 0:
        return y
    rng = np.random.RandomState(seed)
    y = y.copy()
    mask = rng.rand(len(y)) < noise
    y[mask] = 1 - y[mask]
    return y


def evaluate(model, X_test, y_test, name: str) -> None:
    pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    out = f"{name}: acc={acc*100:.2f}%"
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
        ll = float(log_loss(y_test, np.column_stack([1 - proba, proba]), labels=[0, 1]))
        out += f" | auc={auc:.4f} | logloss={ll:.4f}"
    print(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champ-a", type=int, default=None, help="Champion ID A (Team1 must have A to win)")
    ap.add_argument("--champ-b", type=int, default=None, help="Champion ID B (Team1 must have B to win)")
    ap.add_argument("--noise", type=float, default=0.0, help="Flip label with this probability (e.g. 0.05)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-bans", action="store_true", help="Exclude bans (default: True)")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / "data" / "datasets"
    champselect = data_dir / "59334_champselect_features.csv"
    winrates = data_dir / "59334_with_winrates.csv"
    augmented = data_dir / "59334_filtered_augmented_dataset.csv"

    if champselect.exists():
        df = pd.read_csv(champselect)
    elif winrates.exists():
        df = pd.read_csv(winrates)
    elif augmented.exists():
        df = pd.read_csv(augmented)
    else:
        raise FileNotFoundError("No dataset found in data/datasets/")

    champ_a = args.champ_a
    champ_b = args.champ_b
    if champ_a is None or champ_b is None:
        ca, cb = pick_default_champs(df)
        champ_a = ca if champ_a is None else champ_a
        champ_b = cb if champ_b is None else champ_b

    y = synth_label_team1_has_both(df, champ_a, champ_b)
    y = flip_noise(y, noise=float(args.noise), seed=int(args.seed))

    # Same exclusion approach as training scripts (+ ignore synergy always).
    exclude_cols = [
        "match_id",
        "game_version",
        "game_mode",
        "queue_id",
        "game_duration",
        "target",
        "team1_win",
        "team2_win",
        "has_complete_positions",
    ]
    # Exclude synergy columns if present
    exclude_cols.extend([c for c in df.columns if "synergy" in c.lower()])
    # Exclude bans (default behavior in the project)
    if args.exclude_bans or True:
        exclude_cols.extend([c for c in df.columns if "ban" in c.lower()])

    spec = infer_feature_spec(df, exclude_cols=exclude_cols)
    X = df[spec.feature_cols].copy()

    groups = make_groups(df)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=int(args.seed))
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pos_rate = float(np.mean(y_train))
    print("=" * 72)
    print("SYNTHETIC LABEL SANITY CHECK")
    print("=" * 72)
    print(f"Rule: y=1 iff Team1 has BOTH champs A={champ_a} and B={champ_b}")
    print(f"Noise (label flip prob): {float(args.noise):.3f}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)} | Train pos rate: {pos_rate:.3f}")
    print(f"Raw input features (columns): {len(spec.feature_cols)}")
    print()

    pre = make_preprocessor(spec)

    rf = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", RandomForestClassifier(n_estimators=300, random_state=int(args.seed), n_jobs=-1)),
        ]
    )
    rf.fit(X_train, y_train)
    evaluate(rf, X_test, y_test, "RandomForest")

    if xgb is not None:
        xgb_model = Pipeline(
            steps=[
                ("preprocess", pre),
                ("model", xgb.XGBClassifier(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=int(args.seed),
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                )),
            ]
        )
        xgb_model.fit(X_train, y_train)
        evaluate(xgb_model, X_test, y_test, "XGBoost")
    else:
        print("XGBoost: skipped (xgboost not installed)")

    nn = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64,),
                max_iter=500,
                learning_rate_init=0.001,
                alpha=1e-4,
                random_state=int(args.seed),
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ]
    )
    nn.fit(X_train, y_train)
    evaluate(nn, X_test, y_test, "NeuralNet(MLP)")

    print()
    print("Expected behavior:")
    print("- With noise=0.0, models should reach ~100% (or very close) because the rule is deterministic and one-hot makes it linearly/tree separable.")
    print("- With noise>0, max achievable accuracy is <100% by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





