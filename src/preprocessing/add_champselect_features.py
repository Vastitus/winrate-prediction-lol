"""
Erstellt ein erweitertes Champ-Select Feature-Dataset.

Ziel: Outcome nur aus Champ-Select besser vorhersagen.

Features (zusätzlich zu Picks):
- Champion Winrates (aus Trainings-Split berechnet)
- Lane-Matchup Winrates (z.B. team1_top vs team2_top)
- Team-Synergy (pairwise Winrate von Champion-Paaren im selben Team)

WICHTIG (gegen Data Leakage):
- Statistiken werden NUR aus dem TRAINING-SET berechnet.
- Das Splitting ist absichtlich identisch zu den Trainings-Scripts:
  GroupShuffleSplit(test_size=0.15, random_state=42) und danach Validation Split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from itertools import combinations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


POSITIONS = ["top", "jungle", "mid", "adc", "support"]


def _make_groups(df: pd.DataFrame) -> pd.Series:
    match_ids = df["match_id"].astype(str)
    return match_ids.str.replace("_mirrored", "", regex=False)


def _train_indices_like_training_scripts(df: pd.DataFrame) -> np.ndarray:
    """
    Nutzt exakt das gleiche Split-Verfahren wie die Trainings-Scripts:
    - 15% Test (group-basiert)
    - 15% Validation aus dem Rest
    Returned: Index-Array für TRAINING (70%).
    """
    groups = _make_groups(df)
    y = df["target"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, _test_idx = next(gss.split(df, y, groups))

    df_train_val = df.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=42)
    train_idx_rel, _val_idx_rel = next(gss2.split(df_train_val, y_train_val, groups_train_val))

    return df_train_val.index.values[train_idx_rel]


def _laplace(wins: int, total: int, alpha: float = 10.0) -> float:
    # Binärziel -> Beta(α, α) Prior => (wins+α)/(total+2α)
    return float((wins + alpha) / (total + 2.0 * alpha))


def compute_champion_winrates(df_train: pd.DataFrame) -> Dict[int, float]:
    """Winrate je Champion über alle Rollen (team1 + team2) aus df_train."""
    winrates: Dict[int, float] = {}

    # Zähle Vorkommen/Wins pro Champion (egal welche Lane)
    wins: Dict[int, int] = {}
    total: Dict[int, int] = {}

    for pos in POSITIONS:
        c1 = df_train[f"team1_{pos}"].fillna(-1).astype(int)
        c2 = df_train[f"team2_{pos}"].fillna(-1).astype(int)
        y = df_train["target"].astype(int)

        # Team1: win = y==1
        for champ_id, win in zip(c1.values, y.values):
            if champ_id == -1:
                continue
            total[champ_id] = total.get(champ_id, 0) + 1
            wins[champ_id] = wins.get(champ_id, 0) + int(win == 1)

        # Team2: win = y==0
        for champ_id, win in zip(c2.values, y.values):
            if champ_id == -1:
                continue
            total[champ_id] = total.get(champ_id, 0) + 1
            wins[champ_id] = wins.get(champ_id, 0) + int(win == 0)

    # Smoothed winrate
    for champ_id, t in total.items():
        w = wins.get(champ_id, 0)
        winrates[champ_id] = _laplace(w, t, alpha=10.0)

    return winrates


def compute_lane_matchup_winrates(df_train: pd.DataFrame) -> Dict[Tuple[str, int, int], float]:
    """
    Für jede Lane: P(team1_win | team1_pos=champA, team2_pos=champB).
    Key: (pos, champA, champB)
    """
    wins: Dict[Tuple[str, int, int], int] = {}
    total: Dict[Tuple[str, int, int], int] = {}

    y = df_train["target"].astype(int).values

    for pos in POSITIONS:
        a = df_train[f"team1_{pos}"].fillna(-1).astype(int).values
        b = df_train[f"team2_{pos}"].fillna(-1).astype(int).values
        for champ_a, champ_b, win in zip(a, b, y):
            if champ_a == -1 or champ_b == -1:
                continue
            key = (pos, int(champ_a), int(champ_b))
            total[key] = total.get(key, 0) + 1
            wins[key] = wins.get(key, 0) + int(win == 1)

    out: Dict[Tuple[str, int, int], float] = {}
    for key, t in total.items():
        out[key] = _laplace(wins.get(key, 0), t, alpha=10.0)
    return out


def compute_team_synergy_winrates(df_train: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    """
    Pairwise Synergy: P(team_win | pair in team).
    Key: (minChamp, maxChamp) (unordered)
    """
    wins: Dict[Tuple[int, int], int] = {}
    total: Dict[Tuple[int, int], int] = {}

    y = df_train["target"].astype(int).values

    # Team1 pairs => win if y==1
    t1 = df_train[[f"team1_{p}" for p in POSITIONS]].fillna(-1).astype(int).values
    for champs, win in zip(t1, y):
        champs = [int(c) for c in champs if int(c) != -1]
        for c1, c2 in combinations(sorted(champs), 2):
            key = (c1, c2)
            total[key] = total.get(key, 0) + 1
            wins[key] = wins.get(key, 0) + int(win == 1)

    # Team2 pairs => win if y==0
    t2 = df_train[[f"team2_{p}" for p in POSITIONS]].fillna(-1).astype(int).values
    for champs, win in zip(t2, y):
        champs = [int(c) for c in champs if int(c) != -1]
        for c1, c2 in combinations(sorted(champs), 2):
            key = (c1, c2)
            total[key] = total.get(key, 0) + 1
            wins[key] = wins.get(key, 0) + int(win == 0)

    out: Dict[Tuple[int, int], float] = {}
    for key, t in total.items():
        out[key] = _laplace(wins.get(key, 0), t, alpha=10.0)
    return out


def add_features(
    df: pd.DataFrame,
    champ_wr: Dict[int, float],
    matchup_wr: Dict[Tuple[str, int, int], float],
    synergy_wr: Dict[Tuple[int, int], float],
) -> pd.DataFrame:
    df = df.copy()

    def get_wr(champ_id: int) -> float:
        if champ_id == -1:
            return 0.5
        return champ_wr.get(int(champ_id), 0.5)

    # Role winrates
    for pos in POSITIONS:
        df[f"team1_{pos}_winrate"] = df[f"team1_{pos}"].fillna(-1).astype(int).map(get_wr)
        df[f"team2_{pos}_winrate"] = df[f"team2_{pos}"].fillna(-1).astype(int).map(get_wr)

    # Team average + diff
    df["team1_avg_winrate"] = df[[f"team1_{p}_winrate" for p in POSITIONS]].mean(axis=1)
    df["team2_avg_winrate"] = df[[f"team2_{p}_winrate" for p in POSITIONS]].mean(axis=1)
    df["winrate_diff"] = df["team1_avg_winrate"] - df["team2_avg_winrate"]

    # Lane matchups
    for pos in POSITIONS:
        a = df[f"team1_{pos}"].fillna(-1).astype(int).values
        b = df[f"team2_{pos}"].fillna(-1).astype(int).values
        vals = []
        for champ_a, champ_b in zip(a, b):
            if champ_a == -1 or champ_b == -1:
                vals.append(0.5)
            else:
                vals.append(matchup_wr.get((pos, int(champ_a), int(champ_b)), 0.5))
        df[f"lane_{pos}_matchup_winrate"] = vals

    # Team synergy (avg of 10 pairs)
    def team_synergy(row, prefix: str) -> float:
        champs = [int(row[f"{prefix}_{p}"]) for p in POSITIONS if int(row[f"{prefix}_{p}"]) != -1]
        if len(champs) < 2:
            return 0.5
        scores = []
        for c1, c2 in combinations(sorted(champs), 2):
            scores.append(synergy_wr.get((c1, c2), 0.5))
        return float(np.mean(scores)) if scores else 0.5

    df["team1_synergy"] = df.apply(lambda r: team_synergy(r, "team1"), axis=1)
    df["team2_synergy"] = df.apply(lambda r: team_synergy(r, "team2"), axis=1)
    df["synergy_diff"] = df["team1_synergy"] - df["team2_synergy"]

    return df


def parse_args():
    """CLI-Argumente für flexible Input/Output-Dateinamen."""
    project_root = Path(__file__).parent.parent.parent
    parser = argparse.ArgumentParser(description="Erstellt erweitertes Champ-Select Feature-Dataset.")
    parser.add_argument(
        "--input",
        default=str(project_root / "data" / "datasets" / "59334_filtered_augmented_dataset.csv"),
        help="Pfad zur Eingabedatei (CSV).",
    )
    parser.add_argument(
        "--output",
        default=str(project_root / "data" / "datasets" / "59334_champselect_features.csv"),
        help="Pfad zur Ausgabedatei (CSV).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_path = Path(args.input)
    if not src_path.exists():
        raise FileNotFoundError(f"Dataset nicht gefunden: {src_path}")

    print("[INFO] Lade Dataset...")
    df = pd.read_csv(src_path)
    print(f"[OK] Geladen: {len(df)} Matches")

    train_idx = _train_indices_like_training_scripts(df)
    df_train = df.loc[train_idx].copy()
    print(f"[OK] Training-Split für Feature-Stats: {len(df_train)} Matches")

    champ_wr = compute_champion_winrates(df_train)
    matchup_wr = compute_lane_matchup_winrates(df_train)
    synergy_wr = compute_team_synergy_winrates(df_train)

    df_out = add_features(df, champ_wr, matchup_wr, synergy_wr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    added_cols = [c for c in df_out.columns if c.endswith("_winrate") or c.endswith("_diff") or c.startswith("lane_") or c.endswith("_synergy")]
    print(f"[OK] Gespeichert: {out_path}")
    print(f"[OK] Neue/erweiterte Feature-Spalten (count): {len(added_cols)}")


if __name__ == "__main__":
    main()


