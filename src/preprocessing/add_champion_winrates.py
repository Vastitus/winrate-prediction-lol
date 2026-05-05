"""
Fügt Champion-Winrates als Features hinzu.

Berechnet Winrates für jeden Champion basierend auf historischen Daten
(NUR aus Trainingsdaten, um Data Leakage zu vermeiden).
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


def calculate_champion_winrates(df_train):
    """Berechnet Champion-Winrates (champion_id -> winrate) aus Trainingsdaten."""
    champion_winrates = {}

    champion_cols = [col for col in df_train.columns if any(pos in col for pos in ['top', 'jungle', 'mid', 'adc', 'support'])]

    all_champions = set()
    for col in champion_cols:
        all_champions.update(df_train[col].unique())
    all_champions.discard(-1)

    for champ_id in all_champions:
        wins = 0
        total = 0

        for pos in ['top', 'jungle', 'mid', 'adc', 'support']:
            t1 = df_train[df_train[f'team1_{pos}'] == champ_id]
            if len(t1) > 0:
                wins += t1['target'].sum()
                total += len(t1)

            t2 = df_train[df_train[f'team2_{pos}'] == champ_id]
            if len(t2) > 0:
                wins += len(t2) - t2['target'].sum()
                total += len(t2)

        champion_winrates[champ_id] = wins / total if total > 0 else 0.5

    return champion_winrates


def add_winrate_features(df, champion_winrates):
    """Fügt Champion-Winrates als Features hinzu (pro Position + Team-Mittel + Diff)."""
    df = df.copy()

    def get_champ_winrate(champ_id):
        if champ_id == -1:
            return 0.5
        return champion_winrates.get(champ_id, 0.5)

    team1_champs = ['team1_top', 'team1_jungle', 'team1_mid', 'team1_adc', 'team1_support']
    team2_champs = ['team2_top', 'team2_jungle', 'team2_mid', 'team2_adc', 'team2_support']

    df['team1_avg_winrate'] = df[team1_champs].apply(lambda s: s.map(get_champ_winrate)).mean(axis=1)
    df['team2_avg_winrate'] = df[team2_champs].apply(lambda s: s.map(get_champ_winrate)).mean(axis=1)
    df['winrate_diff'] = df['team1_avg_winrate'] - df['team2_avg_winrate']

    for pos in ['top', 'jungle', 'mid', 'adc', 'support']:
        df[f'team1_{pos}_winrate'] = df[f'team1_{pos}'].apply(get_champ_winrate)
        df[f'team2_{pos}_winrate'] = df[f'team2_{pos}'].apply(get_champ_winrate)

    return df


def parse_args():
    """CLI-Argumente für flexible Input/Output-Dateinamen."""
    project_root = Path(__file__).parent.parent.parent
    parser = argparse.ArgumentParser(description="Fügt Champion-Winrate-Features hinzu.")
    parser.add_argument(
        "--input",
        default=str(project_root / "data" / "datasets" / "59334_filtered_augmented_dataset.csv"),
        help="Pfad zur Eingabedatei (CSV).",
    )
    parser.add_argument(
        "--output",
        default=str(project_root / "data" / "datasets" / "59334_with_winrates.csv"),
        help="Pfad zur Ausgabedatei (CSV).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.input)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset nicht gefunden: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Group-basierter Split, damit Originale und gespiegelte Matches zusammenbleiben.
    # Stats werden ausschließlich aus den Trainingsdaten berechnet.
    match_ids = df["match_id"].astype(str)
    groups = match_ids.str.replace('_mirrored', '', regex=False)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, _ = next(gss.split(df, df['target'], groups))
    df_train = df.iloc[train_idx].copy()

    champion_winrates = calculate_champion_winrates(df_train)
    df_with_features = add_winrate_features(df, champion_winrates)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_features.to_csv(output_path, index=False)

    n_features = len([c for c in df_with_features.columns if 'winrate' in c])
    print(f"Trainings-Set: {len(df_train)} Matches  |  Champions: {len(champion_winrates)}")
    print(f"Gespeichert: {output_path}  ({n_features} Winrate-Features)")


if __name__ == "__main__":
    main()

