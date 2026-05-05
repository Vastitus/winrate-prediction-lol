"""Filtert Dataset: Entfernt Matches mit Champions, die <N-mal auf ihrer Position gespielt wurden."""

import argparse
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent

def analyze_champion_distribution(df: pd.DataFrame):
    """Analysiert Champion-Verteilung pro Position."""
    positions = ['top', 'jungle', 'mid', 'adc', 'support']
    champ_position_counts = {}
    
    for pos in positions:
        for team in [1, 2]:
            col = f'team{team}_{pos}'
            champ_counts = df[col].value_counts()
            for champ_id, count in champ_counts.items():
                if champ_id != -1:
                    key = (champ_id, pos)
                    champ_position_counts[key] = champ_position_counts.get(key, 0) + count
    
    return champ_position_counts


def filter_dataset(df: pd.DataFrame, min_matches: int = 50):
    """Filtert Matches: Behält nur Matches, wo alle Champions >=min_matches auf ihrer Position vorkommen."""
    champ_position_counts = analyze_champion_distribution(df)
    valid_combinations = {
        (champ_id, pos) 
        for (champ_id, pos), count in champ_position_counts.items()
        if count >= min_matches
    }
    
    positions = ['top', 'jungle', 'mid', 'adc', 'support']
    valid_matches = []
    
    for idx, row in df.iterrows():
        is_valid = True
        for team in [1, 2]:
            for pos in positions:
                col = f'team{team}_{pos}'
                champ_id = row[col]
                if champ_id == -1 or (champ_id, pos) not in valid_combinations:
                    is_valid = False
                    break
            if not is_valid:
                break
        if is_valid:
            valid_matches.append(idx)
    
    return df.loc[valid_matches].copy()


def parse_args():
    """CLI-Argumente für flexible Dataset-Dateinamen."""
    parser = argparse.ArgumentParser(
        description="Filtert Matches basierend auf Mindestanzahl pro Champion-Position."
    )
    parser.add_argument(
        "--input",
        default=str(project_root / "data" / "datasets" / "35k_matches_dataset.csv"),
        help="Pfad zur Eingabedatei (CSV).",
    )
    parser.add_argument(
        "--output",
        default=str(project_root / "data" / "datasets" / "29668_filtered_dataset.csv"),
        help="Pfad zur Ausgabedatei (CSV).",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=50,
        help="Mindestanzahl Spiele pro Champion-Position.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    filtered_df = filter_dataset(df, min_matches=args.min_matches)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    print(f"Gefiltert: {len(df)} -> {len(filtered_df)} Matches")
    print(f"Gespeichert: {output_path}")


if __name__ == "__main__":
    main()
