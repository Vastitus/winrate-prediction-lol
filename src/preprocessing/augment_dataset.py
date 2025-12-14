"""Data Augmentation: Team-Spiegelung verdoppelt die Datenmenge."""

import pandas as pd
from pathlib import Path


def augment_match(row):
    """Spiegelt Match (Team 1 ↔ Team 2) und modifiziert match_id für Group-Splitting."""
    augmented = row.copy()
    
    # Modifiziere Match-ID für gespiegelte Version
    # Original: "EUW1_123456" → Gespiegelt: "EUW1_123456_mirrored"
    # Beim Splitting können wir dann nach Basis-ID gruppieren
    original_match_id = str(row.get('match_id', 'unknown'))
    if not original_match_id.endswith('_mirrored'):
        # Extrahiere Basis-ID (entferne eventuell vorhandenes "_mirrored")
        base_id = original_match_id.replace('_mirrored', '')
        augmented['match_id'] = f"{base_id}_mirrored"
    else:
        # Falls bereits gespiegelt, behalte es
        augmented['match_id'] = original_match_id
    
    # Spiegle alle Team-Features
    positions = ['top', 'jungle', 'mid', 'adc', 'support']
    
    for pos in positions:
        # Team 1 ↔ Team 2
        team1_val = row[f'team1_{pos}']
        team2_val = row[f'team2_{pos}']
        augmented[f'team1_{pos}'] = team2_val
        augmented[f'team2_{pos}'] = team1_val
    
    # Spiegle Bans
    for i in range(1, 6):
        team1_ban = row[f'team1_ban_{i}']
        team2_ban = row[f'team2_ban_{i}']
        augmented[f'team1_ban_{i}'] = team2_ban
        augmented[f'team2_ban_{i}'] = team1_ban
    
    # Spiegle Win-Flags
    augmented['team1_win'] = row['team2_win']
    augmented['team2_win'] = row['team1_win']
    
    # WICHTIG: Target spiegeln (1 → 0, 0 → 1)
    # Wenn Team 1 gewinnt (1), dann gewinnt nach Spiegelung Team 2 (0)
    augmented['target'] = 1 - row['target']
    
    return augmented


def augment_dataset(df):
    """Erstellt augmentierte Version durch Team-Spiegelung."""
    augmented_rows = [augment_match(row) for _, row in df.iterrows()]
    augmented_df = pd.DataFrame(augmented_rows)
    return pd.concat([df, augmented_df], ignore_index=True)


def main():
    """Lädt Dataset, augmentiert es, speichert es."""
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data" / "datasets" / "29668_filtered_dataset.csv"
    output_path = project_root / "data" / "datasets" / "59334_filtered_augmented_dataset.csv"
    
    df = pd.read_csv(input_path)
    augmented_df = augment_dataset(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    print(f"Augmentiert: {len(df)} → {len(augmented_df)} Matches")


if __name__ == "__main__":
    main()
