"""
Data Augmentation für LoL Match-Daten.

Bei Bildern: Rotation, Flipping, etc. um mehr Trainingsdaten zu generieren.
Bei LoL Matches: Team-Spiegelung (Team 1 ↔ Team 2).

Warum funktioniert das?
- Ein Match ist symmetrisch: Wenn Team 1 gewinnt, könnte man auch sagen "Team 2 verliert"
- Durch Spiegeln verdoppeln wir die Datenmenge
- Das Modell lernt, dass die Teams gleichwertig sind (keine Bias zu Team 1)

Vorteile:
- Verdoppelt die Datenmenge (z.B. 29k → 58k Matches)
- Bessere Generalisierung
- Modell lernt, dass Team-Position irrelevant ist
"""

import pandas as pd
from pathlib import Path


def augment_match(row):
    """
    Spiegelt einen Match (Team 1 ↔ Team 2).
    
    Beispiel:
    Original: Team1=[Champ1, Champ2], Team2=[Champ3, Champ4], Target=1 (Team1 gewinnt)
    Gespiegelt: Team1=[Champ3, Champ4], Team2=[Champ1, Champ2], Target=0 (Team1 verliert)
    
    WICHTIG: Match-ID wird modifiziert, damit Original und gespiegelte Version
    beim Splitting als Gruppe behandelt werden können (verhindert Data Leakage).
    """
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
    """
    Erstellt augmentierte Version des Datasets durch Team-Spiegelung.
    
    Jeder Match wird gespiegelt, sodass wir doppelt so viele Daten haben.
    """
    print("Erstelle augmentierte Matches (Team-Spiegelung)...")
    
    augmented_rows = []
    for idx, row in df.iterrows():
        # Original Match behalten
        # (wird schon im DataFrame sein)
        
        # Gespiegelten Match hinzufügen
        augmented_row = augment_match(row)
        augmented_rows.append(augmented_row)
    
    # Erstelle DataFrame aus augmentierten Rows
    augmented_df = pd.DataFrame(augmented_rows)
    
    # Kombiniere Original + Augmentiert
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    
    print(f"Original: {len(df)} Matches")
    print(f"Augmentiert: {len(augmented_df)} Matches")
    print(f"Gesamt: {len(combined_df)} Matches")
    
    return combined_df


def main():
    """Hauptfunktion: Lädt Dataset, augmentiert es, speichert es."""
    project_root = Path(__file__).parent.parent.parent
    
    input_path = project_root / "data" / "datasets" / "29668_filtered_dataset.csv"
    output_path = project_root / "data" / "datasets" / "59334_filtered_augmented_dataset.csv"
    
    print("=" * 60)
    print("DATA AUGMENTATION")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    print()
    
    # Lade Dataset
    print("Lade Dataset...")
    df = pd.read_csv(input_path)
    print(f"Geladen: {len(df)} Matches")
    
    # Augmentiere Dataset
    augmented_df = augment_dataset(df)
    
    # Speichere augmentiertes Dataset
    print(f"\nSpeichere augmentiertes Dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    
    print(f"✓ Augmentiertes Dataset gespeichert: {output_path}")
    print()
    print("=" * 60)
    print("STATISTIKEN")
    print("=" * 60)
    print(f"Original Matches: {len(df)}")
    print(f"Augmentierte Matches: {len(augmented_df)}")
    print(f"Gesamt Matches: {len(augmented_df)}")
    print(f"Verdopplung: {len(augmented_df) / len(df):.1f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
