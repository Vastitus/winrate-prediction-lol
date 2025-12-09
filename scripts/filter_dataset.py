"""
Filtert das Dataset: Entfernt Matches mit Champions, die zu selten auf ihrer Position gespielt werden.

Dieses Script:
1. Analysiert wie oft jeder Champion auf jeder Position vorkommt
2. Filtert Matches, wo Champions <50x auf ihrer Position vorkommen
3. Erstellt ein bereinigtes Dataset für Training
"""

import pandas as pd
import sys
from pathlib import Path

# Projekt-Root
project_root = Path(__file__).parent.parent

def analyze_champion_distribution(df: pd.DataFrame):
    """
    Analysiert wie oft jeder Champion auf jeder Position vorkommt.
    
    Returns:
        Dictionary: {(champion_id, position): count}
    """
    positions = ['top', 'jungle', 'mid', 'adc', 'support']
    champ_position_counts = {}
    
    print("Analysiere Champion-Verteilung pro Position...")
    
    for pos in positions:
        # Team 1 und Team 2
        for team in [1, 2]:
            col = f'team{team}_{pos}'
            champ_counts = df[col].value_counts()
            
            for champ_id, count in champ_counts.items():
                if champ_id != -1:  # Ignoriere fehlende Werte
                    key = (champ_id, pos)
                    champ_position_counts[key] = champ_position_counts.get(key, 0) + count
    
    return champ_position_counts


def filter_dataset(df: pd.DataFrame, min_matches: int = 50):
    """
    Filtert Matches: Behält nur Matches, wo alle Champions mindestens min_matches
    auf ihrer Position vorkommen.
    
    Args:
        df: DataFrame mit Match-Daten
        min_matches: Minimum Anzahl Matches pro Champion auf Position
    
    Returns:
        Gefiltertes DataFrame
    """
    print(f"\nFiltere Dataset (min {min_matches} Matches pro Champion auf Position)...")
    
    # Analysiere Champion-Verteilung
    champ_position_counts = analyze_champion_distribution(df)
    
    # Erstelle Set von gültigen (champion_id, position) Kombinationen
    valid_combinations = {
        (champ_id, pos) 
        for (champ_id, pos), count in champ_position_counts.items()
        if count >= min_matches
    }
    
    print(f"Gültige Champion-Position Kombinationen: {len(valid_combinations)}")
    
    # Zähle wie viele Champions pro Position gültig sind
    valid_by_position = {}
    for pos in ['top', 'jungle', 'mid', 'adc', 'support']:
        valid_champs = {champ_id for (champ_id, p), count in champ_position_counts.items() 
                       if p == pos and count >= min_matches}
        valid_by_position[pos] = valid_champs
        print(f"  {pos}: {len(valid_champs)} Champions")
    
    # Filtere Matches: Behalte nur Matches, wo alle Champions gültig sind
    positions = ['top', 'jungle', 'mid', 'adc', 'support']
    valid_matches = []
    
    for idx, row in df.iterrows():
        is_valid = True
        
        # Prüfe alle Positionen beider Teams
        for team in [1, 2]:
            for pos in positions:
                col = f'team{team}_{pos}'
                champ_id = row[col]
                
                if champ_id == -1:
                    is_valid = False
                    break
                
                # Prüfe ob (champion_id, position) gültig ist
                if (champ_id, pos) not in valid_combinations:
                    is_valid = False
                    break
            
            if not is_valid:
                break
        
        if is_valid:
            valid_matches.append(idx)
    
    filtered_df = df.loc[valid_matches].copy()
    
    print(f"\nOriginal: {len(df)} Matches")
    print(f"Gefiltert: {len(filtered_df)} Matches")
    print(f"Entfernt: {len(df) - len(filtered_df)} Matches ({(len(df) - len(filtered_df))/len(df)*100:.1f}%)")
    
    return filtered_df


def main():
    input_path = project_root / "data" / "datasets" / "35k_matches_dataset.csv"
    output_path = project_root / "data" / "datasets" / "29668_filtered_dataset.csv"
    
    print("=" * 60)
    print("DATASET FILTERUNG")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    print()
    
    # Lade Dataset
    print(f"Lade Dataset...")
    df = pd.read_csv(input_path)
    print(f"Geladen: {len(df)} Matches")
    
    # Filtere Dataset
    filtered_df = filter_dataset(df, min_matches=50)
    
    # Speichere gefiltertes Dataset
    print(f"\nSpeichere gefiltertes Dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    
    print(f"✓ Gefiltertes Dataset gespeichert: {output_path}")
    print()
    print("=" * 60)
    print("STATISTIKEN")
    print("=" * 60)
    print(f"Original Matches: {len(df)}")
    print(f"Gefilterte Matches: {len(filtered_df)}")
    print(f"Verlust: {len(df) - len(filtered_df)} Matches ({(len(df) - len(filtered_df))/len(df)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
