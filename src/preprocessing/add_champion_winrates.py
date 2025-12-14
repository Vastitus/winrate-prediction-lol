"""
Fügt Champion-Winrates als Features hinzu.

Berechnet Winrates für jeden Champion basierend auf historischen Daten
(NUR aus Trainingsdaten, um Data Leakage zu vermeiden).
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


def calculate_champion_winrates(df_train):
    """
    Berechnet Champion-Winrates aus Trainingsdaten.
    
    Returns:
        dict: {champion_id: winrate}
    """
    champion_winrates = {}
    
    # Sammle alle Champion-IDs
    champion_cols = [col for col in df_train.columns if any(pos in col for pos in ['top', 'jungle', 'mid', 'adc', 'support'])]
    
    all_champions = set()
    for col in champion_cols:
        unique_champs = df_train[col].unique()
        all_champions.update(unique_champs)
    all_champions.discard(-1)  # Entferne "fehlende" Champions
    
    print(f"Berechne Winrates für {len(all_champions)} Champions...")
    
    # Für jeden Champion: Berechne Winrate
    for champ_id in all_champions:
        wins = 0
        total = 0
        
        # Prüfe Team 1 Champions
        for pos in ['top', 'jungle', 'mid', 'adc', 'support']:
            champ_matches = df_train[df_train[f'team1_{pos}'] == champ_id]
            if len(champ_matches) > 0:
                wins += champ_matches['target'].sum()  # target=1 bedeutet Team 1 gewinnt
                total += len(champ_matches)
        
        # Prüfe Team 2 Champions
        for pos in ['top', 'jungle', 'mid', 'adc', 'support']:
            champ_matches = df_train[df_train[f'team2_{pos}'] == champ_id]
            if len(champ_matches) > 0:
                wins += (len(champ_matches) - champ_matches['target'].sum())  # target=0 bedeutet Team 2 gewinnt
                total += len(champ_matches)
        
        if total > 0:
            winrate = wins / total
            champion_winrates[champ_id] = winrate
        else:
            champion_winrates[champ_id] = 0.5  # Default: 50% wenn keine Daten
    
    print(f"[OK] Winrates für {len(champion_winrates)} Champions berechnet")
    return champion_winrates


def add_winrate_features(df, champion_winrates):
    """
    Fügt Champion-Winrates als Features hinzu.
    
    Erstellt Features wie:
    - team1_avg_winrate: Durchschnittliche Winrate aller Champions in Team 1
    - team2_avg_winrate: Durchschnittliche Winrate aller Champions in Team 2
    - winrate_diff: Differenz (Team 1 - Team 2)
    """
    df = df.copy()
    
    # Funktion um Winrate eines Champions zu bekommen
    def get_champ_winrate(champ_id):
        if champ_id == -1:
            return 0.5  # Default für fehlende Champions
        return champion_winrates.get(champ_id, 0.5)
    
    # Team 1: Durchschnittliche Winrate aller Champions
    team1_champs = ['team1_top', 'team1_jungle', 'team1_mid', 'team1_adc', 'team1_support']
    # applymap ist deprecated (pandas 2.3+). Nutze apply + Series.map.
    team1_winrates = df[team1_champs].apply(lambda s: s.map(get_champ_winrate))
    df['team1_avg_winrate'] = team1_winrates.mean(axis=1)
    
    # Team 2: Durchschnittliche Winrate aller Champions
    team2_champs = ['team2_top', 'team2_jungle', 'team2_mid', 'team2_adc', 'team2_support']
    team2_winrates = df[team2_champs].apply(lambda s: s.map(get_champ_winrate))
    df['team2_avg_winrate'] = team2_winrates.mean(axis=1)
    
    # Winrate-Differenz (Team 1 - Team 2)
    df['winrate_diff'] = df['team1_avg_winrate'] - df['team2_avg_winrate']
    
    # Individuelle Champion-Winrates pro Position
    for pos in ['top', 'jungle', 'mid', 'adc', 'support']:
        df[f'team1_{pos}_winrate'] = df[f'team1_{pos}'].apply(get_champ_winrate)
        df[f'team2_{pos}_winrate'] = df[f'team2_{pos}'].apply(get_champ_winrate)
    
    print(f"[OK] {len([c for c in df.columns if 'winrate' in c])} Winrate-Features hinzugefügt")
    return df


def main():
    """Hauptfunktion: Lädt Daten, berechnet Stats, fügt Features hinzu."""
    project_root = Path(__file__).parent.parent.parent
    
    # Lade Dataset
    dataset_path = project_root / "data" / "datasets" / "59334_filtered_augmented_dataset.csv"
    if not dataset_path.exists():
        print(f"[ERROR] Dataset nicht gefunden: {dataset_path}")
        return
    
    print("Lade Dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Geladen: {len(df)} Matches")
    
    # Erstelle Gruppen für Group-basiertes Splitting
    match_ids = df["match_id"].astype(str)
    groups = match_ids.str.replace('_mirrored', '', regex=False)
    
    # Teile in Training und Test (70/30)
    # WICHTIG: Stats werden NUR aus Trainingsdaten berechnet!
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, _ = next(gss.split(df, df['target'], groups))
    
    df_train = df.iloc[train_idx].copy()
    print(f"Trainingsdaten für Winrates: {len(df_train)} Matches")
    
    # Berechne Champion-Winrates (NUR aus Trainingsdaten!)
    champion_winrates = calculate_champion_winrates(df_train)
    
    # Zeige Top 10 Champions mit höchster Winrate
    sorted_winrates = sorted(champion_winrates.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Champions (höchste Winrate):")
    for i, (champ_id, wr) in enumerate(sorted_winrates[:10], 1):
        print(f"  {i:2d}. Champion {champ_id:3d}: {wr*100:.2f}%")
    
    # Füge Features zu ALLEN Daten hinzu
    df_with_features = add_winrate_features(df, champion_winrates)
    
    # Speichere erweitertes Dataset
    output_path = project_root / "data" / "datasets" / "59334_with_winrates.csv"
    df_with_features.to_csv(output_path, index=False)
    print(f"\n[OK] Erweitertes Dataset gespeichert: {output_path}")
    print(f"  Neue Features: {len([c for c in df_with_features.columns if 'winrate' in c])}")


if __name__ == "__main__":
    main()

