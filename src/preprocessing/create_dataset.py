"""
Erstellt strukturiertes Dataset aus rohen Match-Daten.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict


def load_match_files(data_dir: Path) -> List[Dict]:
    """Lädt alle Match-JSON Dateien."""
    match_files = list(data_dir.glob("*.json"))
    matches = []
    total = len(match_files)
    
    print(f"Lade {total} JSON-Dateien...")
    for i, file_path in enumerate(match_files, 1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                matches.append(json.load(f))
            
            # Status alle 1000 Dateien
            if i % 1000 == 0 or i == total:
                print(f"  Geladen: {i}/{total} Dateien ({len(matches)} erfolgreich)")
        except Exception:
            continue  # Fehler ignorieren und weitermachen
    
    return matches


def extract_features(match_data: Dict) -> Dict:
    """Extrahiert Features aus Match-Daten."""
    info = match_data.get("info", {})
    metadata = match_data.get("metadata", {})
    
    features = {
        "match_id": metadata.get("matchId"),
        "game_version": info.get("gameVersion"),
        "queue_id": info.get("queueId"),
        "game_duration": info.get("gameDuration"),
        "game_mode": info.get("gameMode"),
    }
    
    # Teams
    teams = info.get("teams", [])
    for i, team in enumerate(teams, 1):
        features[f"team{i}_win"] = 1 if team.get("win") else 0
        
        # Bans
        bans = team.get("bans", [])
        for j, ban in enumerate(bans[:5], 1):  # Max 5 Bans
            features[f"team{i}_ban_{j}"] = ban.get("championId", -1)
    
    # Participants mit Positionen
    participants = info.get("participants", [])
    
    # Position Mapping (Riot API verwendet diese Werte)
    position_map = {
        "TOP": "top",
        "JUNGLE": "jungle",
        "MIDDLE": "mid",
        "BOTTOM": "adc",
        "UTILITY": "support"
    }
    
    # Organisiere Champions nach Team und Position
    team1_champions = {}
    team2_champions = {}
    
    for participant in participants:
        team_id = participant.get("teamId", 100)
        champion_id = participant.get("championId")
        # Verwende individualPosition (genauer) oder teamPosition als Fallback
        position = participant.get("individualPosition", participant.get("teamPosition", ""))
        
        # Normalisiere Position
        position_normalized = position_map.get(position, "").lower()
        
        if not position_normalized:
            # Skip wenn keine gültige Position
            continue
        
        if team_id == 100:
            team1_champions[position_normalized] = champion_id
        elif team_id == 200:
            team2_champions[position_normalized] = champion_id
    
    # Team 1 Champions nach Position
    for pos in ["top", "jungle", "mid", "adc", "support"]:
        features[f"team1_{pos}"] = team1_champions.get(pos, -1)
    
    # Team 2 Champions nach Position
    for pos in ["top", "jungle", "mid", "adc", "support"]:
        features[f"team2_{pos}"] = team2_champions.get(pos, -1)
    
    # Target Variable: Team 1 gewinnt (1) oder Team 2 gewinnt (0)
    features["target"] = features.get("team1_win", 0)
    
    # Validiere, dass alle Positionen vorhanden sind (für beide Teams)
    required_positions = ["top", "jungle", "mid", "adc", "support"]
    team1_complete = all(features.get(f"team1_{pos}", -1) != -1 for pos in required_positions)
    team2_complete = all(features.get(f"team2_{pos}", -1) != -1 for pos in required_positions)
    
    # Setze Flag für vollständige Daten
    features["has_complete_positions"] = team1_complete and team2_complete
    
    return features


def create_dataset(raw_data_dir: Path, output_path: Path, format: str = "parquet"):
    """Erstellt strukturiertes Dataset aus rohen Daten."""
    print("=" * 60)
    print("ERSTELLE DATASET")
    print("=" * 60)
    print(f"Input-Verzeichnis: {raw_data_dir}")
    print(f"Output-Datei: {output_path}")
    print("=" * 60)
    print()
    
    matches = load_match_files(raw_data_dir)
    
    print(f"Gefunden: {len(matches)} Matches")
    print(f"Verarbeite Matches...")
    features_list = []
    total = len(matches)
    
    for i, match in enumerate(matches, 1):
        try:
            features = extract_features(match)
            # Nur Matches mit vollständigen Positionen verwenden
            if features.get("has_complete_positions", False):
                features_list.append(features)
            
            # Status alle 1000 Matches
            if i % 1000 == 0 or i == total:
                print(f"  Verarbeitet: {i}/{total} Matches ({len(features_list)} mit vollständigen Daten)")
        except Exception:
            continue  # Fehler ignorieren und weitermachen
    
    print()
    print("Erstelle DataFrame...")
    df = pd.DataFrame(features_list)
    
    print(f"Dataset Shape: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    
    # Speichere Dataset
    print(f"Speichere Dataset nach {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "parquet":
        df.to_parquet(output_path, compression="snappy", index=False)
    elif format == "csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unbekanntes Format: {format}")
    
    print(f"✓ Dataset erfolgreich gespeichert!")
    
    # Zeige Statistiken
    print()
    print("=" * 60)
    print("DATASET STATISTIKEN")
    print("=" * 60)
    print(f"Gesamt Matches: {len(df)}")
    print(f"Target Verteilung:")
    target_counts = df["target"].value_counts()
    print(f"  Team 1 gewinnt: {target_counts.get(1, 0)}")
    print(f"  Team 2 gewinnt: {target_counts.get(0, 0)}")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    # Pfade relativ zum Projekt-Root
    project_root = Path(__file__).parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"
    output_path = project_root / "data" / "datasets" / "lol_matches.csv"
    
    create_dataset(raw_data_dir, output_path, format="csv")

