"""
Hauptskript für die Sammlung von Match-Daten.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Füge parent directory zum path hinzu, damit imports funktionieren
sys.path.append(str(Path(__file__).parent))
from riot_api import RiotAPI


def save_match_data(match_data: Dict, output_dir: Path):
    """Speichert Match-Daten als JSON."""
    match_id = match_data.get("metadata", {}).get("matchId", "unknown")
    output_file = output_dir / f"{match_id}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(match_data, f, indent=2, ensure_ascii=False)


def collect_matches(api: RiotAPI, target_matches: int = 200000, output_dir: Path = None):
    """Sammelt Matches von Top-Spielern."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Hole Top-Spieler...")
    # Sammle Spieler aus verschiedenen Rängen
    players = []
    players.extend(api.get_challenger_players())
    players.extend(api.get_grandmaster_players())
    players.extend(api.get_master_players())
    
    print(f"Gefunden: {len(players)} Spieler")
    
    collected_matches = 0
    seen_match_ids = set()
    
    # Hole PUUIDs für alle Spieler
    print("Hole Spieler-Informationen...")
    player_puuids = []
    for player in tqdm(players[:1000], desc="Hole PUUIDs"):  # Limitiere auf 1000 Spieler für Start
        try:
            puuid = api.get_player_puuid(player["summonerId"])
            player_puuids.append(puuid)
        except Exception as e:
            print(f"Fehler beim Abrufen von PUUID für {player['summonerId']}: {e}")
            continue
    
    print(f"Verarbeite {len(player_puuids)} Spieler...")
    
    # Sammle Matches
    for puuid in tqdm(player_puuids, desc="Sammle Matches"):
        if collected_matches >= target_matches:
            break
        
        try:
            match_ids = api.get_match_ids(puuid, count=100, queue=420)  # Ranked Solo/Duo
            
            for match_id in match_ids:
                if match_id in seen_match_ids:
                    continue
                
                if collected_matches >= target_matches:
                    break
                
                try:
                    match_data = api.get_match_details(match_id)
                    save_match_data(match_data, output_dir)
                    seen_match_ids.add(match_id)
                    collected_matches += 1
                    
                    if collected_matches % 100 == 0:
                        print(f"\nGesammelt: {collected_matches}/{target_matches} Matches")
                
                except Exception as e:
                    print(f"Fehler beim Abrufen von Match {match_id}: {e}")
                    continue
        
        except Exception as e:
            print(f"Fehler beim Abrufen von Matches für Spieler {puuid}: {e}")
            continue
    
    print(f"\nFertig! {collected_matches} Matches gesammelt.")


if __name__ == "__main__":
    api = RiotAPI()
    collect_matches(api, target_matches=200000)

