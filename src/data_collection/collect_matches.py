"""
Hauptskript für die Sammlung von Match-Daten.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Füge parent directory zum path hinzu, damit imports funktionieren
sys.path.append(str(Path(__file__).parent))
from riot_api import RiotAPI, REGION_CONFIGS


def save_match_data(match_data: Dict, output_dir: Path):
    """Speichert Match-Daten als JSON."""
    match_id = match_data.get("metadata", {}).get("matchId", "unknown")
    output_file = output_dir / f"{match_id}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(match_data, f, indent=2, ensure_ascii=False)


def collect_matches_from_region(api: RiotAPI, target_matches: int, seen_match_ids: set, 
                                 output_dir: Path, collected_matches: int, 
                                 max_players_per_region: int = 500) -> int:
    """Sammelt Matches von einer spezifischen Region."""
    print(f"\n{'='*60}")
    print(f"Region: {api.region.upper()}")
    print(f"{'='*60}")
    
    # Hole Top-Spieler aus dieser Region
    players = []
    try:
        players.extend(api.get_challenger_players())
        players.extend(api.get_grandmaster_players())
        players.extend(api.get_master_players())
    except Exception as e:
        print(f"Fehler beim Abrufen von Spielern für {api.region}: {e}")
        return collected_matches
    
    print(f"Gefunden: {len(players)} Spieler in {api.region}")
    
    if len(players) == 0:
        print(f"Keine Spieler in {api.region} gefunden. Überspringe...")
        return collected_matches
    
    # Hole PUUIDs für Spieler
    player_puuids = []
    for player in tqdm(players[:max_players_per_region], desc=f"PUUIDs {api.region}"):
        if collected_matches >= target_matches:
            break
        
        try:
            puuid = api.get_player_puuid(player["summonerId"])
            player_puuids.append(puuid)
        except Exception as e:
            continue
    
    print(f"Verarbeite {len(player_puuids)} Spieler aus {api.region}...")
    
    # Sammle Matches
    for puuid in tqdm(player_puuids, desc=f"Matches {api.region}"):
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
                        print(f"\nGesammelt: {collected_matches}/{target_matches} Matches (Region: {api.region})")
                
                except Exception as e:
                    continue
        
        except Exception as e:
            continue
    
    return collected_matches


def collect_matches(api_key: Optional[str] = None, target_matches: int = 200000, 
                    output_dir: Path = None, regions: Optional[List[str]] = None):
    """Sammelt Matches von Top-Spielern aus allen Regionen."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wenn keine Regionen angegeben, verwende alle
    if regions is None:
        regions = list(REGION_CONFIGS.keys())
    
    print(f"Sammle Matches aus {len(regions)} Regionen...")
    print(f"Ziel: {target_matches} Matches")
    
    collected_matches = 0
    seen_match_ids = set()
    
    # Durchlaufe alle Regionen
    for region in regions:
        if collected_matches >= target_matches:
            break
        
        try:
            api = RiotAPI(api_key=api_key, region=region)
            collected_matches = collect_matches_from_region(
                api, target_matches, seen_match_ids, output_dir, collected_matches
            )
        except Exception as e:
            print(f"Fehler mit Region {region}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Fertig! {collected_matches} Matches gesammelt.")
    print(f"{'='*60}")


if __name__ == "__main__":
    collect_matches(target_matches=200000)

