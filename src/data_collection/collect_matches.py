"""
Hauptskript für die Sammlung von Match-Daten.
"""

import json
import os
import sys
import threading
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
                                 output_dir: Path, collected_matches_ref: List[int], 
                                 max_players_per_region: int = 500,
                                 max_matches_per_region: Optional[int] = None,
                                 lock: Optional[threading.Lock] = None) -> int:
    """Sammelt Matches von einer spezifischen Region."""
    # Hole Top-Spieler aus dieser Region
    players = []
    try:
        players.extend(api.get_challenger_players())
        players.extend(api.get_grandmaster_players())
        players.extend(api.get_master_players())
    except Exception as e:
        print(f"Fehler {api.region}: {e}")
        return collected_matches_ref[0]
    
    if len(players) == 0:
        return collected_matches_ref[0]
    
    # Hole PUUIDs für Spieler (sind bereits in den Player-Daten enthalten)
    player_puuids = []
    for player in tqdm(players[:max_players_per_region], desc=f"{api.region.upper()}", leave=False):
        with lock if lock else threading.Lock():
            if collected_matches_ref[0] >= target_matches:
                break
        
        puuid = player.get("puuid")
        if puuid:
            player_puuids.append(puuid)
    
    # Berechne wie viele Matches noch aus dieser Region gesammelt werden sollen
    matches_from_this_region = 0
    if max_matches_per_region:
        remaining_for_region = max_matches_per_region
    else:
        remaining_for_region = target_matches  # Kein Limit
    
    # Sammle Matches
    for puuid in tqdm(player_puuids, desc=f"Matches {api.region}"):
        with lock if lock else threading.Lock():
            if collected_matches_ref[0] >= target_matches:
                break
            if max_matches_per_region and matches_from_this_region >= max_matches_per_region:
                break
        
        try:
            match_ids = api.get_match_ids(puuid, count=100, queue=420)  # Ranked Solo/Duo
            
            for match_id in match_ids:
                # Thread-safe: Prüfe ob Match schon gesammelt wurde
                with lock if lock else threading.Lock():
                    if match_id in seen_match_ids:
                        continue
                    if collected_matches_ref[0] >= target_matches:
                        break
                    if max_matches_per_region and matches_from_this_region >= max_matches_per_region:
                        break
                    seen_match_ids.add(match_id)
                    collected_matches_ref[0] += 1
                    current_count = collected_matches_ref[0]
                
                if current_count >= target_matches:
                    break
                if max_matches_per_region and matches_from_this_region >= max_matches_per_region:
                    break
                
                try:
                    match_data = api.get_match_details(match_id)
                    save_match_data(match_data, output_dir)
                    matches_from_this_region += 1
                    
                    if current_count % 1000 == 0:
                        print(f"Gesammelt: {current_count}/{target_matches} Matches")
                
                except Exception as e:
                    continue
        
        except Exception as e:
            continue
    
    return collected_matches_ref[0]


def collect_matches_worker(api_key: str, routing_value: str, regions_in_routing: List[str],
                           target_matches: int, seen_match_ids: set, output_dir: Path,
                           collected_matches: List[int], lock: threading.Lock, 
                           matches_per_routing: int, matches_per_region: int,
                           total_regions: int):
    """Worker-Thread für parallele Sammlung aus einem Routing Value."""
    # Rotiere durch alle Regionen, bevorzuge Regionen mit weniger Matches
    iteration = 0
    
    while collected_matches[0] < target_matches and iteration < 200:  # Max Iterationen als Sicherheit
        with lock:
            if collected_matches[0] >= target_matches:
                break
            
            # Hole aktuelle Verteilung
            region_counts = get_matches_per_region(output_dir)
            # Berechne Ziel pro Region: gleichmäßig über ALLE Regionen
            target_per_region = max(1, target_matches // total_regions)
            
            # Sortiere Regionen: zuerst die mit weniger Matches
            regions_sorted = sorted(regions_in_routing, 
                                   key=lambda r: region_counts.get(r, 0))
            
            # Wähle Region, die noch unter dem exakten Ziel liegt (strikt gleichmäßig)
            selected_region = None
            for region in regions_sorted:
                current_count = region_counts.get(region, 0)
                if current_count < target_per_region:  # Streng: nur unter dem Ziel
                    selected_region = region
                    break
            
            # Falls alle Regionen bereits das Ziel erreicht haben, prüfe ob wir noch Matches brauchen
            if selected_region is None:
                # Wenn noch Matches fehlen, nimm die mit den wenigsten (für Rest-Verteilung)
                if collected_matches[0] < target_matches:
                    selected_region = regions_sorted[0]
                else:
                    selected_region = None
        
        if selected_region is None:
            break
        
        try:
            api = RiotAPI(api_key=api_key, region=selected_region)
            # Berechne dynamisch, wie viele Matches noch aus dieser Region gesammelt werden sollen
            with lock:
                region_counts = get_matches_per_region(output_dir)
                current_count = region_counts.get(selected_region, 0)
                remaining_for_region = max(1, target_per_region - current_count + 10)  # +10 als Puffer
            
            collect_matches_from_region(
                api, target_matches, seen_match_ids, output_dir, collected_matches,
                max_matches_per_region=remaining_for_region,
                lock=lock
            )
        except Exception as e:
            print(f"Fehler mit Region {selected_region} ({routing_value}): {e}")
        
        iteration += 1
        
        with lock:
            if collected_matches[0] >= target_matches:
                break


def load_existing_match_ids(output_dir: Path) -> set:
    """Lädt bereits existierende Match-IDs aus JSON-Dateien (für Resume)."""
    existing_ids = set()
    json_files = list(output_dir.glob("*.json"))
    
    if json_files:
        print(f"Lade {len(json_files)} bereits existierende Matches für Resume...")
        for json_file in json_files:
            try:
                # Match-ID aus Dateiname extrahieren (z.B. "EUW1_123456.json" -> "EUW1_123456")
                match_id = json_file.stem
                existing_ids.add(match_id)
            except Exception:
                continue
    
    return existing_ids


def get_matches_per_region(output_dir: Path) -> Dict[str, int]:
    """Zählt wie viele Matches pro Region bereits gesammelt wurden."""
    region_counts = {}
    
    for json_file in output_dir.glob("*.json"):
        try:
            # Extrahiere Region aus Dateinamen (Format: REGION_MATCHID.json)
            filename = json_file.stem
            parts = filename.split("_")
            if len(parts) >= 2:
                region = parts[0].lower()
                region_counts[region] = region_counts.get(region, 0) + 1
        except Exception:
            continue
    
    return region_counts


def collect_matches(api_key: Optional[str] = None, target_matches: int = 200000, 
                    output_dir: Path = None, regions: Optional[List[str]] = None,
                    use_parallel: bool = True):
    """Sammelt Matches von Top-Spielern aus allen Regionen (parallel oder sequenziell)."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wenn keine Regionen angegeben, verwende alle
    if regions is None:
        regions = list(REGION_CONFIGS.keys())
    
    # Gruppiere Regionen nach Routing Values für Parallelisierung
    routing_groups = {}
    for region in regions:
        routing = REGION_CONFIGS[region]["v5_region"]
        if routing not in routing_groups:
            routing_groups[routing] = []
        routing_groups[routing].append(region)
    
    # Berechne Matches pro Routing Value und pro Region (gleichmäßig verteilt)
    matches_per_routing = max(1, target_matches // len(routing_groups))
    matches_per_region = max(1, target_matches // len(regions))
    
    # Lade bereits existierende Match-IDs (für Resume nach Unterbrechung)
    seen_match_ids = load_existing_match_ids(output_dir)
    if seen_match_ids:
        print(f"Resume: {len(seen_match_ids)} Matches vorhanden")
    
    collected_matches = [len(seen_match_ids)]  # Starte mit bereits vorhandenen Matches
    lock = threading.Lock()
    
    if use_parallel and len(routing_groups) > 1:
        # PARALLEL: Nutze Threading für verschiedene Routing Values
        threads = []
        for routing_value, regions_in_routing in routing_groups.items():
            thread = threading.Thread(
                target=collect_matches_worker,
                args=(api_key, routing_value, regions_in_routing, target_matches,
                      seen_match_ids, output_dir, collected_matches, lock, 
                      matches_per_routing, matches_per_region, len(regions)),
                daemon=True
            )
            threads.append(thread)
            thread.start()
        
        # Warte auf alle Threads
        for thread in threads:
            thread.join()
    else:
        # SEQUENZIELL: Fallback für einzelne Routing Values
        matches_per_region = max(1, target_matches // len(regions))
        region_index = 0
        iteration = 0
        while collected_matches[0] < target_matches and iteration < 100:
            region = regions[region_index]
            
            try:
                api = RiotAPI(api_key=api_key, region=region)
                collect_matches_from_region(
                    api, target_matches, seen_match_ids, output_dir, collected_matches,
                    max_matches_per_region=matches_per_region,
                    lock=lock
                )
            except Exception as e:
                print(f"Fehler mit Region {region}: {e}")
            
            region_index = (region_index + 1) % len(regions)
            iteration += 1
            
            if collected_matches[0] >= target_matches:
                break
    
    print(f"\nFertig: {collected_matches[0]} Matches gesammelt")


if __name__ == "__main__":
    collect_matches(target_matches=200000)

