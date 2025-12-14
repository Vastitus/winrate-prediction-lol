"""
Hauptskript für die Sammlung von Match-Daten von der Riot Games API.

Dieses Modul sammelt Match-Daten von Top-Spielern (Challenger, Grandmaster, Master)
aus verschiedenen Regionen und speichert sie als JSON-Dateien.
"""

import json
import sys
import threading
from pathlib import Path
from typing import List, Dict, Optional

# Import-Pfad für riot_api Modul
sys.path.append(str(Path(__file__).parent))
from riot_api import RiotAPI, REGION_CONFIGS


def save_match_data(match_data: Dict, output_dir: Path):
    """
    Speichert Match-Daten als JSON-Datei.
    
    Die Datei wird nach der Match-ID benannt (z.B. "EUW1_123456.json").
    Prüft vor dem Speichern, ob die Datei bereits existiert (verhindert Duplikate).
    """
    match_id = match_data.get("metadata", {}).get("matchId", "unknown")
    output_file = output_dir / f"{match_id}.json"
    
    # Zusätzliche Sicherheit: Prüfe ob Datei bereits existiert
    if output_file.exists():
        return  # Datei existiert bereits, überspringe
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(match_data, f, indent=2, ensure_ascii=False)


def collect_matches_from_region(api: RiotAPI, target_matches: int, seen_match_ids: set, 
                                 output_dir: Path, collected_matches_ref: List[int], 
                                 max_players_per_region: int = 2000,
                                 max_matches_per_region: Optional[int] = None,
                                 lock: Optional[threading.Lock] = None) -> int:
    """
    Sammelt Matches von einer spezifischen Region.
    
    Args:
        api: RiotAPI Instanz für die Region
        target_matches: Gesamtziel an Matches
        seen_match_ids: Set mit bereits gesammelten Match-IDs (verhindert Duplikate)
        output_dir: Verzeichnis zum Speichern der JSON-Dateien
        collected_matches_ref: Liste mit einem Element [count] für Thread-sichere Zählung
        max_players_per_region: Max. Anzahl Spieler pro Region
        max_matches_per_region: Max. Matches aus dieser Region (None = kein Limit)
        lock: Thread-Lock für Thread-sichere Operationen
    
    Returns:
        Aktuelle Anzahl gesammelter Matches
    """
    # Hole Top-Spieler aus Challenger, Grandmaster und Master Rängen
    players = []
    try:
        players.extend(api.get_challenger_players())
        players.extend(api.get_grandmaster_players())
        players.extend(api.get_master_players())
    except Exception:
        return collected_matches_ref[0]
    
    if not players:
        return collected_matches_ref[0]
    
    # Extrahiere PUUIDs (Player Unique IDs) aus den Spieler-Daten
    player_puuids = []
    for player in players[:max_players_per_region]:
        # Prüfe ob Ziel bereits erreicht (Thread-sicher)
        if lock:
            with lock:
                if collected_matches_ref[0] >= target_matches:
                    break
        
        puuid = player.get("puuid")
        if puuid:
            player_puuids.append(puuid)
    
    # Sammle Matches von jedem Spieler
    matches_from_this_region = 0
    for puuid in player_puuids:
        # Prüfe ob Ziel erreicht oder Region-Limit überschritten
        if lock:
            with lock:
                if collected_matches_ref[0] >= target_matches:
                    break
                if max_matches_per_region and matches_from_this_region >= max_matches_per_region:
                    break
        
        try:
            # Passe Anzahl der Match-IDs an: Wenn wir schon viele haben, hole weniger pro Spieler
            # um schneller neue zu finden
            if lock:
                with lock:
                    current_total = collected_matches_ref[0]
            else:
                current_total = collected_matches_ref[0]
            
            # Bei vielen Matches: viel weniger Match-IDs pro Spieler für schnellere Suche
            if current_total > 30000:
                match_count = 10  # Sehr wenig bei sehr vielen Matches
            elif current_total > 10000:
                match_count = 15
            elif current_total > 5000:
                match_count = 30
            else:
                match_count = 100
            
            # Hole Match-IDs für diesen Spieler (Queue 420 = Ranked Solo/Duo)
            match_ids = api.get_match_ids(puuid, count=match_count, queue=420)
            
            # Zähle Duplikate - wenn zu viele, überspringe diesen Spieler sehr früh
            duplicates_found = 0
            new_matches_found = 0
            checked_count = 0
            
            for match_id in match_ids:
                checked_count += 1
                
                # Thread-sicher: Prüfe ob Match bereits gesammelt wurde
                is_duplicate = False
                if lock:
                    with lock:
                        if match_id in seen_match_ids:
                            is_duplicate = True
                            duplicates_found += 1
                        else:
                            if collected_matches_ref[0] >= target_matches:
                                break
                            if max_matches_per_region and matches_from_this_region >= max_matches_per_region:
                                break
                            seen_match_ids.add(match_id)
                            collected_matches_ref[0] += 1
                            current_count = collected_matches_ref[0]
                            new_matches_found += 1
                else:
                    if match_id in seen_match_ids:
                        is_duplicate = True
                        duplicates_found += 1
                    else:
                        seen_match_ids.add(match_id)
                        collected_matches_ref[0] += 1
                        current_count = collected_matches_ref[0]
                        new_matches_found += 1
                
                # Aggressiv abbrechen: Wenn die ersten 5 alle Duplikate sind, direkt zum nächsten Spieler
                if checked_count >= 5 and duplicates_found == checked_count and new_matches_found == 0:
                    break  # Alle ersten 5 sind Duplikate → nächster Spieler
                
                # Wenn nach 10 geprüften Match-IDs mehr als 80% Duplikate und keine neuen gefunden
                if checked_count >= 10:
                    duplicate_ratio = duplicates_found / checked_count
                    if duplicate_ratio >= 0.8 and new_matches_found == 0:
                        break  # Zu viele Duplikate, nächster Spieler
                
                if is_duplicate:
                    continue
                
                # Hole Match-Details und speichere sie
                try:
                    match_data = api.get_match_details(match_id)
                    save_match_data(match_data, output_dir)
                    matches_from_this_region += 1
                    
                    # Status-Output alle 1000 Matches
                    if current_count % 1000 == 0:
                        print(f"Gesammelt: {current_count}/{target_matches} Matches")
                
                except Exception:
                    continue  # Fehler ignorieren und weitermachen
        
        except Exception:
            continue  # Fehler ignorieren und weitermachen
    
    return collected_matches_ref[0]


def collect_matches_worker(api_key: str, routing_value: str, regions_in_routing: List[str],
                           target_matches: int, seen_match_ids: set, output_dir: Path,
                           collected_matches: List[int], lock: threading.Lock, 
                           total_regions: int):
    """
    Worker-Thread für parallele Sammlung aus einem Routing Value.
    
    Diese Funktion läuft in einem separaten Thread und sammelt Matches aus mehreren
    Regionen, die dasselbe Routing Value teilen (z.B. "europe" für EUW1, EUN1, etc.).
    Sie verteilt die Matches gleichmäßig über alle Regionen.
    """
    iteration = 0
    max_iterations = 200  # Sicherheit gegen Endlosschleifen
    
    while collected_matches[0] < target_matches and iteration < max_iterations:
        # Wähle Region, die noch Matches braucht (gleichmäßige Verteilung)
        with lock:
            if collected_matches[0] >= target_matches:
                break
            
            # Zähle aktuelle Matches pro Region
            region_counts = get_matches_per_region(output_dir)
            target_per_region = max(1, target_matches // total_regions)
            
            # Sortiere Regionen nach Anzahl (wenigste zuerst)
            regions_sorted = sorted(regions_in_routing, 
                                   key=lambda r: region_counts.get(r, 0))
            
            # Wähle Region, die noch unter dem Ziel liegt
            selected_region = None
            for region in regions_sorted:
                if region_counts.get(region, 0) < target_per_region:
                    selected_region = region
                    break
            
            # Falls alle Regionen Ziel erreicht haben, nimm die mit wenigsten Matches
            if selected_region is None and collected_matches[0] < target_matches:
                selected_region = regions_sorted[0] if regions_sorted else None
        
        if selected_region is None:
            break
        
        # Sammle Matches aus der ausgewählten Region
        try:
            api = RiotAPI(api_key=api_key, region=selected_region)
            
            # Berechne wie viele Matches noch aus dieser Region benötigt werden
            with lock:
                region_counts = get_matches_per_region(output_dir)
                current_count = region_counts.get(selected_region, 0)
                remaining = max(1, target_per_region - current_count + 10)  # +10 Puffer
            
            collect_matches_from_region(
                api, target_matches, seen_match_ids, output_dir, collected_matches,
                max_matches_per_region=remaining,
                lock=lock
            )
        except Exception:
            pass  # Fehler ignorieren und weitermachen
        
        iteration += 1


def load_existing_match_ids(output_dir: Path) -> set:
    """
    Lädt bereits existierende Match-IDs aus JSON-Dateien.
    
    Ermöglicht Resume nach Unterbrechung - bereits gesammelte Matches werden
    übersprungen. Die Match-ID ist im Dateinamen enthalten (z.B. "EUW1_123456.json").
    """
    existing_ids = set()
    json_files = list(output_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            match_id = json_file.stem  # Dateiname ohne .json Extension
            existing_ids.add(match_id)
        except Exception:
            continue
    
    return existing_ids


def get_matches_per_region(output_dir: Path) -> Dict[str, int]:
    """
    Zählt wie viele Matches pro Region bereits gesammelt wurden.
    
    Extrahiert die Region aus dem Dateinamen (Format: "REGION_MATCHID.json").
    Returns ein Dictionary: {"euw1": 50, "na1": 45, ...}
    """
    region_counts = {}
    
    for json_file in output_dir.glob("*.json"):
        try:
            filename = json_file.stem
            parts = filename.split("_")
            if len(parts) >= 2:
                region = parts[0].lower()  # Erster Teil ist die Region
                region_counts[region] = region_counts.get(region, 0) + 1
        except Exception:
            continue
    
    return region_counts


def collect_matches(api_key: Optional[str] = None, target_matches: int = 50000, 
                    output_dir: Path = None, regions: Optional[List[str]] = None,
                    use_parallel: bool = True):
    """
    Hauptfunktion zum Sammeln von Matches.
    
    Args:
        api_key: Riot API Key (optional, wird aus .env oder config.py geladen)
        target_matches: Anzahl der zu sammelnden Matches
        output_dir: Verzeichnis zum Speichern (Standard: data/raw)
        regions: Liste der Regionen (Standard: alle verfügbaren)
        use_parallel: Ob parallele Sammlung verwendet werden soll
    
    Die Funktion sammelt Matches gleichmäßig über alle Regionen verteilt.
    """
    # Standard-Output-Verzeichnis
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard: Alle Regionen verwenden
    if regions is None:
        regions = list(REGION_CONFIGS.keys())
    
    # Gruppiere Regionen nach Routing Values (für parallele Sammlung)
    # Regionen mit gleichem Routing Value können parallel verarbeitet werden
    routing_groups = {}
    for region in regions:
        routing = REGION_CONFIGS[region]["v5_region"]
        if routing not in routing_groups:
            routing_groups[routing] = []
        routing_groups[routing].append(region)
    
    # Lade bereits existierende Matches (für Resume)
    seen_match_ids = load_existing_match_ids(output_dir)
    collected_matches = [len(seen_match_ids)]  # Liste für Thread-sichere Zählung
    lock = threading.Lock()
    
    # Parallele Sammlung: Ein Thread pro Routing Value
    if use_parallel and len(routing_groups) > 1:
        threads = []
        for routing_value, regions_in_routing in routing_groups.items():
            thread = threading.Thread(
                target=collect_matches_worker,
                args=(api_key, routing_value, regions_in_routing, target_matches,
                      seen_match_ids, output_dir, collected_matches, lock, 
                      len(regions)),
                daemon=True
            )
            threads.append(thread)
            thread.start()
        
        # Warte bis alle Threads fertig sind
        for thread in threads:
            thread.join()
    else:
        # Sequenzielle Sammlung (Fallback)
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
            except Exception:
                pass
            
            region_index = (region_index + 1) % len(regions)
            iteration += 1
            
            if collected_matches[0] >= target_matches:
                break
    
    print(f"Fertig: {collected_matches[0]} Matches gesammelt")


if __name__ == "__main__":
    collect_matches(target_matches=50000)

