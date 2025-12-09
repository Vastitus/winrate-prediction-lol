"""
Sammelt 50.000 Matches gleichmäßig verteilt über alle Regionen.

Dieses Script sammelt Match-Daten von Top-Spielern (Challenger, Grandmaster, Master)
und speichert sie als JSON-Dateien im data/raw Verzeichnis.

Hinweis: Dies kann mehrere Stunden dauern!

Ausführung:
    python scripts/collect_50k_matches.py
"""

import sys
from pathlib import Path

# Füge src/data_collection zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "data_collection"))

from collect_matches import collect_matches

if __name__ == "__main__":
    # Prüfe wie viele Matches bereits vorhanden sind
    raw_dir = project_root / "data" / "raw"
    existing_count = len(list(raw_dir.glob("*.json"))) if raw_dir.exists() else 0
    
    print("=" * 60)
    print("SAMMLE 50.000 MATCHES")
    print("=" * 60)
    print(f"Bereits vorhanden: {existing_count} Matches")
    print(f"Ziel: 50.000 Matches")
    print(f"Noch benötigt: {max(0, 50000 - existing_count)} Matches")
    print("=" * 60)
    print()
    
    collect_matches(target_matches=50000, use_parallel=True)
    print("\nFertig!")

