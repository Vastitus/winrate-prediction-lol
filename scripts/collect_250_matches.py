"""
Sammelt 250 Matches gleichmäßig verteilt über alle Regionen.

Dieses Script sammelt Match-Daten von Top-Spielern (Challenger, Grandmaster, Master)
und speichert sie als JSON-Dateien im data/raw Verzeichnis.

Ausführung:
    python scripts/collect_250_matches.py
"""

import sys
from pathlib import Path

# Füge src/data_collection zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "data_collection"))

from collect_matches import collect_matches

if __name__ == "__main__":
    print("Sammle 250 Matches...")
    collect_matches(target_matches=250, use_parallel=True)
    print("Fertig!")
