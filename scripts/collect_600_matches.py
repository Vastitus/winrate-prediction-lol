"""
Script für parallele Match-Sammlung.
Sammelt 600 Matches gleichmäßig verteilt über alle Regionen.
"""

import sys
from pathlib import Path

# Füge src/data_collection zum path hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "data_collection"))

from collect_matches import collect_matches

if __name__ == "__main__":
    print("="*60)
    print("PARALLELE SAMMLUNG - 600 MATCHES")
    print("="*60)
    print()
    print("Sammelt 600 Matches gleichmäßig verteilt über alle Regionen.")
    print("Geschätzte Zeit: ~3-5 Minuten")
    print()
    
    # Sammle 600 Matches mit Parallelisierung (Standard)
    collect_matches(
        target_matches=600,
        use_parallel=True  # Nutzt Threading für verschiedene Routing Values
    )
    
    print()
    print("="*60)
    print("Sammlung abgeschlossen! Prüfe data/raw/ für die JSON-Dateien.")
    print("="*60)

