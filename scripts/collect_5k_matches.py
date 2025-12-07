"""
Sammelt 5k Matches als Test für Model-Training.
"""

import sys
from pathlib import Path

# Füge src/data_collection zum path hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "data_collection"))

from collect_matches import collect_matches

if __name__ == "__main__":
    print("="*60)
    print("SAMMLE 5K MATCHES ZUM TESTEN")
    print("="*60)
    print()
    print("Dies wird ca. 10-15 Minuten dauern (mit allen Regionen)...")
    print()
    
    # Sammle 5k Matches aus allen Regionen mit Parallelisierung
    collect_matches(
        target_matches=5000,
        use_parallel=True  # Nutzt Threading für verschiedene Routing Values
    )
    
    print()
    print("="*60)
    print("Fertig! Prüfe data/raw/ für die JSON-Dateien.")
    print("Dann führe aus: py src/preprocessing/create_dataset.py")
    print("="*60)

