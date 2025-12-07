"""
Kleines Test Script um die Daten Sammlung zu testen. 10 Matches only.
"""

import sys
from pathlib import Path

# Füge src/data_collection zum path hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "data_collection"))

from collect_matches import collect_matches

if __name__ == "__main__":
    print("="*60)
    print("TEST: Sammle 10 Matches zum Testen")
    print("="*60)
    print()
    
    # Sammle nur 10 Matches aus einer Region (EUW) zum Testen
    collect_matches(
        target_matches=10,
        regions=["euw1"]  # Nur eine Region für den Test
    )
    
    print()
    print("="*60)
    print("Test abgeschlossen! Prüfe die data/raw/ Ordner für die JSON-Dateien.")
    print("="*60)

