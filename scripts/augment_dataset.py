"""
Erstellt augmentiertes Dataset durch Team-Spiegelung.

Dieses Script verdoppelt die Datenmenge, indem es jeden Match spiegelt
(Team 1 ↔ Team 2). Das hilft dem Modell zu lernen, dass Team-Position irrelevant ist.

Ausführung:
    python scripts/augment_dataset.py
"""

import sys
from pathlib import Path

# Füge src/preprocessing zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "preprocessing"))

from augment_dataset import main

if __name__ == "__main__":
    main()
