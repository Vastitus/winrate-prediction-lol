"""
Erstellt augmentiertes Dataset durch Team Spiegelung.
Also die Gamezahl wird verdoppelt und Sidebiased wird vermieden.

Ausführung: python scripts/augment_dataset.py
"""

import sys
from pathlib import Path

# Füge src/preprocessing zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "preprocessing"))

from augment_dataset import main

if __name__ == "__main__":
    main()
