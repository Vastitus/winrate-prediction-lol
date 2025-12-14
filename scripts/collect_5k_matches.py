"""
5k Matches gleichverteilt über alle Regions.
Ich schaue nur auf die top Spieler, also Master+.
Ausführung: python scripts/collect_5k_matches.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "data_collection"))

from collect_matches import collect_matches

if __name__ == "__main__":
    collect_matches(target_matches=5000, use_parallel=True)

