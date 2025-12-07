"""Sammelt 5000 Matches gleichmäßig verteilt über alle Regionen."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "data_collection"))

from collect_matches import collect_matches

if __name__ == "__main__":
    print("Sammle 5000 Matches...")
    collect_matches(target_matches=5000, use_parallel=True)
    print("Fertig!")

