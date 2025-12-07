"""
Berechnet Speicherplatz-Bedarf für Datasets.
"""

import os
from pathlib import Path

# Aktuelle JSONs analysieren
raw_dir = Path("data/raw")
json_files = list(raw_dir.glob("*.json"))

if json_files:
    total_size = sum(f.stat().st_size for f in json_files)
    avg_size_kb = (total_size / len(json_files)) / 1024
    total_size_mb = total_size / (1024 * 1024)
    
    print("="*60)
    print("AKTUELLE DATEN")
    print("="*60)
    print(f"JSON-Dateien: {len(json_files)}")
    print(f"Gesamt: {total_size_mb:.2f} MB")
    print(f"Durchschnitt: {avg_size_kb:.1f} KB pro JSON")
    print()
    
    # Test-Dataset analysieren
    test_csv = Path("data/datasets/5k_matches_dataset.csv")
    if test_csv.exists():
        test_size_kb = test_csv.stat().st_size / 1024
        test_matches = 10
        avg_csv_kb = test_size_kb / test_matches
        
        print("="*60)
        print("PROGNOSE FÜR 200K MATCHES")
        print("="*60)
        print(f"Raw JSONs: ~{avg_size_kb * 200000 / 1024 / 1024:.1f} GB")
        print(f"CSV Dataset: ~{avg_csv_kb * 200000 / 1024:.1f} MB")
        print(f"Parquet (komprimiert): ~{avg_csv_kb * 200000 * 0.3 / 1024:.1f} MB")
        print()
        
        print("="*60)
        print("SPEICHERPLATZ-OPTIMIERUNG")
        print("="*60)
        print("Option 1: JSONs nach Dataset-Erstellung löschen")
        print(f"  → Spart: ~{avg_size_kb * 200000 / 1024 / 1024:.1f} GB")
        print(f"  → Behält: ~{avg_csv_kb * 200000 / 1024:.1f} MB (CSV)")
        print()
        print("Option 2: Parquet statt CSV verwenden")
        print(f"  → CSV: ~{avg_csv_kb * 200000 / 1024:.1f} MB")
        print(f"  → Parquet: ~{avg_csv_kb * 200000 * 0.3 / 1024:.1f} MB")
        print(f"  → Spart: ~{avg_csv_kb * 200000 * 0.7 / 1024:.1f} MB")
        print()
        print("Option 3: JSONs auf anderes Laufwerk verschieben")
        print("  → z.B. D: oder externes Laufwerk")
        print()
        
        print("="*60)
        print("ÜBERTRAGUNG AUF LAPTOP")
        print("="*60)
        print("Option 1: GitHub (nur Code, nicht Daten)")
        print("  → Dataset zu groß für GitHub")
        print()
        print("Option 2: Externes Laufwerk/USB")
        print(f"  → CSV: ~{avg_csv_kb * 200000 / 1024:.1f} MB")
        print(f"  → Parquet: ~{avg_csv_kb * 200000 * 0.3 / 1024:.1f} MB")
        print()
        print("Option 3: Cloud (Google Drive, Dropbox, etc.)")
        print(f"  → Upload: ~{avg_csv_kb * 200000 / 1024:.1f} MB (CSV)")
        print()
        print("Option 4: Dataset auf beiden Geräten neu erstellen")
        print("  → Code auf GitHub, dann auf Laptop: py scripts/collect_5k_matches.py")
        print("  → Oder: Nur Dataset übertragen (kleiner)")
else:
    print("Keine JSON-Dateien gefunden in data/raw/")

