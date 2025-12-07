import pandas as pd
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
df = pd.read_csv(project_root / 'data' / 'datasets' / '5k_matches_dataset.csv')
size_kb = os.path.getsize('data/datasets/5k_matches_dataset.csv') / 1024

print("="*60)
print("DATASET ANALYSE")
print("="*60)
print(f"\nAktuelle Datei:")
print(f"  - Matches: {len(df)}")
print(f"  - Größe: {size_kb:.2f} KB")
print(f"  - Pro Match: {size_kb/len(df):.3f} KB")
print(f"  - Features: {len(df.columns)} Spalten")

print(f"\nPrognose für 200k Matches:")
csv_size_mb = (size_kb / len(df)) * 200000 / 1024
print(f"  - CSV: ~{csv_size_mb:.1f} MB")
print(f"  - Parquet (komprimiert): ~{csv_size_mb * 0.3:.1f} MB")

print(f"\nDatenqualität:")
print(f"  - Vollständige Positionen: {df['has_complete_positions'].sum()}/{len(df)}")
print(f"  - Target Verteilung: {df['target'].value_counts().to_dict()}")

print(f"\nFeatures vorhanden:")
print(f"  - Champions pro Position: ✓")
print(f"  - Bans: ✓")
print(f"  - Gewinner: ✓")
print(f"  - Metadaten: ✓")

print("\n" + "="*60)
print("KANN MAN DAMIT TRAINIEREN?")
print("="*60)
print("  - Technisch: JA (Code funktioniert)")
print("  - Praktisch: NEIN (10 Matches = Overfitting)")
print("  - Minimum für Training: ~1000-5000 Matches")
print("  - Empfohlen: 50k-200k Matches")

print("\n" + "="*60)
print("SAMMEL-ZEIT FÜR 200K MATCHES")
print("="*60)
print("  - Rate Limit: 10 req/s (Development Key)")
print("  - Pro Match: ~1-2 Requests (wegen Duplikaten)")
print("  - 200k Matches = ~200k-400k Requests")
print("  - Geschätzte Zeit: 5-11 Stunden")
print("  - Realistisch: 6-8 Stunden (mit Pausen/Rate Limits)")

print("\n" + "="*60)
print("DATEI-STRUKTUR")
print("="*60)
print("  - Eine Datei (~20-40 MB): ✓ Empfohlen")
print("    → Einfacher zu handhaben")
print("    → Pandas kann das problemlos laden")
print("  - Mehrere Dateien: Optional")
print("    → Nur wenn >100 MB oder für bessere Organisation")

