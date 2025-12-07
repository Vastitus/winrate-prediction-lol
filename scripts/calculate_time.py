"""
Berechnet die geschätzte Zeit für Match-Sammlung basierend auf Riot API Rate Limits.
"""

print("="*60)
print("RIOT API RATE LIMITS")
print("="*60)
print("  - 20 requests pro 1 Sekunde")
print("  - 100 requests pro 2 Minuten")
print("  - Pro Routing Value (Region)")
print()

# Rate Limits
requests_per_2min = 100
requests_per_min = requests_per_2min / 2
requests_per_hour = requests_per_min * 60

print(f"Rate Limit: {requests_per_min:.1f} requests/Minute = {requests_per_hour:.0f} requests/Stunde pro Region")
print()

# Für 5k Matches
target_matches = 5000
requests_per_match = 1.5  # Realistisch wegen Duplikaten
total_requests = target_matches * requests_per_match

print("="*60)
print("FÜR 5K MATCHES")
print("="*60)
print(f"  - Benötigte Requests: ~{total_requests:.0f}")
print(f"  - Mit 1 Region: ~{total_requests / requests_per_hour:.1f} Stunden")
print(f"  - Mit 3 Regionen: ~{total_requests / (requests_per_hour * 3):.1f} Stunden")
print(f"  - Mit allen Regionen (~15): ~{total_requests / (requests_per_hour * 15):.1f} Stunden")
print()

# Für 200k Matches
target_matches_200k = 200000
total_requests_200k = target_matches_200k * requests_per_match

print("="*60)
print("FÜR 200K MATCHES")
print("="*60)
print(f"  - Benötigte Requests: ~{total_requests_200k:.0f}")
print(f"  - Mit 1 Region: ~{total_requests_200k / requests_per_hour:.1f} Stunden")
print(f"  - Mit 3 Regionen: ~{total_requests_200k / (requests_per_hour * 3):.1f} Stunden")
print(f"  - Mit allen Regionen (~15): ~{total_requests_200k / (requests_per_hour * 15):.1f} Stunden")
print()

print("="*60)
print("EMPFEHLUNG")
print("="*60)
print("  - 5k Matches: 3 Regionen → ~20-30 Minuten")
print("  - 200k Matches: Alle Regionen → ~2-3 Stunden (über Nacht)")
print()

