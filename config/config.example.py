"""
Konfigurationsdatei für das Projekt.
Kopiere diese Datei zu config.py und fülle die Werte aus.
"""

# Riot API Konfiguration
RIOT_API_KEY = "dein_api_key_hier"
RIOT_API_BASE_URL = "https://euw1.api.riotgames.com"  # Für EUW Server

# Daten-Sammlung Einstellungen
TARGET_MATCHES = 200000  # Anzahl der gewünschten Matches
MAX_MATCHES_PER_PLAYER = 100  # Max. Matches pro Spieler
RANKED_QUEUE_IDS = [420, 440]  # Ranked Solo/Duo und Flex

# Dataset Einstellungen
DATASET_FORMAT = "parquet"  # "csv" oder "parquet"
COMPRESSION = "snappy"  # Für Parquet

# Model Einstellungen
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

