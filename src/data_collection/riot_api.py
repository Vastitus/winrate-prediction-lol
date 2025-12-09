"""
Riot Games API Wrapper für Daten-Sammlung.

Dieses Modul stellt eine einfache Schnittstelle zur Riot Games API bereit
mit automatischem Rate Limiting und Fehlerbehandlung.

Wichtig: Die Riot API verwendet zwei verschiedene Systeme:
- V4 API (League, Summoner): Verwendet Region Codes (z.B. "euw1", "na1")
- V5 API (Match): Verwendet Routing Values (z.B. "europe", "americas")
"""

import requests
import time
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

load_dotenv()

# Versuche config.py zu laden (Fallback wenn .env nicht vorhanden)
try:
    sys.path.append(str(Path(__file__).parent.parent.parent / "config"))
    import config
except ImportError:
    config = None

# Region-Konfiguration für Riot API
# Jede Region hat einen V4 Base URL und ein V5 Routing Value
REGION_CONFIGS = {
    "euw1": {
        "v4_base": "https://euw1.api.riotgames.com",
        "v5_region": "europe"
    },
    "eun1": {
        "v4_base": "https://eun1.api.riotgames.com",
        "v5_region": "europe"
    },
    "na1": {
        "v4_base": "https://na1.api.riotgames.com",
        "v5_region": "americas"
    },
    "kr": {
        "v4_base": "https://kr.api.riotgames.com",
        "v5_region": "asia"
    },
    "jp1": {
        "v4_base": "https://jp1.api.riotgames.com",
        "v5_region": "asia"
    },
    "br1": {
        "v4_base": "https://br1.api.riotgames.com",
        "v5_region": "americas"
    },
    "la1": {
        "v4_base": "https://la1.api.riotgames.com",
        "v5_region": "americas"
    },
    "la2": {
        "v4_base": "https://la2.api.riotgames.com",
        "v5_region": "americas"
    },
    "oc1": {
        "v4_base": "https://oc1.api.riotgames.com",
        "v5_region": "sea"
    },
    "tr1": {
        "v4_base": "https://tr1.api.riotgames.com",
        "v5_region": "europe"
    },
    "ru": {
        "v4_base": "https://ru.api.riotgames.com",
        "v5_region": "europe"
    },
    "ph2": {
        "v4_base": "https://ph2.api.riotgames.com",
        "v5_region": "sea"
    },
    "sg2": {
        "v4_base": "https://sg2.api.riotgames.com",
        "v5_region": "sea"
    },
    "th2": {
        "v4_base": "https://th2.api.riotgames.com",
        "v5_region": "sea"
    },
    "tw2": {
        "v4_base": "https://tw2.api.riotgames.com",
        "v5_region": "sea"
    },
    "vn2": {
        "v4_base": "https://vn2.api.riotgames.com",
        "v5_region": "sea"
    }
}


class RiotAPI:
    """
    Wrapper für Riot Games API mit automatischem Rate Limiting.
    
    Diese Klasse verwaltet API-Requests und stellt sicher, dass die Rate Limits
    der Riot API eingehalten werden. Bei Fehlern wird automatisch retry durchgeführt.
    """
    
    def __init__(self, api_key: Optional[str] = None, region: str = "euw1"):
        """
        Initialisiert die RiotAPI Instanz.
        
        Args:
            api_key: Optional - wird aus .env oder config.py geladen wenn nicht angegeben
            region: Region Code (z.B. "euw1", "na1", "kr")
        """
        # API Key Priorität: Parameter > .env > config.py
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.getenv("RIOT_API_KEY")
        if not self.api_key and config:
            self.api_key = getattr(config, "RIOT_API_KEY", None)
        if not self.api_key:
            raise ValueError("RIOT_API_KEY muss gesetzt sein (in .env, config/config.py oder als Parameter)")
        
        if region not in REGION_CONFIGS:
            raise ValueError(f"Unbekannte Region: {region}. Verfügbar: {list(REGION_CONFIGS.keys())}")
        
        self.region = region
        self.config = REGION_CONFIGS[region]
        self.base_url = self.config["v4_base"]  # Für V4 API (League, Summoner)
        self.v5_region = self.config["v5_region"]  # Für V5 API (Match)
        
        self.headers = {"X-Riot-Token": self.api_key}
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms = max 10 Requests/Sekunde (Development Key)
    
    def _make_request(self, endpoint: str, max_retries: int = 3) -> Dict:
        """
        Macht einen API Request mit Rate Limiting und automatischer Retry-Logik.
        
        Args:
            endpoint: API Endpoint (z.B. "/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5")
            max_retries: Maximale Anzahl Retry-Versuche bei Fehlern
        
        Returns:
            JSON-Daten als Dictionary
        
        Raises:
            requests.exceptions.RequestException: Bei dauerhaften Fehlern
        """
        # Rate Limiting: Warte zwischen Requests
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        url = f"{self.base_url}{endpoint}"
        
        # Retry-Logik: Bei Netzwerkfehlern mehrfach versuchen
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                self.last_request_time = time.time()
                
                # Rate Limit erreicht: Warte und versuche erneut
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    time.sleep(retry_after)
                    return self._make_request(endpoint, max_retries)
                
                response.raise_for_status()  # Wirft Exception bei HTTP-Fehlern
                return response.json()
            
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout, 
                    requests.exceptions.RequestException):
                # Bei Netzwerkfehlern: Warte länger und versuche erneut
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                    time.sleep(wait_time)
                else:
                    raise  # Nach max_retries: Fehler weiterwerfen
    
    def get_challenger_players(self, queue: str = "RANKED_SOLO_5x5") -> List[Dict]:
        """Holt Challenger Spieler."""
        endpoint = f"/lol/league/v4/challengerleagues/by-queue/{queue}"
        data = self._make_request(endpoint)
        return data.get("entries", [])
    
    def get_grandmaster_players(self, queue: str = "RANKED_SOLO_5x5") -> List[Dict]:
        """Holt Grandmaster Spieler."""
        endpoint = f"/lol/league/v4/grandmasterleagues/by-queue/{queue}"
        data = self._make_request(endpoint)
        return data.get("entries", [])
    
    def get_master_players(self, queue: str = "RANKED_SOLO_5x5") -> List[Dict]:
        """Holt Master Spieler."""
        endpoint = f"/lol/league/v4/masterleagues/by-queue/{queue}"
        data = self._make_request(endpoint)
        return data.get("entries", [])
    
    def get_player_puuid(self, summoner_id: str) -> str:
        """Holt PUUID für einen Spieler."""
        endpoint = f"/lol/summoner/v4/summoners/{summoner_id}"
        data = self._make_request(endpoint)
        return data.get("puuid", "")
    
    def get_match_ids(self, puuid: str, count: int = 100, queue: Optional[int] = None) -> List[str]:
        """
        Holt Match-IDs für einen Spieler.
        
        Args:
            puuid: Player Unique ID
            count: Anzahl der Match-IDs (max 100)
            queue: Optional - Queue ID (420 = Ranked Solo/Duo)
        
        Returns:
            Liste von Match-IDs
        """
        endpoint = f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"count": count}
        if queue:
            params["queue"] = queue
        
        # V5 API verwendet regionale Routing Values (nicht Region Codes)
        url = f"https://{self.v5_region}.api.riotgames.com{endpoint}"
        
        # Rate Limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        response = requests.get(url, headers=self.headers, params=params)
        self.last_request_time = time.time()
        
        # Rate Limit: Warte und versuche erneut
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            time.sleep(retry_after)
            return self.get_match_ids(puuid, count, queue)
        
        response.raise_for_status()
        return response.json()
    
    def get_match_details(self, match_id: str) -> Dict:
        """
        Holt detaillierte Informationen für einen Match.
        
        Args:
            match_id: Match-ID (z.B. "EUW1_1234567890")
        
        Returns:
            Match-Daten als Dictionary (enthält alle Match-Informationen)
        """
        # V5 API verwendet regionale Routing Values
        url = f"https://{self.v5_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        
        # Rate Limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        response = requests.get(url, headers=self.headers)
        self.last_request_time = time.time()
        
        # Rate Limit: Warte und versuche erneut
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            time.sleep(retry_after)
            return self.get_match_details(match_id)
        
        response.raise_for_status()
        return response.json()

