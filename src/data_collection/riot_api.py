"""
Riot Games API Wrapper für Daten-Sammlung.
"""

import requests
import time
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

# Region Mapping für Riot API
# V4 API (League, Summoner) verwendet region codes
# V5 API (Match) verwendet regional routing values
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
    """Wrapper für Riot Games API mit Rate Limiting."""
    
    def __init__(self, api_key: Optional[str] = None, region: str = "euw1"):
        self.api_key = api_key or os.getenv("RIOT_API_KEY")
        if not self.api_key:
            raise ValueError("RIOT_API_KEY muss gesetzt sein (in .env oder als Parameter)")
        
        if region not in REGION_CONFIGS:
            raise ValueError(f"Unbekannte Region: {region}. Verfügbar: {list(REGION_CONFIGS.keys())}")
        
        self.region = region
        self.config = REGION_CONFIGS[region]
        self.base_url = self.config["v4_base"]
        self.v5_region = self.config["v5_region"]
        
        self.headers = {
            "X-Riot-Token": self.api_key
        }
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms zwischen Requests (10 req/s für Development Key)
    
    def _make_request(self, endpoint: str) -> Dict:
        """Macht API Request mit Rate Limiting."""
        # Rate Limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers)
        self.last_request_time = time.time()
        
        if response.status_code == 429:  # Rate Limit exceeded
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"Rate Limit erreicht. Warte {retry_after} Sekunden...")
            time.sleep(retry_after)
            return self._make_request(endpoint)
        
        response.raise_for_status()
        return response.json()
    
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
        """Holt Match-IDs für einen Spieler."""
        endpoint = f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"count": count}
        if queue:
            params["queue"] = queue
        
        # V5 API verwendet regionale Routing Values
        url = f"https://{self.v5_region}.api.riotgames.com{endpoint}"
        
        # Rate Limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        response = requests.get(url, headers=self.headers, params=params)
        self.last_request_time = time.time()
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"Rate Limit erreicht. Warte {retry_after} Sekunden...")
            time.sleep(retry_after)
            return self.get_match_ids(puuid, count, queue)
        
        response.raise_for_status()
        return response.json()
    
    def get_match_details(self, match_id: str) -> Dict:
        """Holt Details für einen Match."""
        # V5 API verwendet regionale Routing Values
        url = f"https://{self.v5_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        
        # Rate Limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        response = requests.get(url, headers=self.headers)
        self.last_request_time = time.time()
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"Rate Limit erreicht. Warte {retry_after} Sekunden...")
            time.sleep(retry_after)
            return self.get_match_details(match_id)
        
        response.raise_for_status()
        return response.json()

