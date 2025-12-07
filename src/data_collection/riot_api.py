"""
Riot Games API Wrapper für Daten-Sammlung.
"""

import requests
import time
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class RiotAPI:
    """Wrapper für Riot Games API mit Rate Limiting."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("RIOT_API_KEY")
        if not self.api_key:
            raise ValueError("RIOT_API_KEY muss gesetzt sein (in .env oder als Parameter)")
        
        self.base_url = "https://euw1.api.riotgames.com"
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
        
        # V5 API verwendet eine andere Base URL
        url = f"https://europe.api.riotgames.com{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            time.sleep(retry_after)
            return self.get_match_ids(puuid, count, queue)
        
        response.raise_for_status()
        return response.json()
    
    def get_match_details(self, match_id: str) -> Dict:
        """Holt Details für einen Match."""
        # V5 API
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            time.sleep(retry_after)
            return self.get_match_details(match_id)
        
        response.raise_for_status()
        return response.json()

