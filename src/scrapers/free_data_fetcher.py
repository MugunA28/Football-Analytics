<<<<<<< HEAD
"""
FREE Data Fetcher - With league-aware fallbacks
"""

import requests
import time
from typing import Dict, Optional, List
import sys
sys.path.insert(0, 'src')

from scrapers.league_estimator import LeagueEstimator

class FreeDataFetcher:
    """Fetches data with smart league-aware fallbacks."""
    
    def __init__(self, debug=True):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        })
        self.cache = {}
        self.debug = debug
        self.league_estimator = LeagueEstimator()
        self.current_league = 'Default'
        
    def set_league(self, league: str):
        """Set current league for better estimates."""
        self.current_league = league
    
    def debug_print(self, msg):
        """Print debug messages."""
        if self.debug:
            print(f"      [DEBUG] {msg}")
    
    def get_team_data(self, team_name: str, is_home: bool = True) -> Dict:
        """Get team data with league-aware fallbacks."""
        if team_name in self.cache:
            return self.cache[team_name]
        
        print(f"   🔍 Searching: {team_name}")
        
        # FotMob is currently not working (404 errors)
        # Using league-aware estimates instead
        
        print(f"      📊 Using {self.current_league} league data")
        estimate = self.league_estimator.get_league_estimate(
            team_name, 
            self.current_league, 
            is_home
        )
        
        self.cache[team_name] = estimate
        return estimate

=======
# free_data_fetcher.py content here... (actual content needs to be read) 
# This file fetches free football data from various sources.


# further implementation here...
>>>>>>> 4cab15094e8a14ad692f4b9d37983ad988eb16a1
