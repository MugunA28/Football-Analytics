"""
FREE Working APIs - No payment required
Uses: football-data.org (free tier) and api-football.com (free 100/day)
"""

import requests
import time
from typing import Dict, Optional
from datetime import datetime, timedelta

class FreeAPIs:
    """Working free football APIs."""
    
    def __init__(self):
        self.session = requests.Session()
        # API-Football free tier (100 requests/day)
        # Get free key at: https://www.api-football.com/
        self.api_football_key = None  # User can add their own
        
    def search_team_api_football(self, team_name: str) -> Optional[Dict]:
        """
        API-Football FREE tier: 100 requests/day
        Get key at: https://dashboard.api-football.com/register
        """
        if not self.api_football_key:
            return None
        
        try:
            headers = {
                'x-rapidapi-host': 'v3.football.api-sports.io',
                'x-rapidapi-key': self.api_football_key
            }
            
            # Search for team
            url = f"https://v3.football.api-sports.io/teams?search={team_name}"
            response = self.session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results', 0) > 0:
                    team = data['response'][0]['team']
                    team_id = team['id']
                    
                    # Get team statistics
                    return self.get_api_football_stats(team_id, headers)
            
            return None
        except:
            return None
    
    def get_api_football_stats(self, team_id: int, headers: dict) -> Optional[Dict]:
        """Get team stats from API-Football."""
        try:
            # Get current season stats
            year = datetime.now().year
            url = f"https://v3.football.api-sports.io/teams/statistics?team={team_id}&season={year}"
            
            response = self.session.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('results', 0) > 0:
                    stats_data = data['response']
                    
                    # Extract useful stats
                    goals = stats_data.get('goals', {})
                    matches = stats_data.get('fixtures', {})
                    
                    return {
                        'matches_played': matches.get('played', {}).get('total', 0),
                        'avg_goals_scored': goals.get('for', {}).get('average', {}).get('total', 0),
                        'avg_goals_conceded': goals.get('against', {}).get('average', {}).get('total', 0),
                        'home_avg_scored': goals.get('for', {}).get('average', {}).get('home', 0),
                        'away_avg_scored': goals.get('for', {}).get('average', {}).get('away', 0),
                        'source': 'api_football'
                    }
        except:
            pass
        
        return None

