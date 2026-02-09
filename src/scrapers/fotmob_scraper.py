"""
FotMob API Scraper

This module provides functionality to scrape football match data from FotMob's API,
with a focus on Expected Goals (xG) data and detailed match information.
"""

import requests
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class FotMobScraper:
    """
    A scraper class for extracting football data from FotMob API.
    
    Attributes:
        base_url (str): Base URL for FotMob API
        headers (dict): HTTP headers for API requests
        timeout (int): Request timeout in seconds
        retry_attempts (int): Number of retry attempts for failed requests
        retry_delay (int): Delay between retries in seconds
        rate_limit_delay (float): Delay between requests to respect rate limits
    """
    
    def __init__(self, base_url: str = "https://www.fotmob.com/api",
                 timeout: int = 30, retry_attempts: int = 3,
                 retry_delay: int = 2, rate_limit_delay: float = 1.5):
        """
        Initialize the FotMob scraper.
        
        Args:
            base_url: Base URL for the FotMob API
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
            rate_limit_delay: Delay between requests in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.fotmob.com/',
        }
        
        logger.info("FotMob scraper initialized")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make an API request with retry logic and error handling.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters for the request
        
        Returns:
            JSON response as dictionary or None if request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(f"Making request to {url}, attempt {attempt + 1}/{self.retry_attempts}")
                response = requests.get(url, headers=self.headers, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                return response.json()
            
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error occurred: {e}")
                if response.status_code == 429:  # Too many requests
                    wait_time = self.retry_delay * (attempt + 1) * 2
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif response.status_code == 404:
                    logger.error(f"Resource not found: {url}")
                    return None
                elif attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None
            
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None
            
            except requests.exceptions.Timeout as e:
                logger.error(f"Request timeout: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None
        
        return None
    
    def get_match_details(self, match_id: int) -> Optional[Dict]:
        """
        Get detailed match information including xG data and statistics.
        
        Args:
            match_id: FotMob match ID
        
        Returns:
            Dictionary containing match details or None if request fails
        """
        logger.info(f"Fetching details for match {match_id}")
        endpoint = f"/matchDetails"
        params = {'matchId': match_id}
        return self._make_request(endpoint, params)
    
    def get_team_data(self, team_id: int) -> Optional[Dict]:
        """
        Get team data including squad, fixtures, and statistics.
        
        Args:
            team_id: FotMob team ID
        
        Returns:
            Dictionary containing team data or None if request fails
        """
        logger.info(f"Fetching data for team {team_id}")
        endpoint = f"/teams"
        params = {'id': team_id}
        return self._make_request(endpoint, params)
    
    def get_league_matches(self, league_id: int, season: Optional[str] = None) -> Optional[Dict]:
        """
        Get matches for a specific league.
        
        Args:
            league_id: FotMob league ID
            season: Optional season parameter (e.g., "2023/2024")
        
        Returns:
            Dictionary containing league matches or None if request fails
        """
        logger.info(f"Fetching matches for league {league_id}, season {season}")
        endpoint = f"/leagues"
        params = {'id': league_id}
        if season:
            params['season'] = season
        return self._make_request(endpoint, params)
    
    def extract_xg_data(self, match_details: Dict) -> Optional[Dict]:
        """
        Extract Expected Goals (xG) data from match details.
        
        Args:
            match_details: Raw match details dictionary from get_match_details()
        
        Returns:
            Dictionary with xG data for home and away teams or None if not available
        """
        if not match_details:
            logger.warning("No match details provided")
            return None
        
        try:
            xg_data = {}
            
            # Try to extract xG from header
            if 'header' in match_details:
                header = match_details['header']
                teams = header.get('teams', [])
                
                if len(teams) >= 2:
                    home_team = teams[0]
                    away_team = teams[1]
                    
                    xg_data['home_team'] = home_team.get('name', 'Unknown')
                    xg_data['away_team'] = away_team.get('name', 'Unknown')
                    xg_data['home_xg'] = None
                    xg_data['away_xg'] = None
            
            # Try to extract xG from content
            if 'content' in match_details:
                content = match_details['content']
                stats = content.get('stats', {})
                
                # Look for xG in different possible locations
                if 'Expected goals (xG)' in stats:
                    xg_stats = stats['Expected goals (xG)']
                    xg_data['home_xg'] = xg_stats.get('stats', [{}])[0].get('value')
                    xg_data['away_xg'] = xg_stats.get('stats', [{}])[1].get('value') if len(xg_stats.get('stats', [])) > 1 else None
                
                # Alternative location for xG data
                if 'matchFacts' in content and 'xG' in content['matchFacts']:
                    match_facts_xg = content['matchFacts']['xG']
                    xg_data['home_xg'] = match_facts_xg.get('home')
                    xg_data['away_xg'] = match_facts_xg.get('away')
            
            logger.info(f"Extracted xG data: {xg_data}")
            return xg_data if xg_data else None
        
        except Exception as e:
            logger.error(f"Error extracting xG data: {e}")
            return None
    
    def get_player_statistics(self, match_id: int) -> Optional[List[Dict]]:
        """
        Get player statistics for a specific match.
        
        Args:
            match_id: FotMob match ID
        
        Returns:
            List of dictionaries containing player statistics or None if request fails
        """
        logger.info(f"Fetching player statistics for match {match_id}")
        
        match_details = self.get_match_details(match_id)
        if not match_details:
            return None
        
        try:
            players_data = []
            
            if 'content' in match_details and 'lineup' in match_details['content']:
                lineup = match_details['content']['lineup']
                
                for team in lineup:
                    team_name = team.get('teamName', 'Unknown')
                    players = team.get('players', [])
                    
                    for player in players:
                        player_info = {
                            'team': team_name,
                            'name': player.get('name', {}).get('fullName', 'Unknown'),
                            'position': player.get('role', 'Unknown'),
                            'rating': player.get('rating', {}).get('num'),
                            'goals': player.get('goals', 0),
                            'assists': player.get('assists', 0),
                        }
                        players_data.append(player_info)
            
            logger.info(f"Extracted data for {len(players_data)} players")
            return players_data if players_data else None
        
        except Exception as e:
            logger.error(f"Error extracting player statistics: {e}")
            return None
    
    def get_team_form(self, team_id: int, num_matches: int = 5) -> Optional[List[Dict]]:
        """
        Get recent form for a team.
        
        Args:
            team_id: FotMob team ID
            num_matches: Number of recent matches to retrieve
        
        Returns:
            List of recent match results or None if request fails
        """
        logger.info(f"Fetching recent form for team {team_id}")
        
        team_data = self.get_team_data(team_id)
        if not team_data:
            return None
        
        try:
            fixtures = team_data.get('fixtures', {}).get('allFixtures', {}).get('fixtures', [])
            
            recent_matches = []
            for fixture in fixtures[:num_matches]:
                if fixture.get('status', {}).get('finished'):
                    match_info = {
                        'date': fixture.get('status', {}).get('utcTime'),
                        'opponent': fixture.get('opponent', {}).get('name', 'Unknown'),
                        'home': fixture.get('home'),
                        'result': fixture.get('result'),
                        'score': f"{fixture.get('home_score', 0)}-{fixture.get('away_score', 0)}"
                    }
                    recent_matches.append(match_info)
            
            logger.info(f"Retrieved {len(recent_matches)} recent matches")
            return recent_matches if recent_matches else None
        
        except Exception as e:
            logger.error(f"Error extracting team form: {e}")
            return None
    
    def parse_match_to_dataframe(self, match_details: Dict) -> pd.DataFrame:
        """
        Parse match details into a pandas DataFrame.
        
        Args:
            match_details: Raw match details dictionary
        
        Returns:
            DataFrame with parsed match data
        """
        if not match_details:
            logger.warning("No match details to parse")
            return pd.DataFrame()
        
        try:
            match_data = []
            
            if 'header' in match_details:
                header = match_details['header']
                teams = header.get('teams', [])
                
                if len(teams) >= 2:
                    xg_data = self.extract_xg_data(match_details)
                    
                    match_info = {
                        'match_id': match_details.get('general', {}).get('matchId'),
                        'date': header.get('status', {}).get('utcTime'),
                        'home_team': teams[0].get('name'),
                        'away_team': teams[1].get('name'),
                        'home_score': teams[0].get('score'),
                        'away_score': teams[1].get('score'),
                        'home_xg': xg_data.get('home_xg') if xg_data else None,
                        'away_xg': xg_data.get('away_xg') if xg_data else None,
                        'status': header.get('status', {}).get('reason', {}).get('short')
                    }
                    match_data.append(match_info)
            
            df = pd.DataFrame(match_data)
            logger.info(f"Parsed match data into DataFrame with {len(df)} rows")
            return df
        
        except Exception as e:
            logger.error(f"Error parsing match to DataFrame: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    scraper = FotMobScraper()
    
    # Example: Get match details and extract xG
    # match_details = scraper.get_match_details(match_id=4193490)
    # if match_details:
    #     xg_data = scraper.extract_xg_data(match_details)
    #     print(f"xG Data: {xg_data}")
    
    print("FotMob scraper initialized. Use the methods to scrape data.")
