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
        Get player statistics for a specific match, including xG, xA, shots, and key passes.
        
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
                        # Extract basic info
                        player_info = {
                            'team': team_name,
                            'name': player.get('name', {}).get('fullName', 'Unknown'),
                            'player_id': player.get('id'),
                            'position': player.get('role', 'Unknown'),
                            'rating': player.get('rating', {}).get('num'),
                            'goals': player.get('goals', 0),
                            'assists': player.get('assists', 0),
                            'minutes_played': player.get('minutesPlayed', 0),
                        }
                        
                        # Extract advanced stats if available
                        stats = player.get('stats', [])
                        for stat in stats:
                            stat_key = stat.get('key', '')
                            stat_value = stat.get('value')
                            
                            if stat_key == 'expected_goals':
                                player_info['xg'] = stat_value
                            elif stat_key == 'expected_assists':
                                player_info['xa'] = stat_value
                            elif stat_key == 'total_shots':
                                player_info['shots'] = stat_value
                            elif stat_key == 'ontarget_scoring_att':
                                player_info['shots_on_target'] = stat_value
                            elif stat_key == 'big_chance_created':
                                player_info['key_passes'] = stat_value
                        
                        players_data.append(player_info)
            
            logger.info(f"Extracted data for {len(players_data)} players")
            return players_data if players_data else None
        
        except Exception as e:
            logger.error(f"Error extracting player statistics: {e}")
            return None
    
    def get_premier_league_fixtures(self, matchweek: int, season: str = "2025/2026") -> Optional[List[Dict]]:
        """
        Get Premier League fixtures for a specific matchweek.
        
        Args:
            matchweek: Matchweek number
            season: Season (e.g., "2025/2026")
        
        Returns:
            List of fixtures or None if request fails
        """
        logger.info(f"Fetching Premier League fixtures for matchweek {matchweek}")
        
        # Premier League ID in FotMob is 47
        league_data = self.get_league_matches(league_id=47, season=season)
        
        if not league_data:
            return None
        
        try:
            fixtures = []
            
            # Check for fixtures in API response
            if 'fixtures' not in league_data:
                logger.warning("Fixtures key not found in API response")
                return None
            
            fixtures_data = league_data.get('fixtures', {})
            if 'allMatches' in fixtures_data:
                all_matches = fixtures_data['allMatches']
                for match in all_matches:
                    # Handle round as both string and int
                    match_round = match.get('round')
                    if match_round is not None:
                        # Convert to string for comparison
                        if str(match_round) == str(matchweek):
                            fixture = {
                                'match_id': match.get('id'),
                                'home_team': match.get('home', {}).get('name'),
                                'away_team': match.get('away', {}).get('name'),
                                'home_team_id': match.get('home', {}).get('id'),
                                'away_team_id': match.get('away', {}).get('id'),
                                'round': match.get('round'),
                                'status': match.get('status', {}).get('utcTime'),
                                'timestamp': match.get('status', {}).get('utcTime'),
                            }
                            fixtures.append(fixture)
            
            logger.info(f"Found {len(fixtures)} fixtures for matchweek {matchweek}")
            return fixtures if fixtures else None
        
        except Exception as e:
            logger.error(f"Error extracting fixtures: {e}")
            return None
    
    def get_player_season_stats(self, player_id: int, season: str = "2024/2025") -> Optional[Dict]:
        """
        Get player season statistics.
        
        Args:
            player_id: FotMob player ID
            season: Season (e.g., "2024/2025")
        
        Returns:
            Dictionary containing player season stats or None if request fails
        """
        logger.info(f"Fetching season stats for player {player_id}")
        endpoint = f"/playerData"
        params = {'id': player_id}
        
        player_data = self._make_request(endpoint, params)
        
        if not player_data:
            return None
        
        try:
            season_stats = {}
            
            # Extract primary stats
            if 'primaryTeam' in player_data:
                primary_team = player_data['primaryTeam']
                season_stats['team'] = primary_team.get('name')
                season_stats['team_id'] = primary_team.get('id')
            
            # Extract season stats from playerProps
            if 'playerProps' in player_data:
                props = player_data['playerProps']
                season_stats['name'] = props.get('name')
                season_stats['position'] = props.get('position')
                season_stats['age'] = props.get('age')
            
            # Extract statistics from statSeason
            if 'statSeasons' in player_data:
                stat_seasons = player_data['statSeasons']
                for stat_season in stat_seasons:
                    if season in stat_season.get('seasonName', ''):
                        tournaments = stat_season.get('tournaments', [])
                        for tournament in tournaments:
                            if tournament.get('name') == 'Premier League':
                                stats = tournament.get('stats', {})
                                season_stats.update({
                                    'appearances': stats.get('appearances'),
                                    'goals': stats.get('goals'),
                                    'assists': stats.get('assists'),
                                    'minutes_played': stats.get('minutesPlayed'),
                                    'rating': stats.get('rating'),
                                    'expected_goals': stats.get('expectedGoals'),
                                    'expected_assists': stats.get('expectedAssists'),
                                })
            
            return season_stats if season_stats else None
        
        except Exception as e:
            logger.error(f"Error extracting player season stats: {e}")
            return None
    
    def get_player_xg_stats(self, player_id: int, season: str = "2024/2025") -> Optional[Dict]:
        """
        Get player xG statistics for a specific season.
        
        Args:
            player_id: FotMob player ID
            season: Season (e.g., "2024/2025")
        
        Returns:
            Dictionary containing xG-related statistics or None if request fails
        """
        logger.info(f"Fetching xG stats for player {player_id}")
        
        season_stats = self.get_player_season_stats(player_id, season)
        
        if not season_stats:
            return None
        
        try:
            xg_stats = {
                'player_id': player_id,
                'name': season_stats.get('name'),
                'team': season_stats.get('team'),
                'expected_goals': season_stats.get('expected_goals', 0.0),
                'expected_assists': season_stats.get('expected_assists', 0.0),
                'actual_goals': season_stats.get('goals', 0),
                'actual_assists': season_stats.get('assists', 0),
                'minutes_played': season_stats.get('minutes_played', 0),
            }
            
            # Calculate per 90 metrics
            minutes = xg_stats['minutes_played']
            if minutes > 0:
                xg_stats['xg_per_90'] = (xg_stats['expected_goals'] / minutes) * 90
                xg_stats['xa_per_90'] = (xg_stats['expected_assists'] / minutes) * 90
                xg_stats['goals_per_90'] = (xg_stats['actual_goals'] / minutes) * 90
                xg_stats['assists_per_90'] = (xg_stats['actual_assists'] / minutes) * 90
            else:
                xg_stats['xg_per_90'] = 0.0
                xg_stats['xa_per_90'] = 0.0
                xg_stats['goals_per_90'] = 0.0
                xg_stats['assists_per_90'] = 0.0
            
            return xg_stats
        
        except Exception as e:
            logger.error(f"Error calculating xG stats: {e}")
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
