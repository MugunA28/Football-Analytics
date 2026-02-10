"""
SofaScore API Scraper

This module provides functionality to scrape football match data from SofaScore's API.
It includes methods to retrieve match statistics, team data, upcoming fixtures, and head-to-head records.
"""

import requests
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class SofaScoreScraper:
    """
    A scraper class for extracting football data from SofaScore API.
    
    Attributes:
        base_url (str): Base URL for SofaScore API
        headers (dict): HTTP headers for API requests
        timeout (int): Request timeout in seconds
        retry_attempts (int): Number of retry attempts for failed requests
        retry_delay (int): Delay between retries in seconds
        rate_limit_delay (float): Delay between requests to respect rate limits
    """
    
    def __init__(self, base_url: str = "https://api.sofascore.com/api/v1",
                 timeout: int = 30, retry_attempts: int = 3,
                 retry_delay: int = 2, rate_limit_delay: float = 1.5):
        """
        Initialize the SofaScore scraper.
        
        Args:
            base_url: Base URL for the SofaScore API
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
            'Referer': 'https://www.sofascore.com/',
        }
        
        logger.info("SofaScore scraper initialized")
    
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
    
    def get_match_statistics(self, match_id: int) -> Optional[Dict]:
        """
        Get detailed statistics for a specific match.
        
        Args:
            match_id: SofaScore match ID
        
        Returns:
            Dictionary containing match statistics or None if request fails
        """
        logger.info(f"Fetching statistics for match {match_id}")
        endpoint = f"/event/{match_id}/statistics"
        return self._make_request(endpoint)
    
    def get_team_statistics(self, team_id: int, tournament_id: int, season_id: int) -> Optional[Dict]:
        """
        Get team statistics for a specific tournament and season.
        
        Args:
            team_id: SofaScore team ID
            tournament_id: Tournament/league ID
            season_id: Season ID
        
        Returns:
            Dictionary containing team statistics or None if request fails
        """
        logger.info(f"Fetching statistics for team {team_id} in tournament {tournament_id}, season {season_id}")
        endpoint = f"/team/{team_id}/tournament/{tournament_id}/season/{season_id}/statistics/overall"
        return self._make_request(endpoint)
    
    def get_upcoming_matches(self, tournament_id: int, season_id: int) -> Optional[Dict]:
        """
        Get upcoming matches for a specific tournament and season.
        
        Args:
            tournament_id: Tournament/league ID
            season_id: Season ID
        
        Returns:
            Dictionary containing upcoming matches or None if request fails
        """
        logger.info(f"Fetching upcoming matches for tournament {tournament_id}, season {season_id}")
        endpoint = f"/tournament/{tournament_id}/season/{season_id}/events/next/0"
        return self._make_request(endpoint)
    
    def get_h2h_statistics(self, team1_id: int, team2_id: int) -> Optional[Dict]:
        """
        Get head-to-head statistics between two teams.
        
        Args:
            team1_id: First team's SofaScore ID
            team2_id: Second team's SofaScore ID
        
        Returns:
            Dictionary containing head-to-head statistics or None if request fails
        """
        logger.info(f"Fetching H2H statistics between teams {team1_id} and {team2_id}")
        endpoint = f"/team/{team1_id}/head-to-head/{team2_id}/events/0"
        return self._make_request(endpoint)
    
    def get_match_details(self, match_id: int) -> Optional[Dict]:
        """
        Get detailed match information including score and basic stats.
        
        Args:
            match_id: SofaScore match ID
        
        Returns:
            Dictionary containing match details or None if request fails
        """
        logger.info(f"Fetching details for match {match_id}")
        endpoint = f"/event/{match_id}"
        return self._make_request(endpoint)
    
    def parse_statistics_to_dataframe(self, match_statistics: Dict) -> pd.DataFrame:
        """
        Parse match statistics into a pandas DataFrame for easier analysis.
        
        Args:
            match_statistics: Raw statistics dictionary from get_match_statistics()
        
        Returns:
            DataFrame with parsed statistics
        """
        if not match_statistics or 'statistics' not in match_statistics:
            logger.warning("No statistics data to parse")
            return pd.DataFrame()
        
        try:
            # Extract statistics groups
            stats_groups = match_statistics.get('statistics', [])
            
            parsed_stats = []
            for group in stats_groups:
                group_name = group.get('groupName', 'Unknown')
                stats_items = group.get('statisticsItems', [])
                
                for item in stats_items:
                    stat_name = item.get('name', 'Unknown')
                    home_value = item.get('home', 0)
                    away_value = item.get('away', 0)
                    
                    parsed_stats.append({
                        'group': group_name,
                        'statistic': stat_name,
                        'home_value': home_value,
                        'away_value': away_value
                    })
            
            df = pd.DataFrame(parsed_stats)
            logger.info(f"Parsed {len(df)} statistics into DataFrame")
            return df
        
        except Exception as e:
            logger.error(f"Error parsing statistics to DataFrame: {e}")
            return pd.DataFrame()
    
    def get_tournament_matches(self, tournament_id: int, season_id: int, round_num: Optional[int] = None) -> Optional[Dict]:
        """
        Get matches for a specific tournament and season, optionally filtered by round.
        
        Args:
            tournament_id: Tournament/league ID
            season_id: Season ID
            round_num: Optional round number to filter matches
        
        Returns:
            Dictionary containing matches or None if request fails
        """
        logger.info(f"Fetching matches for tournament {tournament_id}, season {season_id}, round {round_num}")
        
        if round_num:
            endpoint = f"/tournament/{tournament_id}/season/{season_id}/events/round/{round_num}"
        else:
            endpoint = f"/tournament/{tournament_id}/season/{season_id}/events/last/0"
        
        return self._make_request(endpoint)
    
    def get_tournament_fixtures(self, tournament_id: int, season_id: int, round_num: int) -> Optional[List[Dict]]:
        """
        Get fixtures for a specific tournament round (Premier League = tournament_id: 17).
        
        Args:
            tournament_id: Tournament/league ID (Premier League = 17)
            season_id: Season ID
            round_num: Round/matchweek number
        
        Returns:
            List of fixture dictionaries or None if request fails
        """
        logger.info(f"Fetching fixtures for tournament {tournament_id}, season {season_id}, round {round_num}")
        
        matches_data = self.get_tournament_matches(tournament_id, season_id, round_num)
        
        if not matches_data:
            return None
        
        try:
            fixtures = []
            events = matches_data.get('events', [])
            
            for event in events:
                fixture = {
                    'match_id': event.get('id'),
                    'home_team': event.get('homeTeam', {}).get('name'),
                    'home_team_id': event.get('homeTeam', {}).get('id'),
                    'away_team': event.get('awayTeam', {}).get('name'),
                    'away_team_id': event.get('awayTeam', {}).get('id'),
                    'start_timestamp': event.get('startTimestamp'),
                    'status': event.get('status', {}).get('type'),
                    'round': event.get('roundInfo', {}).get('round'),
                    'home_score': event.get('homeScore', {}).get('current'),
                    'away_score': event.get('awayScore', {}).get('current'),
                }
                fixtures.append(fixture)
            
            logger.info(f"Found {len(fixtures)} fixtures for round {round_num}")
            return fixtures if fixtures else None
        
        except Exception as e:
            logger.error(f"Error extracting fixtures: {e}")
            return None
    
    def get_player_stats(self, player_id: int, tournament_id: int, season_id: int) -> Optional[Dict]:
        """
        Get player statistics for a specific tournament and season.
        
        Args:
            player_id: SofaScore player ID
            tournament_id: Tournament/league ID
            season_id: Season ID
        
        Returns:
            Dictionary containing player statistics or None if request fails
        """
        logger.info(f"Fetching stats for player {player_id} in tournament {tournament_id}, season {season_id}")
        endpoint = f"/player/{player_id}/statistics/season/{season_id}"
        
        player_data = self._make_request(endpoint)
        
        if not player_data:
            return None
        
        try:
            player_stats = {}
            
            # Extract player info
            if 'player' in player_data:
                player = player_data['player']
                player_stats['name'] = player.get('name')
                player_stats['position'] = player.get('position')
                player_stats['team_name'] = player.get('team', {}).get('name')
            
            # Extract statistics
            if 'statistics' in player_data:
                stats = player_data['statistics']
                
                # Find relevant tournament stats
                for stat_group in stats:
                    if stat_group.get('tournament', {}).get('id') == tournament_id:
                        stat_dict = stat_group.get('statistics', {})
                        
                        player_stats.update({
                            'appearances': stat_dict.get('appearances'),
                            'goals': stat_dict.get('goals'),
                            'assists': stat_dict.get('assists'),
                            'minutes_played': stat_dict.get('minutesPlayed'),
                            'rating': stat_dict.get('rating'),
                            'expected_goals': stat_dict.get('expectedGoals'),
                            'expected_assists': stat_dict.get('expectedAssists'),
                            'shots_total': stat_dict.get('shotsTotal'),
                            'shots_on_target': stat_dict.get('shotsOnTarget'),
                            'key_passes': stat_dict.get('keyPasses'),
                            'big_chances_created': stat_dict.get('bigChancesCreated'),
                            'big_chances_missed': stat_dict.get('bigChancesMissed'),
                            # Defensive stats
                            'clean_sheets': stat_dict.get('cleanSheets'),
                            'tackles': stat_dict.get('tackles'),
                            'interceptions': stat_dict.get('interceptions'),
                            'clearances': stat_dict.get('clearances'),
                            'goals_conceded': stat_dict.get('goalsConceded'),
                        })
            
            return player_stats if player_stats else None
        
        except Exception as e:
            logger.error(f"Error extracting player stats: {e}")
            return None
    
    def get_team_defensive_stats(self, team_id: int, tournament_id: int, season_id: int) -> Optional[Dict]:
        """
        Get defensive statistics for a team.
        
        Args:
            team_id: SofaScore team ID
            tournament_id: Tournament/league ID
            season_id: Season ID
        
        Returns:
            Dictionary containing defensive statistics or None if request fails
        """
        logger.info(f"Fetching defensive stats for team {team_id}")
        
        team_stats = self.get_team_statistics(team_id, tournament_id, season_id)
        
        if not team_stats:
            return None
        
        try:
            defensive_stats = {}
            
            if 'statistics' in team_stats:
                stats = team_stats['statistics']
                
                defensive_stats = {
                    'team_id': team_id,
                    'clean_sheets': stats.get('cleanSheets'),
                    'goals_conceded': stats.get('goalsConceded'),
                    'xg_conceded': stats.get('expectedGoalsAgainst'),
                    'shots_conceded': stats.get('shotsAgainst'),
                    'tackles': stats.get('tackles'),
                    'interceptions': stats.get('interceptions'),
                    'matches_played': stats.get('matchesPlayed'),
                }
                
                # Calculate per match averages
                matches = defensive_stats.get('matches_played', 1)
                if matches > 0:
                    defensive_stats['clean_sheets_per_match'] = defensive_stats.get('clean_sheets', 0) / matches
                    defensive_stats['goals_conceded_per_match'] = defensive_stats.get('goals_conceded', 0) / matches
                    defensive_stats['xg_conceded_per_match'] = defensive_stats.get('xg_conceded', 0) / matches
            
            return defensive_stats if defensive_stats else None
        
        except Exception as e:
            logger.error(f"Error extracting defensive stats: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    scraper = SofaScoreScraper()
    
    # Example: Get match statistics
    # match_stats = scraper.get_match_statistics(match_id=10612128)
    # if match_stats:
    #     df = scraper.parse_statistics_to_dataframe(match_stats)
    #     print(df)
    
    print("SofaScore scraper initialized. Use the methods to scrape data.")
