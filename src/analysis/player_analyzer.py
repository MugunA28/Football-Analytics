"""
Player Analyzer Module

This module provides comprehensive player analysis functionality for English Premier League,
analyzing player statistics based on Expected Goals (xG) data and generating probability rankings.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from ..scrapers.fotmob_scraper import FotMobScraper
from ..scrapers.sofascore_scraper import SofaScoreScraper
from ..utils.player_stats_processor import (
    calculate_rolling_average,
    normalize_probability,
    apply_home_advantage_factor,
    adjust_for_opponent_strength,
    calculate_form_score,
    calculate_weighted_probability,
    calculate_minutes_adjustment,
    scale_to_percentage,
    get_form_stars
)

logger = logging.getLogger(__name__)


class PlayerAnalyzer:
    """
    A comprehensive player analysis system for Premier League matches.
    
    Analyzes player statistics based on xG data and generates probability rankings for:
    - Players most likely to score goals
    - Players most likely to provide assists
    - Defenders/Goalkeepers most likely to keep clean sheets
    """
    
    def __init__(self, fotmob_scraper: Optional[FotMobScraper] = None,
                 sofascore_scraper: Optional[SofaScoreScraper] = None,
                 config: Optional[Dict] = None):
        """
        Initialize the PlayerAnalyzer with scrapers and configuration.
        
        Args:
            fotmob_scraper: FotMob scraper instance (creates new if None)
            sofascore_scraper: SofaScore scraper instance (creates new if None)
            config: Configuration dictionary with analysis parameters
        """
        self.fotmob = fotmob_scraper or FotMobScraper()
        self.sofascore = sofascore_scraper or SofaScoreScraper()
        
        # Default configuration
        self.config = {
            'premier_league': {
                'fotmob_league_id': 47,
                'sofascore_tournament_id': 17,
                'sofascore_season_id': 52760,  # 2024/2025 season
                'season': '2024/2025'
            },
            'weights': {
                'goal_probability': {
                    'xg_weight': 0.5,
                    'recent_goals_weight': 0.3,
                    'shot_conversion_weight': 0.2
                },
                'assist_probability': {
                    'xa_weight': 0.5,
                    'recent_assists_weight': 0.3,
                    'key_passes_weight': 0.2
                },
                'clean_sheet_probability': {
                    'defensive_xg_weight': 0.4,
                    'recent_clean_sheets_weight': 0.3,
                    'opponent_strength_weight': 0.3
                }
            },
            'rolling_window': 5,
            'min_minutes_played': 180,
            'home_advantage_boost': 0.1,
            'league_avg_rating': 70.0
        }
        
        # Update with provided config
        if config:
            self._update_config(config)
        
        logger.info("PlayerAnalyzer initialized")
    
    def _update_config(self, config: Dict):
        """Recursively update configuration."""
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = update_dict(self.config, config)
    
    def get_premier_league_fixtures(self, matchweek: int) -> List[Dict]:
        """
        Fetch Premier League fixtures for a specific matchweek.
        
        Args:
            matchweek: Matchweek number
        
        Returns:
            List of fixture dictionaries
        """
        logger.info(f"Fetching Premier League fixtures for matchweek {matchweek}")
        
        # Try FotMob first
        season = self.config['premier_league']['season']
        fixtures = self.fotmob.get_premier_league_fixtures(matchweek, season)
        
        if fixtures:
            logger.info(f"Retrieved {len(fixtures)} fixtures from FotMob")
            return fixtures
        
        # Fallback to SofaScore
        logger.info("FotMob failed, trying SofaScore...")
        tournament_id = self.config['premier_league']['sofascore_tournament_id']
        season_id = self.config['premier_league']['sofascore_season_id']
        
        fixtures = self.sofascore.get_tournament_fixtures(tournament_id, season_id, matchweek)
        
        if fixtures:
            logger.info(f"Retrieved {len(fixtures)} fixtures from SofaScore")
            return fixtures
        
        logger.warning(f"No fixtures found for matchweek {matchweek}")
        return []
    
    def get_player_xg_data(self, match_id: int, source: str = 'fotmob') -> List[Dict]:
        """
        Extract player-level xG data from a match.
        
        Args:
            match_id: Match ID
            source: Data source ('fotmob' or 'sofascore')
        
        Returns:
            List of player statistics with xG data
        """
        logger.info(f"Fetching player xG data for match {match_id} from {source}")
        
        if source == 'fotmob':
            players = self.fotmob.get_player_statistics(match_id)
        else:
            # SofaScore doesn't have direct match-level player xG in the same format
            # Would need to fetch individual player stats
            logger.warning("SofaScore player xG data requires individual player IDs")
            players = []
        
        return players or []
    
    def calculate_goal_probability(self, player_stats: Dict) -> float:
        """
        Calculate scoring probability based on player statistics.
        
        Args:
            player_stats: Dictionary with player statistics including:
                - xg_per_90: Expected goals per 90 minutes
                - recent_goals: Goals in last N matches
                - shots_total: Total shots
                - shots_on_target: Shots on target
                - minutes_played: Total minutes played
                - is_home: Whether playing at home
                - opponent_rating: Opponent's defensive rating
        
        Returns:
            Probability value (0-1)
        """
        weights = self.config['weights']['goal_probability']
        min_minutes = self.config['min_minutes_played']
        
        # Extract values with defaults
        xg_per_90 = player_stats.get('xg_per_90', 0.0)
        recent_goals = player_stats.get('recent_goals', [])
        shots_total = player_stats.get('shots_total', 0)
        shots_on_target = player_stats.get('shots_on_target', 0)
        minutes_played = player_stats.get('minutes_played', 0)
        is_home = player_stats.get('is_home', False)
        opponent_rating = player_stats.get('opponent_rating', 70.0)
        
        # Calculate individual probability components
        
        # 1. xG-based probability (normalize based on typical range 0-1.5 per 90)
        xg_prob = normalize_probability(xg_per_90, 0.0, 1.5)
        
        # 2. Recent form probability
        if isinstance(recent_goals, list) and len(recent_goals) > 0:
            recent_goals_avg = calculate_rolling_average(recent_goals, window=self.config['rolling_window'])
            form_prob = normalize_probability(recent_goals_avg, 0.0, 2.0)  # 0-2 goals per match range
        else:
            form_prob = 0.0
        
        # 3. Shot conversion probability
        if shots_total > 0:
            conversion_rate = shots_on_target / shots_total
            conversion_prob = conversion_rate
        else:
            conversion_prob = 0.0
        
        # Calculate weighted probability
        metrics = {
            'xg': xg_prob,
            'form': form_prob,
            'conversion': conversion_prob
        }
        
        base_probability = calculate_weighted_probability(metrics, weights)
        
        # Apply adjustments
        
        # Minutes adjustment - reduce probability for players with few minutes
        minutes_factor = calculate_minutes_adjustment(
            minutes_played, 
            total_minutes=450,  # 5 matches * 90 minutes
            min_threshold=min_minutes
        )
        adjusted_probability = base_probability * minutes_factor
        
        # Home advantage
        adjusted_probability = apply_home_advantage_factor(
            adjusted_probability, 
            is_home, 
            home_boost=self.config['home_advantage_boost']
        )
        
        # Opponent strength
        adjusted_probability = adjust_for_opponent_strength(
            adjusted_probability,
            opponent_rating,
            league_avg_rating=self.config['league_avg_rating']
        )
        
        return adjusted_probability
    
    def calculate_assist_probability(self, player_stats: Dict) -> float:
        """
        Calculate assist probability based on player statistics.
        
        Args:
            player_stats: Dictionary with player statistics including:
                - xa_per_90: Expected assists per 90 minutes
                - recent_assists: Assists in last N matches
                - key_passes: Key passes per match
                - minutes_played: Total minutes played
                - is_home: Whether playing at home
                - opponent_rating: Opponent's defensive rating
        
        Returns:
            Probability value (0-1)
        """
        weights = self.config['weights']['assist_probability']
        min_minutes = self.config['min_minutes_played']
        
        # Extract values
        xa_per_90 = player_stats.get('xa_per_90', 0.0)
        recent_assists = player_stats.get('recent_assists', [])
        key_passes = player_stats.get('key_passes', 0)
        minutes_played = player_stats.get('minutes_played', 0)
        is_home = player_stats.get('is_home', False)
        opponent_rating = player_stats.get('opponent_rating', 70.0)
        
        # Calculate individual probability components
        
        # 1. xA-based probability (normalize based on typical range 0-1.0 per 90)
        xa_prob = normalize_probability(xa_per_90, 0.0, 1.0)
        
        # 2. Recent form probability
        if isinstance(recent_assists, list) and len(recent_assists) > 0:
            recent_assists_avg = calculate_rolling_average(recent_assists, window=self.config['rolling_window'])
            form_prob = normalize_probability(recent_assists_avg, 0.0, 1.5)  # 0-1.5 assists per match range
        else:
            form_prob = 0.0
        
        # 3. Key passes probability (normalize based on typical range 0-5 per match)
        key_passes_prob = normalize_probability(key_passes, 0.0, 5.0)
        
        # Calculate weighted probability
        metrics = {
            'xa': xa_prob,
            'form': form_prob,
            'key_passes': key_passes_prob
        }
        
        base_probability = calculate_weighted_probability(metrics, weights)
        
        # Apply adjustments
        minutes_factor = calculate_minutes_adjustment(minutes_played, 450, min_minutes)
        adjusted_probability = base_probability * minutes_factor
        
        adjusted_probability = apply_home_advantage_factor(adjusted_probability, is_home)
        adjusted_probability = adjust_for_opponent_strength(adjusted_probability, opponent_rating)
        
        return adjusted_probability
    
    def calculate_clean_sheet_probability(self, team_stats: Dict) -> float:
        """
        Calculate clean sheet probability for a team.
        
        Args:
            team_stats: Dictionary with team statistics including:
                - xg_conceded_per_90: Expected goals conceded per 90
                - recent_clean_sheets: Clean sheets in last N matches
                - goals_conceded_recent: Goals conceded in recent matches
                - opponent_xg_per_90: Opponent's expected goals per 90
                - is_home: Whether playing at home
        
        Returns:
            Probability value (0-1)
        """
        weights = self.config['weights']['clean_sheet_probability']
        
        # Extract values
        xg_conceded = team_stats.get('xg_conceded_per_90', 1.0)
        recent_clean_sheets = team_stats.get('recent_clean_sheets', 0)
        recent_matches = team_stats.get('recent_matches', 5)
        opponent_xg = team_stats.get('opponent_xg_per_90', 1.5)
        is_home = team_stats.get('is_home', False)
        
        # Calculate individual probability components
        
        # 1. Defensive xG probability (lower xG conceded = higher probability)
        # Invert the probability since lower is better
        defensive_prob = 1.0 - normalize_probability(xg_conceded, 0.0, 2.5)
        
        # 2. Recent clean sheets probability
        if recent_matches > 0:
            clean_sheet_rate = recent_clean_sheets / recent_matches
            form_prob = clean_sheet_rate
        else:
            form_prob = 0.0
        
        # 3. Opponent strength (lower opponent xG = easier to keep clean sheet)
        opponent_prob = 1.0 - normalize_probability(opponent_xg, 0.0, 2.5)
        
        # Calculate weighted probability
        metrics = {
            'defensive_xg': defensive_prob,
            'form': form_prob,
            'opponent': opponent_prob
        }
        
        base_probability = calculate_weighted_probability(metrics, weights)
        
        # Apply home advantage
        adjusted_probability = apply_home_advantage_factor(base_probability, is_home, home_boost=0.15)
        
        return adjusted_probability
    
    def rank_players_by_metric(self, players: List[Dict], metric: str, 
                               top_n: Optional[int] = None) -> List[Dict]:
        """
        Rank players by a specific metric.
        
        Args:
            players: List of player dictionaries with calculated probabilities
            metric: Metric to rank by ('goal_probability', 'assist_probability', etc.)
            top_n: Number of top players to return (None for all)
        
        Returns:
            Sorted list of players
        """
        if not players:
            return []
        
        # Sort by metric in descending order
        sorted_players = sorted(
            players,
            key=lambda x: x.get(metric, 0.0),
            reverse=True
        )
        
        # Add rank
        for i, player in enumerate(sorted_players, 1):
            player['rank'] = i
        
        # Return top N if specified
        if top_n:
            return sorted_players[:top_n]
        
        return sorted_players
    
    def generate_analysis_report(self, matchweek: int, top_n: int = 20) -> Dict:
        """
        Generate comprehensive analysis report for a matchweek.
        
        Args:
            matchweek: Matchweek number
            top_n: Number of top players to include in each category
        
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Generating analysis report for matchweek {matchweek}")
        
        # Fetch fixtures
        fixtures = self.get_premier_league_fixtures(matchweek)
        
        if not fixtures:
            logger.error(f"No fixtures found for matchweek {matchweek}")
            return {
                'matchweek': matchweek,
                'fixtures': [],
                'top_goal_scorers': [],
                'top_assist_providers': [],
                'clean_sheet_candidates': [],
                'error': 'No fixtures found'
            }
        
        # For now, return the structure (full implementation would fetch all player data)
        report = {
            'matchweek': matchweek,
            'generated_at': datetime.now().isoformat(),
            'fixtures': fixtures,
            'top_goal_scorers': [],
            'top_assist_providers': [],
            'clean_sheet_candidates': [],
            'config': self.config
        }
        
        logger.info(f"Analysis report generated with {len(fixtures)} fixtures")
        
        return report
    
    def format_console_output(self, report: Dict, output_top_n: int = 20) -> str:
        """
        Format analysis report for console output.
        
        Args:
            report: Analysis report dictionary
            output_top_n: Number of top players to display
        
        Returns:
            Formatted string for console output
        """
        output = []
        output.append("=" * 80)
        output.append(f"PREMIER LEAGUE MATCHWEEK {report['matchweek']} ANALYSIS")
        output.append("=" * 80)
        output.append("")
        
        # Fixtures
        output.append(f"FIXTURES ({len(report['fixtures'])} matches):")
        output.append("-" * 80)
        for fixture in report['fixtures']:
            home = fixture.get('home_team', 'TBD')
            away = fixture.get('away_team', 'TBD')
            output.append(f"  {home} vs {away}")
        output.append("")
        
        # Top Goal Scorers
        output.append(f"TOP {output_top_n} PLAYERS - GOAL SCORING PROBABILITY:")
        output.append("-" * 80)
        output.append(f"{'Rank':<6}{'Player':<25}{'Team':<18}{'Opponent':<18}{'Prob':<8}{'xG/90':<8}{'Form':<6}")
        output.append("-" * 80)
        
        for player in report['top_goal_scorers'][:output_top_n]:
            rank = player.get('rank', '-')
            name = player.get('name', 'Unknown')[:24]
            team = player.get('team', 'Unknown')[:17]
            opponent = player.get('opponent', 'Unknown')[:17]
            prob = scale_to_percentage(player.get('goal_probability', 0.0))
            xg = player.get('xg_per_90', 0.0)
            form_score = player.get('form_score', 0.0)
            form_stars = get_form_stars(form_score)
            
            output.append(f"{rank:<6}{name:<25}{team:<18}{opponent:<18}{prob:>6.1f}%  {xg:>6.2f}  {form_stars}")
        
        output.append("")
        
        # Top Assist Providers
        output.append(f"TOP {output_top_n} PLAYERS - ASSIST PROBABILITY:")
        output.append("-" * 80)
        output.append(f"{'Rank':<6}{'Player':<25}{'Team':<18}{'Opponent':<18}{'Prob':<8}{'xA/90':<8}{'KP':<6}")
        output.append("-" * 80)
        
        for player in report['top_assist_providers'][:output_top_n]:
            rank = player.get('rank', '-')
            name = player.get('name', 'Unknown')[:24]
            team = player.get('team', 'Unknown')[:17]
            opponent = player.get('opponent', 'Unknown')[:17]
            prob = scale_to_percentage(player.get('assist_probability', 0.0))
            xa = player.get('xa_per_90', 0.0)
            kp = player.get('key_passes', 0.0)
            
            output.append(f"{rank:<6}{name:<25}{team:<18}{opponent:<18}{prob:>6.1f}%  {xa:>6.2f}  {kp:>4.1f}")
        
        output.append("")
        
        # Clean Sheet Candidates
        output.append(f"TOP 10 TEAMS - CLEAN SHEET PROBABILITY:")
        output.append("-" * 80)
        output.append(f"{'Rank':<6}{'Team':<25}{'Goalkeeper':<20}{'Opponent':<18}{'Prob':<8}{'CS(L5)'}")
        output.append("-" * 80)
        
        for team in report['clean_sheet_candidates'][:10]:
            rank = team.get('rank', '-')
            team_name = team.get('team', 'Unknown')[:24]
            goalkeeper = team.get('goalkeeper', 'Unknown')[:19]
            opponent = team.get('opponent', 'Unknown')[:17]
            prob = scale_to_percentage(team.get('clean_sheet_probability', 0.0))
            cs = team.get('recent_clean_sheets', 0)
            
            output.append(f"{rank:<6}{team_name:<25}{goalkeeper:<20}{opponent:<18}{prob:>6.1f}%  {cs}")
        
        output.append("")
        output.append("=" * 80)
        
        return "\n".join(output)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    analyzer = PlayerAnalyzer()
    
    # Test getting fixtures
    fixtures = analyzer.get_premier_league_fixtures(matchweek=26)
    print(f"Found {len(fixtures)} fixtures")
    
    # Test probability calculations
    test_player = {
        'xg_per_90': 0.8,
        'recent_goals': [2, 1, 0, 1, 2],
        'shots_total': 15,
        'shots_on_target': 8,
        'minutes_played': 450,
        'is_home': True,
        'opponent_rating': 65.0
    }
    
    goal_prob = analyzer.calculate_goal_probability(test_player)
    print(f"Goal probability: {scale_to_percentage(goal_prob):.1f}%")
