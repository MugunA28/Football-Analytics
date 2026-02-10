"""
Unit tests for PlayerAnalyzer and player statistics utilities.

Tests for player analysis functionality including probability calculations,
utility functions, and ranking algorithms.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.analysis.player_analyzer import PlayerAnalyzer
from src.utils.player_stats_processor import (
    calculate_rolling_average,
    normalize_probability,
    apply_home_advantage_factor,
    adjust_for_opponent_strength,
    calculate_form_score,
    calculate_weighted_probability,
    calculate_minutes_adjustment,
    exponential_moving_average,
    scale_to_percentage,
    get_form_stars
)


class TestPlayerStatsProcessor:
    """Tests for player statistics utility functions."""
    
    def test_calculate_rolling_average(self):
        """Test rolling average calculation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Window of 3 should use last 3 values: 3, 4, 5
        avg = calculate_rolling_average(data, window=3)
        assert avg == 4.0
        
        # Window larger than data should use all data
        avg = calculate_rolling_average(data, window=10)
        assert avg == 3.0
        
        # Empty data should return 0.0
        avg = calculate_rolling_average([], window=5)
        assert avg == 0.0
    
    def test_normalize_probability(self):
        """Test probability normalization."""
        # Value in middle of range
        prob = normalize_probability(5.0, 0.0, 10.0)
        assert prob == 0.5
        
        # Value at min
        prob = normalize_probability(0.0, 0.0, 10.0)
        assert prob == 0.0
        
        # Value at max
        prob = normalize_probability(10.0, 0.0, 10.0)
        assert prob == 1.0
        
        # Value outside range (should clamp)
        prob = normalize_probability(15.0, 0.0, 10.0)
        assert prob == 1.0
        
        # Value below range (should clamp)
        prob = normalize_probability(-5.0, 0.0, 10.0)
        assert prob == 0.0
    
    def test_apply_home_advantage_factor(self):
        """Test home advantage adjustment."""
        # Home team should get boost
        prob = apply_home_advantage_factor(0.5, is_home=True, home_boost=0.1)
        assert prob == 0.55
        
        # Away team should get reduction
        prob = apply_home_advantage_factor(0.5, is_home=False, home_boost=0.1)
        assert prob == 0.475
        
        # Should clamp at 1.0
        prob = apply_home_advantage_factor(0.95, is_home=True, home_boost=0.1)
        assert prob == 1.0
    
    def test_adjust_for_opponent_strength(self):
        """Test opponent strength adjustment."""
        # Weak opponent (rating 50, avg 70) should increase probability
        prob = adjust_for_opponent_strength(0.5, opponent_rating=50.0, league_avg_rating=70.0)
        assert prob > 0.5
        
        # Strong opponent (rating 90, avg 70) should decrease probability
        prob = adjust_for_opponent_strength(0.5, opponent_rating=90.0, league_avg_rating=70.0)
        assert prob < 0.5
        
        # Average opponent should have minimal change
        prob = adjust_for_opponent_strength(0.5, opponent_rating=70.0, league_avg_rating=70.0)
        assert abs(prob - 0.5) < 0.01
    
    def test_calculate_form_score(self):
        """Test form score calculation."""
        # Good form
        matches = [
            {'goals': 2},
            {'goals': 1},
            {'goals': 2},
            {'goals': 1},
            {'goals': 1}
        ]
        form = calculate_form_score(matches, metric='goals', weight_recent=True)
        assert form > 4.0  # Should have decent form score
        
        # Poor form
        matches = [
            {'goals': 0},
            {'goals': 0},
            {'goals': 0},
            {'goals': 0},
            {'goals': 0}
        ]
        form = calculate_form_score(matches, metric='goals', weight_recent=False)
        assert form == 0.0
        
        # Empty matches
        form = calculate_form_score([], metric='goals')
        assert form == 0.0
    
    def test_calculate_weighted_probability(self):
        """Test weighted probability calculation."""
        metrics = {
            'xg': 0.8,
            'form': 0.6,
            'conversion': 0.7
        }
        weights = {
            'xg': 0.5,
            'form': 0.3,
            'conversion': 0.2
        }
        
        prob = calculate_weighted_probability(metrics, weights)
        expected = 0.8 * 0.5 + 0.6 * 0.3 + 0.7 * 0.2
        assert abs(prob - expected) < 0.001
        
        # Should normalize weights if they don't sum to 1
        weights_unnormalized = {'xg': 1.0, 'form': 1.0, 'conversion': 1.0}
        prob = calculate_weighted_probability(metrics, weights_unnormalized)
        assert 0.0 <= prob <= 1.0
    
    def test_calculate_minutes_adjustment(self):
        """Test minutes adjustment factor."""
        # Full minutes should return 1.0
        factor = calculate_minutes_adjustment(450, total_minutes=450, min_threshold=180)
        assert factor == 1.0
        
        # Minutes above threshold but below total
        factor = calculate_minutes_adjustment(300, total_minutes=450, min_threshold=180)
        assert 0.5 < factor < 1.0
        
        # Minutes below threshold should get reduced credit
        factor = calculate_minutes_adjustment(90, total_minutes=450, min_threshold=180)
        assert factor == 0.25  # 90/180 * 0.5
        
        # Zero minutes should return 0.0
        factor = calculate_minutes_adjustment(0, total_minutes=450, min_threshold=180)
        assert factor == 0.0
    
    def test_exponential_moving_average(self):
        """Test exponential moving average calculation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        ema = exponential_moving_average(data, alpha=0.3)
        assert ema > 3.0  # Should be weighted towards recent values
        
        # Empty data should return 0.0
        ema = exponential_moving_average([], alpha=0.3)
        assert ema == 0.0
        
        # Single value should return that value
        ema = exponential_moving_average([5.0], alpha=0.3)
        assert ema == 5.0
    
    def test_scale_to_percentage(self):
        """Test scaling probability to percentage."""
        assert scale_to_percentage(0.5) == 50.0
        assert scale_to_percentage(0.754) == 75.4
        assert scale_to_percentage(1.0) == 100.0
        assert scale_to_percentage(0.0) == 0.0
        
        # Should clamp values
        assert scale_to_percentage(1.5) == 100.0
        assert scale_to_percentage(-0.5) == 0.0
    
    def test_get_form_stars(self):
        """Test form stars conversion."""
        assert get_form_stars(10.0) == '⭐⭐⭐⭐⭐'
        assert get_form_stars(8.0) == '⭐⭐⭐⭐'
        assert get_form_stars(5.0) == '⭐⭐⭐'
        assert get_form_stars(2.0) == '⭐'
        assert get_form_stars(0.0) == ''


class TestPlayerAnalyzer:
    """Tests for PlayerAnalyzer class."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = PlayerAnalyzer()
        
        assert analyzer.fotmob is not None
        assert analyzer.sofascore is not None
        assert 'premier_league' in analyzer.config
        assert 'weights' in analyzer.config
    
    def test_initialization_with_config(self):
        """Test analyzer initialization with custom config."""
        config = {
            'rolling_window': 10,
            'min_minutes_played': 270
        }
        
        analyzer = PlayerAnalyzer(config=config)
        
        assert analyzer.config['rolling_window'] == 10
        assert analyzer.config['min_minutes_played'] == 270
    
    def test_calculate_goal_probability(self):
        """Test goal probability calculation."""
        analyzer = PlayerAnalyzer()
        
        # Test with good stats
        player_stats = {
            'xg_per_90': 0.8,
            'recent_goals': [2, 1, 1, 2, 1],
            'shots_total': 20,
            'shots_on_target': 12,
            'minutes_played': 450,
            'is_home': True,
            'opponent_rating': 65.0
        }
        
        prob = analyzer.calculate_goal_probability(player_stats)
        
        # Should return a reasonable probability
        assert 0.0 <= prob <= 1.0
        assert prob > 0.3  # Good stats should have decent probability
    
    def test_calculate_goal_probability_with_poor_stats(self):
        """Test goal probability with poor stats."""
        analyzer = PlayerAnalyzer()
        
        player_stats = {
            'xg_per_90': 0.1,
            'recent_goals': [0, 0, 0, 0, 0],
            'shots_total': 5,
            'shots_on_target': 1,
            'minutes_played': 450,
            'is_home': False,
            'opponent_rating': 85.0
        }
        
        prob = analyzer.calculate_goal_probability(player_stats)
        
        assert 0.0 <= prob <= 1.0
        assert prob < 0.2  # Poor stats should have low probability
    
    def test_calculate_goal_probability_with_low_minutes(self):
        """Test goal probability with insufficient minutes."""
        analyzer = PlayerAnalyzer()
        
        player_stats = {
            'xg_per_90': 0.8,
            'recent_goals': [1, 1, 1],
            'shots_total': 10,
            'shots_on_target': 6,
            'minutes_played': 90,  # Only 1 match worth
            'is_home': True,
            'opponent_rating': 70.0
        }
        
        prob = analyzer.calculate_goal_probability(player_stats)
        
        # Should be penalized for low minutes
        assert prob < 0.3
    
    def test_calculate_assist_probability(self):
        """Test assist probability calculation."""
        analyzer = PlayerAnalyzer()
        
        player_stats = {
            'xa_per_90': 0.6,
            'recent_assists': [1, 1, 0, 2, 1],
            'key_passes': 3.5,
            'minutes_played': 450,
            'is_home': True,
            'opponent_rating': 70.0
        }
        
        prob = analyzer.calculate_assist_probability(player_stats)
        
        assert 0.0 <= prob <= 1.0
        assert prob > 0.3
    
    def test_calculate_clean_sheet_probability(self):
        """Test clean sheet probability calculation."""
        analyzer = PlayerAnalyzer()
        
        # Good defensive team
        team_stats = {
            'xg_conceded_per_90': 0.8,
            'recent_clean_sheets': 3,
            'recent_matches': 5,
            'opponent_xg_per_90': 1.0,
            'is_home': True
        }
        
        prob = analyzer.calculate_clean_sheet_probability(team_stats)
        
        assert 0.0 <= prob <= 1.0
        assert prob > 0.4  # Good defense should have decent probability
        
        # Poor defensive team
        team_stats = {
            'xg_conceded_per_90': 2.0,
            'recent_clean_sheets': 0,
            'recent_matches': 5,
            'opponent_xg_per_90': 1.8,
            'is_home': False
        }
        
        prob = analyzer.calculate_clean_sheet_probability(team_stats)
        
        assert 0.0 <= prob <= 1.0
        assert prob < 0.3  # Poor defense should have low probability
    
    def test_rank_players_by_metric(self):
        """Test player ranking."""
        analyzer = PlayerAnalyzer()
        
        players = [
            {'name': 'Player A', 'goal_probability': 0.8},
            {'name': 'Player B', 'goal_probability': 0.6},
            {'name': 'Player C', 'goal_probability': 0.9},
            {'name': 'Player D', 'goal_probability': 0.7}
        ]
        
        ranked = analyzer.rank_players_by_metric(players, 'goal_probability')
        
        # Should be sorted by goal_probability descending
        assert ranked[0]['name'] == 'Player C'
        assert ranked[0]['rank'] == 1
        assert ranked[1]['name'] == 'Player A'
        assert ranked[1]['rank'] == 2
        assert ranked[-1]['name'] == 'Player B'
        assert ranked[-1]['rank'] == 4
        
        # Test with top_n
        ranked_top_2 = analyzer.rank_players_by_metric(players, 'goal_probability', top_n=2)
        assert len(ranked_top_2) == 2
        assert ranked_top_2[0]['name'] == 'Player C'
    
    @patch('src.analysis.player_analyzer.FotMobScraper')
    def test_get_premier_league_fixtures(self, mock_fotmob_class):
        """Test getting Premier League fixtures."""
        # Mock FotMob scraper
        mock_fotmob = Mock()
        mock_fixtures = [
            {'match_id': 1, 'home_team': 'Team A', 'away_team': 'Team B'},
            {'match_id': 2, 'home_team': 'Team C', 'away_team': 'Team D'}
        ]
        mock_fotmob.get_premier_league_fixtures.return_value = mock_fixtures
        
        analyzer = PlayerAnalyzer(fotmob_scraper=mock_fotmob)
        
        fixtures = analyzer.get_premier_league_fixtures(matchweek=26)
        
        assert len(fixtures) == 2
        assert fixtures[0]['match_id'] == 1
        mock_fotmob.get_premier_league_fixtures.assert_called_once_with(26, '2025/2026')
    
    @patch('src.analysis.player_analyzer.FotMobScraper')
    def test_generate_analysis_report(self, mock_fotmob_class):
        """Test analysis report generation."""
        # Mock FotMob scraper
        mock_fotmob = Mock()
        mock_fixtures = [
            {'match_id': 1, 'home_team': 'Team A', 'away_team': 'Team B'}
        ]
        mock_fotmob.get_premier_league_fixtures.return_value = mock_fixtures
        
        analyzer = PlayerAnalyzer(fotmob_scraper=mock_fotmob)
        
        report = analyzer.generate_analysis_report(matchweek=26, top_n=20)
        
        assert 'matchweek' in report
        assert report['matchweek'] == 26
        assert 'fixtures' in report
        assert len(report['fixtures']) == 1
        assert 'top_goal_scorers' in report
        assert 'top_assist_providers' in report
        assert 'clean_sheet_candidates' in report
    
    def test_format_console_output(self):
        """Test console output formatting."""
        analyzer = PlayerAnalyzer()
        
        report = {
            'matchweek': 26,
            'fixtures': [
                {'home_team': 'Team A', 'away_team': 'Team B'}
            ],
            'top_goal_scorers': [
                {
                    'rank': 1,
                    'name': 'Test Player',
                    'team': 'Team A',
                    'opponent': 'Team B',
                    'goal_probability': 0.75,
                    'xg_per_90': 1.2,
                    'form_score': 8.0
                }
            ],
            'top_assist_providers': [],
            'clean_sheet_candidates': []
        }
        
        output = analyzer.format_console_output(report)
        
        assert 'PREMIER LEAGUE MATCHWEEK 26 ANALYSIS' in output
        assert 'Test Player' in output
        assert 'Team A' in output
        assert '75.0%' in output


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
