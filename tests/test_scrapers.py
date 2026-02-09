"""
Unit tests for scraper modules.

Tests for SofaScore, FotMob, and 1xBet scrapers with mocked responses.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime

from src.scrapers.sofascore_scraper import SofaScoreScraper
from src.scrapers.fotmob_scraper import FotMobScraper
from src.scrapers.oneXbet_scraper import OneXBetScraper


class TestSofaScoreScraper:
    """Tests for SofaScore scraper."""
    
    def test_initialization(self):
        """Test scraper initialization."""
        scraper = SofaScoreScraper()
        assert scraper.base_url == "https://api.sofascore.com/api/v1"
        assert scraper.timeout == 30
        assert scraper.retry_attempts == 3
    
    @patch('src.scrapers.sofascore_scraper.requests.get')
    def test_get_match_statistics_success(self, mock_get):
        """Test successful match statistics retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'statistics': [
                {
                    'groupName': 'TVData',
                    'statisticsItems': [
                        {'name': 'Ball possession', 'home': 65, 'away': 35},
                        {'name': 'Total shots', 'home': 15, 'away': 8}
                    ]
                }
            ]
        }
        mock_get.return_value = mock_response
        
        scraper = SofaScoreScraper()
        result = scraper.get_match_statistics(12345)
        
        assert result is not None
        assert 'statistics' in result
        assert len(result['statistics']) == 1
    
    @patch('src.scrapers.sofascore_scraper.requests.get')
    def test_get_match_statistics_failure(self, mock_get):
        """Test failed match statistics retrieval."""
        mock_get.side_effect = Exception("Connection error")
        
        scraper = SofaScoreScraper()
        result = scraper.get_match_statistics(12345)
        
        assert result is None
    
    def test_parse_statistics_to_dataframe(self):
        """Test parsing statistics to DataFrame."""
        scraper = SofaScoreScraper()
        
        mock_stats = {
            'statistics': [
                {
                    'groupName': 'TVData',
                    'statisticsItems': [
                        {'name': 'Ball possession', 'home': 65, 'away': 35},
                        {'name': 'Total shots', 'home': 15, 'away': 8}
                    ]
                }
            ]
        }
        
        df = scraper.parse_statistics_to_dataframe(mock_stats)
        
        assert not df.empty
        assert len(df) == 2
        assert 'statistic' in df.columns
        assert 'home_value' in df.columns
        assert 'away_value' in df.columns


class TestFotMobScraper:
    """Tests for FotMob scraper."""
    
    def test_initialization(self):
        """Test scraper initialization."""
        scraper = FotMobScraper()
        assert scraper.base_url == "https://www.fotmob.com/api"
        assert scraper.timeout == 30
        assert scraper.retry_attempts == 3
    
    @patch('src.scrapers.fotmob_scraper.requests.get')
    def test_get_match_details_success(self, mock_get):
        """Test successful match details retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'header': {
                'teams': [
                    {'name': 'Team A', 'score': 2},
                    {'name': 'Team B', 'score': 1}
                ]
            }
        }
        mock_get.return_value = mock_response
        
        scraper = FotMobScraper()
        result = scraper.get_match_details(12345)
        
        assert result is not None
        assert 'header' in result
    
    def test_extract_xg_data(self):
        """Test xG data extraction."""
        scraper = FotMobScraper()
        
        mock_match_details = {
            'header': {
                'teams': [
                    {'name': 'Team A', 'score': 2},
                    {'name': 'Team B', 'score': 1}
                ]
            },
            'content': {
                'matchFacts': {
                    'xG': {
                        'home': 1.8,
                        'away': 1.2
                    }
                }
            }
        }
        
        xg_data = scraper.extract_xg_data(mock_match_details)
        
        assert xg_data is not None
        assert 'home_xg' in xg_data
        assert 'away_xg' in xg_data
        assert xg_data['home_xg'] == 1.8
        assert xg_data['away_xg'] == 1.2


class TestOneXBetScraper:
    """Tests for 1xBet scraper."""
    
    @patch('src.scrapers.oneXbet_scraper.webdriver.Chrome')
    @patch('src.scrapers.oneXbet_scraper.ChromeDriverManager')
    def test_initialization(self, mock_driver_manager, mock_chrome):
        """Test scraper initialization."""
        mock_driver_manager.return_value.install.return_value = "/path/to/driver"
        
        scraper = OneXBetScraper(headless=True)
        
        assert scraper.base_url == "https://1xbet.co.ke"
        assert scraper.headless is True
    
    def test_calculate_implied_probability_decimal(self):
        """Test implied probability calculation from decimal odds."""
        scraper = OneXBetScraper.__new__(OneXBetScraper)
        scraper.driver = None
        
        # Test decimal odds
        odds = 2.5
        implied_prob = scraper.calculate_implied_probability(odds, "decimal")
        
        assert implied_prob == 0.4
    
    def test_calculate_implied_probability_american(self):
        """Test implied probability calculation from American odds."""
        scraper = OneXBetScraper.__new__(OneXBetScraper)
        scraper.driver = None
        
        # Test American odds (positive)
        odds = 150
        implied_prob = scraper.calculate_implied_probability(odds, "american")
        
        assert implied_prob == pytest.approx(0.4, rel=0.01)
    
    def test_calculate_house_edge(self):
        """Test house edge calculation."""
        scraper = OneXBetScraper.__new__(OneXBetScraper)
        scraper.driver = None
        
        odds_list = [2.10, 3.50, 3.20]
        house_edge = scraper.calculate_house_edge(odds_list, "decimal")
        
        # Expected house edge should be positive
        assert house_edge > 0
        assert house_edge < 10  # Typically less than 10%


class TestDataProcessor:
    """Tests for data processing utilities."""
    
    def test_clean_match_data(self):
        """Test match data cleaning."""
        from src.utils.data_processor import clean_match_data
        
        # Create sample data with issues
        df = pd.DataFrame({
            'home_team': ['  Team A  ', 'Team B', 'Team C'],
            'away_team': ['Team B', 'Team C  ', 'Team A'],
            'home_goals': [2, None, 3],
            'away_goals': [1, 1, 2],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        
        cleaned = clean_match_data(df)
        
        # Check whitespace removed
        assert cleaned['home_team'].iloc[0] == 'Team A'
        
        # Check missing values filled
        assert cleaned['home_goals'].iloc[1] == 0
    
    def test_create_match_features(self):
        """Test feature creation."""
        from src.utils.data_processor import create_match_features
        
        df = pd.DataFrame({
            'home_team': ['Team A', 'Team B'],
            'away_team': ['Team B', 'Team A'],
            'home_goals': [2, 1],
            'away_goals': [1, 2],
            'home_xg': [1.8, 1.2],
            'away_xg': [1.1, 1.9]
        })
        
        featured = create_match_features(df)
        
        assert 'total_goals' in featured.columns
        assert 'goal_difference' in featured.columns
        assert 'result' in featured.columns
        assert featured['total_goals'].iloc[0] == 3


class TestPoissonModel:
    """Tests for Poisson model."""
    
    def test_initialization(self):
        """Test model initialization."""
        from src.models.poisson_model import PoissonModel
        
        model = PoissonModel(home_advantage=1.3)
        assert model.home_advantage == 1.3
        assert model.trained is False
    
    def test_train(self):
        """Test model training."""
        from src.models.poisson_model import PoissonModel
        
        # Create sample training data
        df = pd.DataFrame({
            'home_team': ['Team A', 'Team B', 'Team C'] * 10,
            'away_team': ['Team B', 'Team C', 'Team A'] * 10,
            'home_goals': [2, 1, 3, 1, 2, 0, 3, 1, 2, 1] * 3,
            'away_goals': [1, 1, 2, 0, 2, 1, 1, 2, 1, 0] * 3
        })
        
        model = PoissonModel()
        success = model.train(df)
        
        assert success is True
        assert model.trained is True
        assert model.league_avg_goals > 0


class TestEdgeCalculator:
    """Tests for edge calculator."""
    
    def test_initialization(self):
        """Test calculator initialization."""
        from src.models.edge_calculator import EdgeCalculator
        
        calc = EdgeCalculator(kelly_fraction=0.25, max_stake_pct=5.0)
        assert calc.kelly_fraction == 0.25
        assert calc.max_stake_pct == 5.0
    
    def test_calculate_implied_probability(self):
        """Test implied probability calculation."""
        from src.models.edge_calculator import EdgeCalculator
        
        calc = EdgeCalculator()
        odds = 2.5
        implied_prob = calc.calculate_implied_probability(odds)
        
        assert implied_prob == 0.4
    
    def test_calculate_edge(self):
        """Test edge calculation."""
        from src.models.edge_calculator import EdgeCalculator
        
        calc = EdgeCalculator()
        model_prob = 0.50
        odds = 2.5
        
        edge = calc.calculate_edge(model_prob, odds)
        
        # Edge should be positive (value bet)
        assert edge > 0
    
    def test_calculate_kelly_stake(self):
        """Test Kelly Criterion stake calculation."""
        from src.models.edge_calculator import EdgeCalculator
        
        calc = EdgeCalculator(kelly_fraction=0.25, max_stake_pct=5.0)
        
        result = calc.calculate_kelly_stake(
            model_probability=0.50,
            bookmaker_odds=2.5,
            bankroll=1000
        )
        
        assert 'recommended_stake' in result
        assert 'potential_profit' in result
        assert result['recommended_stake'] >= 0
        assert result['recommended_stake'] <= 50  # Max 5% of bankroll


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
