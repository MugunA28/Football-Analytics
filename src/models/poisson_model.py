"""
Poisson-based Match Prediction Model

This module implements a Poisson distribution-based model for predicting football match outcomes.
It calculates team attack/defense strengths and uses Poisson distributions to predict goals and probabilities.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PoissonModel:
    """
    A Poisson-based model for predicting football match outcomes.
    
    The model calculates team attack and defense strengths based on historical data
    and uses Poisson distributions to predict expected goals and match probabilities.
    
    Attributes:
        home_advantage (float): Home advantage factor
        league_avg_goals (float): Average goals per match in the league
        attack_strength (dict): Attack strength for each team
        defense_strength (dict): Defense strength for each team
        trained (bool): Whether the model has been trained
    """
    
    def __init__(self, home_advantage: float = 1.3):
        """
        Initialize the Poisson model.
        
        Args:
            home_advantage: Factor representing home advantage (default 1.3)
        """
        self.home_advantage = home_advantage
        self.league_avg_goals = 0.0
        self.attack_strength = {}
        self.defense_strength = {}
        self.trained = False
        
        logger.info(f"Poisson model initialized with home advantage: {home_advantage}")
    
    def train(self, matches_df: pd.DataFrame, min_matches: int = 20) -> bool:
        """
        Train the model on historical match data.
        
        Args:
            matches_df: DataFrame with columns: home_team, away_team, home_goals, away_goals
            min_matches: Minimum number of matches required for training
        
        Returns:
            True if training was successful, False otherwise
        """
        try:
            logger.info(f"Training Poisson model on {len(matches_df)} matches")
            
            if len(matches_df) < min_matches:
                logger.error(f"Insufficient matches for training. Need at least {min_matches}, got {len(matches_df)}")
                return False
            
            # Validate required columns
            required_columns = ['home_team', 'away_team', 'home_goals', 'away_goals']
            if not all(col in matches_df.columns for col in required_columns):
                logger.error(f"Missing required columns. Need: {required_columns}")
                return False
            
            # Calculate league average goals
            total_goals = matches_df['home_goals'].sum() + matches_df['away_goals'].sum()
            total_matches = len(matches_df)
            self.league_avg_goals = total_goals / (2 * total_matches)
            
            logger.info(f"League average goals per team: {self.league_avg_goals:.2f}")
            
            # Calculate attack and defense strengths for each team
            teams = set(matches_df['home_team'].unique()) | set(matches_df['away_team'].unique())
            
            for team in teams:
                # Home matches
                home_matches = matches_df[matches_df['home_team'] == team]
                home_goals_scored = home_matches['home_goals'].sum()
                home_goals_conceded = home_matches['away_goals'].sum()
                home_count = len(home_matches)
                
                # Away matches
                away_matches = matches_df[matches_df['away_team'] == team]
                away_goals_scored = away_matches['away_goals'].sum()
                away_goals_conceded = away_matches['home_goals'].sum()
                away_count = len(away_matches)
                
                # Total matches
                total_team_matches = home_count + away_count
                
                if total_team_matches == 0:
                    logger.warning(f"No matches found for team {team}")
                    continue
                
                # Calculate attack strength
                # Attack strength = (goals scored / matches) / league average
                avg_goals_scored = (home_goals_scored + away_goals_scored) / total_team_matches
                attack_strength = avg_goals_scored / self.league_avg_goals if self.league_avg_goals > 0 else 1.0
                
                # Calculate defense strength
                # Defense strength = (goals conceded / matches) / league average
                avg_goals_conceded = (home_goals_conceded + away_goals_conceded) / total_team_matches
                defense_strength = avg_goals_conceded / self.league_avg_goals if self.league_avg_goals > 0 else 1.0
                
                self.attack_strength[team] = attack_strength
                self.defense_strength[team] = defense_strength
                
                logger.debug(f"{team}: Attack={attack_strength:.2f}, Defense={defense_strength:.2f}")
            
            self.trained = True
            logger.info(f"Model trained successfully on {len(teams)} teams")
            return True
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict_score(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Predict expected goals for a match using Poisson distribution.
        
        Args:
            home_team: Name of the home team
            away_team: Name of the away team
        
        Returns:
            Dictionary with expected goals for home and away teams
        """
        if not self.trained:
            logger.error("Model has not been trained yet")
            return {'home_expected_goals': 0.0, 'away_expected_goals': 0.0}
        
        # Get team strengths (use 1.0 as default for unknown teams)
        home_attack = self.attack_strength.get(home_team, 1.0)
        home_defense = self.defense_strength.get(home_team, 1.0)
        away_attack = self.attack_strength.get(away_team, 1.0)
        away_defense = self.defense_strength.get(away_team, 1.0)
        
        # Calculate expected goals
        # Home expected goals = league avg * home attack * away defense * home advantage
        home_expected = self.league_avg_goals * home_attack * away_defense * self.home_advantage
        
        # Away expected goals = league avg * away attack * home defense
        away_expected = self.league_avg_goals * away_attack * home_defense
        
        logger.info(f"Predicted score - {home_team}: {home_expected:.2f}, {away_team}: {away_expected:.2f}")
        
        return {
            'home_expected_goals': home_expected,
            'away_expected_goals': away_expected
        }
    
    def predict_match_outcome(self, home_team: str, away_team: str, max_goals: int = 10) -> Dict:
        """
        Predict match outcome probabilities (win/draw/loss) and various betting markets.
        
        Args:
            home_team: Name of the home team
            away_team: Name of the away team
            max_goals: Maximum number of goals to consider in calculations
        
        Returns:
            Dictionary with comprehensive prediction data including probabilities
        """
        # Get expected goals
        score_prediction = self.predict_score(home_team, away_team)
        home_expected = score_prediction['home_expected_goals']
        away_expected = score_prediction['away_expected_goals']
        
        # Calculate outcome probabilities
        prob_home_win = 0.0
        prob_draw = 0.0
        prob_away_win = 0.0
        
        # Calculate probabilities for all score combinations up to max_goals
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob_score = (poisson.pmf(home_goals, home_expected) * 
                             poisson.pmf(away_goals, away_expected))
                
                if home_goals > away_goals:
                    prob_home_win += prob_score
                elif home_goals == away_goals:
                    prob_draw += prob_score
                else:
                    prob_away_win += prob_score
        
        # Calculate over/under 2.5 goals probabilities
        total_expected_goals = home_expected + away_expected
        prob_under_25 = sum([poisson.pmf(i, total_expected_goals) for i in range(3)])
        prob_over_25 = 1 - prob_under_25
        
        # Calculate BTTS (Both Teams To Score) probability
        prob_home_no_goals = poisson.pmf(0, home_expected)
        prob_away_no_goals = poisson.pmf(0, away_expected)
        prob_btts_yes = (1 - prob_home_no_goals) * (1 - prob_away_no_goals)
        prob_btts_no = 1 - prob_btts_yes
        
        prediction = {
            'home_team': home_team,
            'away_team': away_team,
            'home_expected_goals': round(home_expected, 2),
            'away_expected_goals': round(away_expected, 2),
            'total_expected_goals': round(total_expected_goals, 2),
            'probabilities': {
                'home_win': round(prob_home_win, 4),
                'draw': round(prob_draw, 4),
                'away_win': round(prob_away_win, 4),
                'over_2.5': round(prob_over_25, 4),
                'under_2.5': round(prob_under_25, 4),
                'btts_yes': round(prob_btts_yes, 4),
                'btts_no': round(prob_btts_no, 4)
            }
        }
        
        logger.info(f"Match prediction - Home: {prob_home_win:.2%}, Draw: {prob_draw:.2%}, Away: {prob_away_win:.2%}")
        
        return prediction
    
    def find_value_bets(self, predictions: Dict, bookmaker_odds: Dict, min_edge: float = 0.03) -> List[Dict]:
        """
        Identify value bets by comparing model predictions to bookmaker odds.
        
        Args:
            predictions: Dictionary with model predictions
            bookmaker_odds: Dictionary with bookmaker odds (decimal format)
            min_edge: Minimum edge required to consider a value bet (default 3%)
        
        Returns:
            List of value bets with edge calculations
        """
        value_bets = []
        
        try:
            model_probs = predictions.get('probabilities', {})
            
            # Define market mappings
            markets = {
                'home_win': 'home_odds',
                'draw': 'draw_odds',
                'away_win': 'away_odds',
                'over_2.5': 'over_2.5_odds',
                'under_2.5': 'under_2.5_odds',
                'btts_yes': 'btts_yes_odds',
                'btts_no': 'btts_no_odds'
            }
            
            for outcome, odds_key in markets.items():
                if outcome not in model_probs or odds_key not in bookmaker_odds:
                    continue
                
                model_prob = model_probs[outcome]
                odds = bookmaker_odds[odds_key]
                
                if odds <= 1.0:
                    continue
                
                # Calculate implied probability from odds
                implied_prob = 1.0 / odds
                
                # Calculate edge: model probability - implied probability
                edge = model_prob - implied_prob
                
                if edge >= min_edge:
                    value_bet = {
                        'match': f"{predictions['home_team']} vs {predictions['away_team']}",
                        'market': outcome,
                        'model_probability': round(model_prob, 4),
                        'odds': odds,
                        'implied_probability': round(implied_prob, 4),
                        'edge': round(edge, 4),
                        'edge_percent': round(edge * 100, 2)
                    }
                    value_bets.append(value_bet)
                    logger.info(f"Value bet found: {outcome} with {edge*100:.2f}% edge")
            
            logger.info(f"Found {len(value_bets)} value bets with edge >= {min_edge*100}%")
            return value_bets
        
        except Exception as e:
            logger.error(f"Error finding value bets: {e}")
            return []
    
    def get_model_stats(self) -> Dict:
        """
        Get statistics about the trained model.
        
        Returns:
            Dictionary with model statistics
        """
        if not self.trained:
            return {'trained': False}
        
        return {
            'trained': True,
            'num_teams': len(self.attack_strength),
            'league_avg_goals': round(self.league_avg_goals, 2),
            'home_advantage': self.home_advantage,
            'top_attack': sorted(self.attack_strength.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_defense': sorted(self.defense_strength.items(), key=lambda x: x[1])[:5]
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample training data
    sample_data = pd.DataFrame({
        'home_team': ['Team A', 'Team B', 'Team C', 'Team A', 'Team B'],
        'away_team': ['Team B', 'Team C', 'Team A', 'Team C', 'Team A'],
        'home_goals': [2, 1, 3, 1, 2],
        'away_goals': [1, 1, 2, 0, 2]
    })
    
    # Train model
    model = PoissonModel(home_advantage=1.3)
    if model.train(sample_data):
        # Predict match
        prediction = model.predict_match_outcome('Team A', 'Team B')
        print(f"Prediction: {prediction}")
        
        # Find value bets
        bookmaker_odds = {
            'home_odds': 2.5,
            'draw_odds': 3.2,
            'away_odds': 2.8,
            'over_2.5_odds': 1.9,
            'under_2.5_odds': 2.0
        }
        value_bets = model.find_value_bets(prediction, bookmaker_odds)
        print(f"Value bets: {value_bets}")
