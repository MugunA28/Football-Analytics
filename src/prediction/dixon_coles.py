"""
Dixon-Coles Model Implementation
Statistical model for football match prediction
"""

from math import exp, factorial
from typing import Dict

class DixonColesModel:
    """Dixon-Coles statistical model for football predictions."""
    
    def __init__(self, rho=-0.13):
        """
        Initialize Dixon-Coles model.
        
        Args:
            rho: Correlation parameter for low scores (typically -0.13)
        """
        self.rho = rho
    
    def tau(self, x: int, y: int, lambda1: float, lambda2: float) -> float:
        """
        Dixon-Coles correlation adjustment for low scores.
        
        Args:
            x: Home team goals
            y: Away team goals
            lambda1: Home team expected goals
            lambda2: Away team expected goals
        
        Returns:
            Correlation adjustment factor
        """
        if x == 0 and y == 0:
            return 1 - lambda1 * lambda2 * self.rho
        elif x == 0 and y == 1:
            return 1 + lambda1 * self.rho
        elif x == 1 and y == 0:
            return 1 + lambda2 * self.rho
        elif x == 1 and y == 1:
            return 1 - self.rho
        else:
            return 1.0
    
    def poisson_prob(self, k: int, lambda_param: float) -> float:
        """
        Calculate Poisson probability.
        
        Args:
            k: Number of goals
            lambda_param: Expected goals (lambda)
        
        Returns:
            Probability of exactly k goals
        """
        if lambda_param <= 0:
            lambda_param = 0.01
        
        return (lambda_param ** k) * exp(-lambda_param) / factorial(k)
    
    def match_probability(self, home_goals: int, away_goals: int, 
                         home_lambda: float, away_lambda: float) -> float:
        """
        Calculate probability of specific scoreline using Dixon-Coles.
        
        Args:
            home_goals: Home team goals
            away_goals: Away team goals
            home_lambda: Home team expected goals
            away_lambda: Away team expected goals
        
        Returns:
            Probability of the scoreline
        """
        tau_adj = self.tau(home_goals, away_goals, home_lambda, away_lambda)
        prob_home = self.poisson_prob(home_goals, home_lambda)
        prob_away = self.poisson_prob(away_goals, away_lambda)
        
        return tau_adj * prob_home * prob_away
    
    def over_under_probability(self, home_lambda: float, away_lambda: float, 
                              threshold: float = 2.5) -> Dict[str, float]:
        """
        Calculate Over/Under probability for total goals.
        
        Args:
            home_lambda: Home team expected goals
            away_lambda: Away team expected goals
            threshold: Goals threshold (e.g., 2.5)
        
        Returns:
            Dict with 'over' and 'under' probabilities (as percentages)
        """
        under_prob = 0.0
        threshold_int = int(threshold)
        
        # Sum probabilities for all scorelines under threshold
        for home_goals in range(threshold_int + 2):
            for away_goals in range(threshold_int + 2):
                if home_goals + away_goals < threshold:
                    under_prob += self.match_probability(
                        home_goals, away_goals, home_lambda, away_lambda
                    )
        
        over_prob = 1.0 - under_prob
        
        return {
            'over': over_prob * 100,
            'under': under_prob * 100
        }
    
    def btts_probability(self, home_lambda: float, away_lambda: float) -> float:
        """
        Calculate Both Teams To Score (BTTS) probability.
        
        Args:
            home_lambda: Home team expected goals
            away_lambda: Away team expected goals
        
        Returns:
            BTTS probability (as percentage)
        """
        # P(both score) = 1 - P(home 0) - P(away 0) + P(both 0)
        prob_home_zero = exp(-home_lambda)
        prob_away_zero = exp(-away_lambda)
        prob_both_zero = prob_home_zero * prob_away_zero * self.tau(0, 0, home_lambda, away_lambda)
        
        btts_prob = 1 - prob_home_zero - prob_away_zero + prob_both_zero
        
        return btts_prob * 100

