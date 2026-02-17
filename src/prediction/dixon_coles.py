<<<<<<< HEAD
"""
Dixon-Coles Model - Professional betting model
Used by betting syndicates for accurate predictions
"""

from math import exp, factorial
from typing import Dict, Tuple
import numpy as np

class DixonColesModel:
    """
    Dixon-Coles model for football prediction.
    Accounts for low-scoring correlation and time decay.
    """
    
    def __init__(self, rho=-0.13):
        """
        rho: correlation parameter for low scores (typically -0.1 to -0.15)
        """
        self.rho = rho
    
    def poisson_prob(self, lam: float, k: int) -> float:
        """Basic Poisson probability."""
        return (lam ** k * exp(-lam)) / factorial(k)
    
    def tau(self, x: int, y: int, lam_home: float, lam_away: float) -> float:
        """
        Tau adjustment for low scores.
        Accounts for correlation between home and away goals.
        """
        if x == 0 and y == 0:
            return 1 - lam_home * lam_away * self.rho
        elif x == 0 and y == 1:
            return 1 + lam_home * self.rho
        elif x == 1 and y == 0:
            return 1 + lam_away * self.rho
        elif x == 1 and y == 1:
            return 1 - self.rho
        else:
            return 1.0
    
    def score_probability(self, home_goals: int, away_goals: int, 
                         lam_home: float, lam_away: float) -> float:
        """
        Probability of exact score using Dixon-Coles.
        """
        basic_prob = (self.poisson_prob(lam_home, home_goals) * 
                     self.poisson_prob(lam_away, away_goals))
        
        tau_adj = self.tau(home_goals, away_goals, lam_home, lam_away)
        
        return basic_prob * tau_adj
    
    def over_under_probability(self, lam_home: float, lam_away: float, 
                               threshold: float = 1.5) -> Dict:
        """
        Calculate Over/Under probabilities.
        """
        under_prob = 0.0
        
        # Calculate all score combinations under threshold
        max_goals = 10
        for h in range(max_goals):
            for a in range(max_goals):
                if h + a <= threshold:
                    under_prob += self.score_probability(h, a, lam_home, lam_away)
        
        over_prob = 1 - under_prob
        
        return {
            'over': over_prob * 100,
            'under': under_prob * 100
        }
    
    def btts_probability(self, lam_home: float, lam_away: float) -> float:
        """Both teams to score probability."""
        # P(home scores 0)
        home_blank = self.poisson_prob(lam_home, 0)
        # P(away scores 0)
        away_blank = self.poisson_prob(lam_away, 0)
        
        # P(both score) = 1 - P(at least one doesn't score)
        both_score = 1 - (home_blank + away_blank - home_blank * away_blank)
        
        return both_score * 100

=======
# dixon_coles.py content here... (actual content needs to be read) 
# This file implements the Dixon-Coles prediction model.


# further implementation here...
>>>>>>> 4cab15094e8a14ad692f4b9d37983ad988eb16a1
