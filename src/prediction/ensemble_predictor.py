<<<<<<< HEAD
"""
Ensemble Predictor - With league context
"""

import re
from typing import Dict, List, Tuple
from math import exp, factorial
import sys
sys.path.insert(0, 'src')

from scrapers.free_data_fetcher import FreeDataFetcher
from prediction.dixon_coles import DixonColesModel

class EnsemblePredictor:
    """Advanced predictor with league-aware estimates."""
    
    def __init__(self):
        self.data_fetcher = FreeDataFetcher(debug=True)
        self.dixon_coles = DixonColesModel(rho=-0.13)
        self.league_avg = 2.6
        
    def parse_match(self, match_string: str) -> Dict:
        """Parse match string."""
        pattern = r'(.+?)\s+vs\s+(.+?)\s*\(([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\)'
        match = re.search(pattern, match_string, re.IGNORECASE)
        
        if match:
            return {
                'home_team': match.group(1).strip(),
                'away_team': match.group(2).strip(),
                'odds': {
                    'home': float(match.group(3)),
                    'draw': float(match.group(4)),
                    'away': float(match.group(5))
                }
            }
        return None
    
    def calculate_form_weight(self, form: List[str]) -> float:
        """Calculate form multiplier."""
        if not form:
            return 1.0
        
        weights = [0.4, 0.3, 0.15, 0.1, 0.05]
        score = 0
        
        for i, result in enumerate(form[:5]):
            weight = weights[i] if i < len(weights) else 0.05
            if result == 'W':
                score += 3 * weight
            elif result == 'D':
                score += 1 * weight
        
        max_score = sum(weights) * 3
        normalized = score / max_score if max_score > 0 else 0.5
        
        return 0.8 + (normalized * 0.4)
    
    def ensemble_xg(self, home_stats: Dict, away_stats: Dict, odds: Dict) -> Tuple[float, float]:
        """Calculate expected goals using league data + odds."""
        
        # Use league-specific averages
        home_attack = home_stats.get('home_avg_scored', 1.35)
        away_defense = away_stats.get('away_avg_conceded', 1.35)
        
        away_attack = away_stats.get('away_avg_scored', 1.10)
        home_defense = home_stats.get('home_avg_conceded', 1.20)
        
        # Base xG from league data
        home_xg_base = (home_attack + away_defense) / 2
        away_xg_base = (away_attack + home_defense) / 2
        
        # Adjust with odds
        home_prob = (1 / odds['home']) * 100
        away_prob = (1 / odds['away']) * 100
        total_prob = home_prob + (1/odds['draw'])*100 + away_prob
        
        home_prob = (home_prob / total_prob) * 100
        away_prob = (away_prob / total_prob) * 100
        
        # Blend league data (60%) with odds (40%)
        odds_factor_home = 0.8 + ((home_prob - 33.33) / 100)
        odds_factor_away = 0.8 + ((away_prob - 33.33) / 100)
        
        home_xg = (home_xg_base * 0.6) + (home_xg_base * odds_factor_home * 0.4)
        away_xg = (away_xg_base * 0.6) + (away_xg_base * odds_factor_away * 0.4)
        
        return round(max(0.5, min(3.5, home_xg)), 2), round(max(0.5, min(3.5, away_xg)), 2)
    
    def calculate_over_1_5(self, home_xg: float, away_xg: float) -> Dict:
        """Calculate Over 1.5."""
        result = self.dixon_coles.over_under_probability(home_xg, away_xg, threshold=1.5)
        over_prob = result['over']
        
        if over_prob >= 80:
            confidence, stars = "Very High", "⭐⭐⭐⭐⭐"
        elif over_prob >= 70:
            confidence, stars = "High", "⭐⭐⭐⭐"
        elif over_prob >= 60:
            confidence, stars = "Medium", "⭐⭐⭐"
        elif over_prob >= 50:
            confidence, stars = "Low-Medium", "⭐⭐"
        else:
            confidence, stars = "Low", "⭐"
        
        return {
            'over_1.5': round(over_prob, 1),
            'under_1.5': round(result['under'], 1),
            'prediction': 'Over 1.5' if over_prob > 50 else 'Under 1.5',
            'confidence': confidence,
            'stars': stars
        }
    
    def calculate_team_total(self, xg: float, team_name: str) -> Dict:
        """Calculate team scoring probability."""
        prob_0 = exp(-xg)
        over_0_5 = (1 - prob_0) * 100
        
        if over_0_5 >= 85:
            confidence = "Very High"
        elif over_0_5 >= 75:
            confidence = "High"
        elif over_0_5 >= 60:
            confidence = "Medium"
        elif over_0_5 >= 50:
            confidence = "Low-Medium"
        else:
            confidence = "Low"
        
        if over_0_5 >= 65:
            prediction = "YES"
        elif over_0_5 >= 55:
            prediction = "YES (Moderate)"
        elif over_0_5 >= 45:
            prediction = "MAYBE"
        else:
            prediction = "NO"
        
        return {
            'team': team_name,
            'over_0.5': round(over_0_5, 1),
            'prediction': prediction,
            'confidence': confidence,
            'xg': xg
        }
    
    def analyze_match(self, match_data: Dict, league: str = 'Default') -> Dict:
        """Analyze match."""
        home_team = match_data['home_team']
        away_team = match_data['away_team']
        odds = match_data['odds']
        
        print(f"\n🔍 {home_team} vs {away_team}")
        
        # Set league context
        self.data_fetcher.set_league(league)
        
        # Get team data
        home_stats = self.data_fetcher.get_team_data(home_team, is_home=True)
        away_stats = self.data_fetcher.get_team_data(away_team, is_home=False)
        
        # Calculate xG
        home_xg, away_xg = self.ensemble_xg(home_stats, away_stats, odds)
        
        print(f"   📊 Expected Goals: {home_xg} - {away_xg}")
        
        # Predictions
        over_1_5 = self.calculate_over_1_5(home_xg, away_xg)
        home_total = self.calculate_team_total(home_xg, home_team)
        away_total = self.calculate_team_total(away_xg, away_team)
        
        # BTTS
        btts_prob = self.dixon_coles.btts_probability(home_xg, away_xg)
        btts = {
            'probability': round(btts_prob, 1),
            'prediction': 'YES' if btts_prob > 55 else 'NO',
            'confidence': 'High' if btts_prob > 65 or btts_prob < 35 else 'Medium'
        }
        
        # Logical consistency
        if over_1_5['prediction'] == 'Over 1.5' and over_1_5['over_1.5'] > 60:
            if home_total['prediction'] in ['NO', 'MAYBE'] and away_total['prediction'] in ['NO', 'MAYBE']:
                if home_xg >= away_xg:
                    home_total['prediction'] = 'YES (Moderate)'
                    home_total['confidence'] = 'Medium'
                else:
                    away_total['prediction'] = 'YES (Moderate)'
                    away_total['confidence'] = 'Medium'
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'odds': odds,
            'home_stats': home_stats,
            'away_stats': away_stats,
            'expected_goals': {
                'home': home_xg,
                'away': away_xg,
                'total': round(home_xg + away_xg, 2)
            },
            'over_1_5': over_1_5,
            'home_total': home_total,
            'away_total': away_total,
            'btts': btts,
            'using_real_data': 'league_estimate' in home_stats.get('source', '')
        }
    
    def analyze_matches(self, match_strings: List[str], league: str = 'Default') -> List[Dict]:
        """Analyze multiple matches."""
        results = []
        
        print(f"\n{'='*90}")
        print(f"🚀 ANALYZING {len(match_strings)} MATCHES - {league.upper()}")
        print(f"{'='*90}")
        
        for i, match_string in enumerate(match_strings, 1):
            print(f"\n[{i}/{len(match_strings)}]", end=' ')
            
            match_data = self.parse_match(match_string)
            if match_data:
                analysis = self.analyze_match(match_data, league)
                results.append(analysis)
            else:
                print(f"❌ Could not parse: {match_string}")
        
        return results

=======
# ensemble_predictor.py content here... (actual content needs to be read) 
# This file contains the ensemble prediction logic for the football analytics project.


# further implementation here...
>>>>>>> 4cab15094e8a14ad692f4b9d37983ad988eb16a1
