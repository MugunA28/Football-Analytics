"""
Edge Calculator and Value Bet Identifier

This module provides functionality to calculate betting edges, identify value bets,
and implement Kelly Criterion for optimal stake sizing with risk management.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EdgeCalculator:
    """
    Calculator for identifying value bets and determining optimal stake sizes.
    
    Implements edge calculation, Kelly Criterion, and portfolio risk management
    for betting strategy optimization.
    
    Attributes:
        kelly_fraction (float): Fractional Kelly Criterion for conservative betting
        max_stake_pct (float): Maximum stake as percentage of bankroll
        min_odds (float): Minimum acceptable odds
        max_odds (float): Maximum acceptable odds
    """
    
    def __init__(self, kelly_fraction: float = 0.25, max_stake_pct: float = 5.0,
                 min_odds: float = 1.5, max_odds: float = 10.0):
        """
        Initialize the edge calculator.
        
        Args:
            kelly_fraction: Fractional Kelly (0.25 = quarter Kelly, default)
            max_stake_pct: Maximum stake as percentage of bankroll
            min_odds: Minimum acceptable odds for betting
            max_odds: Maximum acceptable odds for betting
        """
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct
        self.min_odds = min_odds
        self.max_odds = max_odds
        
        logger.info(f"Edge calculator initialized - Kelly fraction: {kelly_fraction}, Max stake: {max_stake_pct}%")
    
    def calculate_implied_probability(self, odds: float, odds_format: str = "decimal") -> float:
        """
        Calculate implied probability from betting odds.
        
        Args:
            odds: The betting odds
            odds_format: Format of odds ('decimal' or 'american')
        
        Returns:
            Implied probability as a decimal (0-1)
        """
        try:
            if odds_format == "decimal":
                if odds <= 1.0:
                    logger.warning(f"Invalid decimal odds: {odds}")
                    return 0.0
                return 1.0 / odds
            
            elif odds_format == "american":
                if odds > 0:
                    return 100 / (odds + 100)
                else:
                    return abs(odds) / (abs(odds) + 100)
            
            else:
                logger.error(f"Unsupported odds format: {odds_format}")
                return 0.0
        
        except Exception as e:
            logger.error(f"Error calculating implied probability: {e}")
            return 0.0
    
    def calculate_true_probability(self, odds_list: List[float], odds_format: str = "decimal") -> List[float]:
        """
        Remove bookmaker margin to get true probabilities.
        
        Args:
            odds_list: List of odds for all outcomes in a market
            odds_format: Format of odds ('decimal' or 'american')
        
        Returns:
            List of true probabilities after removing margin
        """
        try:
            # Calculate implied probabilities
            implied_probs = [self.calculate_implied_probability(odd, odds_format) for odd in odds_list]
            
            # Calculate total (includes bookmaker margin)
            total_implied = sum(implied_probs)
            
            if total_implied <= 1.0:
                logger.warning("Total implied probability <= 1, no margin to remove")
                return implied_probs
            
            # Remove margin proportionally
            true_probs = [prob / total_implied for prob in implied_probs]
            
            logger.debug(f"Margin removed: {(total_implied - 1.0) * 100:.2f}%")
            return true_probs
        
        except Exception as e:
            logger.error(f"Error calculating true probabilities: {e}")
            return [0.0] * len(odds_list)
    
    def calculate_edge(self, model_probability: float, bookmaker_odds: float,
                      odds_format: str = "decimal", remove_margin: bool = False,
                      market_odds: Optional[List[float]] = None) -> float:
        """
        Calculate betting edge.
        
        Edge = Model Probability - Implied Probability
        Positive edge indicates a value bet.
        
        Args:
            model_probability: Probability from prediction model
            bookmaker_odds: Odds offered by bookmaker
            odds_format: Format of odds ('decimal' or 'american')
            remove_margin: Whether to remove bookmaker margin
            market_odds: All odds in the market (needed if remove_margin=True)
        
        Returns:
            Edge as a decimal (positive = value bet)
        """
        try:
            implied_prob = self.calculate_implied_probability(bookmaker_odds, odds_format)
            
            if remove_margin and market_odds:
                # Find the index of our odds in the market
                # and get the true probability
                true_probs = self.calculate_true_probability(market_odds, odds_format)
                idx = market_odds.index(bookmaker_odds) if bookmaker_odds in market_odds else 0
                implied_prob = true_probs[idx]
            
            edge = model_probability - implied_prob
            
            logger.debug(f"Edge calculated: {edge:.4f} (Model: {model_probability:.4f}, Implied: {implied_prob:.4f})")
            return edge
        
        except Exception as e:
            logger.error(f"Error calculating edge: {e}")
            return 0.0
    
    def calculate_kelly_stake(self, model_probability: float, bookmaker_odds: float,
                            bankroll: float, odds_format: str = "decimal") -> Dict:
        """
        Calculate optimal stake using Kelly Criterion.
        
        Kelly Formula: f = (bp - q) / b
        where:
            f = fraction of bankroll to bet
            b = net odds (decimal odds - 1)
            p = probability of winning
            q = probability of losing (1 - p)
        
        Args:
            model_probability: Probability of outcome from model
            bookmaker_odds: Odds offered by bookmaker
            bankroll: Current bankroll size
            odds_format: Format of odds ('decimal' or 'american')
        
        Returns:
            Dictionary with stake calculations
        """
        try:
            # Convert to decimal odds if needed
            if odds_format == "american":
                if bookmaker_odds > 0:
                    decimal_odds = (bookmaker_odds / 100) + 1
                else:
                    decimal_odds = (100 / abs(bookmaker_odds)) + 1
            else:
                decimal_odds = bookmaker_odds
            
            # Calculate net odds
            net_odds = decimal_odds - 1
            
            # Calculate Kelly percentage
            p = model_probability
            q = 1 - p
            
            kelly_pct = (net_odds * p - q) / net_odds
            
            # Apply fractional Kelly
            fractional_kelly_pct = kelly_pct * self.kelly_fraction
            
            # Apply maximum stake limit
            stake_pct = min(fractional_kelly_pct, self.max_stake_pct / 100)
            
            # Ensure non-negative stake
            stake_pct = max(0, stake_pct)
            
            # Calculate stake amount
            stake = bankroll * stake_pct
            
            # Calculate potential profit
            potential_profit = stake * net_odds
            
            result = {
                'kelly_pct': round(kelly_pct * 100, 2),
                'fractional_kelly_pct': round(fractional_kelly_pct * 100, 2),
                'recommended_stake_pct': round(stake_pct * 100, 2),
                'recommended_stake': round(stake, 2),
                'potential_profit': round(potential_profit, 2),
                'potential_return': round(stake + potential_profit, 2)
            }
            
            logger.debug(f"Kelly stake calculated: {stake_pct * 100:.2f}% = ${stake:.2f}")
            return result
        
        except Exception as e:
            logger.error(f"Error calculating Kelly stake: {e}")
            return {
                'kelly_pct': 0.0,
                'fractional_kelly_pct': 0.0,
                'recommended_stake_pct': 0.0,
                'recommended_stake': 0.0,
                'potential_profit': 0.0,
                'potential_return': 0.0
            }
    
    def find_value_bets(self, predictions: Dict, bookmaker_odds: Dict,
                       min_edge: float = 0.03, bankroll: float = 1000) -> pd.DataFrame:
        """
        Find value bets from predictions and odds, with stake recommendations.
        
        Args:
            predictions: Dictionary with model predictions and probabilities
            bookmaker_odds: Dictionary with bookmaker odds
            min_edge: Minimum edge required to consider a value bet (default 3%)
            bankroll: Current bankroll for stake calculations
        
        Returns:
            DataFrame with value bets sorted by edge
        """
        try:
            logger.info(f"Finding value bets with minimum edge: {min_edge * 100}%")
            
            value_bets = []
            
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
                
                # Check odds range
                if odds < self.min_odds or odds > self.max_odds:
                    continue
                
                # Calculate edge
                edge = self.calculate_edge(model_prob, odds)
                
                # Check if it's a value bet
                if edge >= min_edge:
                    # Calculate Kelly stake
                    stake_info = self.calculate_kelly_stake(model_prob, odds, bankroll)
                    
                    value_bet = {
                        'match': f"{predictions.get('home_team', 'Unknown')} vs {predictions.get('away_team', 'Unknown')}",
                        'market': outcome,
                        'model_probability': round(model_prob, 4),
                        'odds': odds,
                        'implied_probability': round(self.calculate_implied_probability(odds), 4),
                        'edge': round(edge, 4),
                        'edge_percent': round(edge * 100, 2),
                        'recommended_stake': stake_info['recommended_stake'],
                        'stake_pct': stake_info['recommended_stake_pct'],
                        'potential_profit': stake_info['potential_profit'],
                        'potential_return': stake_info['potential_return']
                    }
                    value_bets.append(value_bet)
                    logger.info(f"Value bet: {outcome} with {edge*100:.2f}% edge")
            
            # Create DataFrame and sort by edge
            df = pd.DataFrame(value_bets)
            if not df.empty:
                df = df.sort_values('edge', ascending=False)
            
            logger.info(f"Found {len(df)} value bets")
            return df
        
        except Exception as e:
            logger.error(f"Error finding value bets: {e}")
            return pd.DataFrame()
    
    def calculate_portfolio_metrics(self, value_bets_df: pd.DataFrame, bankroll: float) -> Dict:
        """
        Calculate portfolio-level metrics for risk management.
        
        Args:
            value_bets_df: DataFrame with value bets
            bankroll: Current bankroll
        
        Returns:
            Dictionary with portfolio metrics
        """
        if value_bets_df.empty:
            logger.warning("No value bets to analyze")
            return {}
        
        try:
            total_stake = value_bets_df['recommended_stake'].sum()
            total_potential_profit = value_bets_df['potential_profit'].sum()
            avg_edge = value_bets_df['edge_percent'].mean()
            max_edge = value_bets_df['edge_percent'].max()
            num_bets = len(value_bets_df)
            
            # Calculate exposure
            exposure_pct = (total_stake / bankroll) * 100 if bankroll > 0 else 0
            
            # Calculate weighted average odds
            if total_stake > 0:
                weighted_avg_odds = (
                    (value_bets_df['odds'] * value_bets_df['recommended_stake']).sum() / total_stake
                )
            else:
                weighted_avg_odds = 0
            
            # Calculate expected value
            expected_value = sum(
                row['recommended_stake'] * (row['odds'] - 1) * row['model_probability']
                for _, row in value_bets_df.iterrows()
            )
            
            metrics = {
                'num_bets': num_bets,
                'total_stake': round(total_stake, 2),
                'total_potential_profit': round(total_potential_profit, 2),
                'avg_edge': round(avg_edge, 2),
                'max_edge': round(max_edge, 2),
                'exposure_pct': round(exposure_pct, 2),
                'weighted_avg_odds': round(weighted_avg_odds, 2),
                'expected_value': round(expected_value, 2),
                'expected_roi': round((expected_value / total_stake * 100), 2) if total_stake > 0 else 0
            }
            
            logger.info(f"Portfolio metrics: {num_bets} bets, ${total_stake:.2f} total stake, {exposure_pct:.2f}% exposure")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def filter_bets_by_criteria(self, value_bets_df: pd.DataFrame,
                               min_edge: Optional[float] = None,
                               max_stake: Optional[float] = None,
                               min_odds: Optional[float] = None,
                               max_odds: Optional[float] = None) -> pd.DataFrame:
        """
        Filter value bets based on additional criteria.
        
        Args:
            value_bets_df: DataFrame with value bets
            min_edge: Minimum edge percentage
            max_stake: Maximum stake amount
            min_odds: Minimum odds
            max_odds: Maximum odds
        
        Returns:
            Filtered DataFrame
        """
        if value_bets_df.empty:
            return value_bets_df
        
        try:
            filtered_df = value_bets_df.copy()
            
            if min_edge is not None:
                filtered_df = filtered_df[filtered_df['edge_percent'] >= min_edge]
            
            if max_stake is not None:
                filtered_df = filtered_df[filtered_df['recommended_stake'] <= max_stake]
            
            if min_odds is not None:
                filtered_df = filtered_df[filtered_df['odds'] >= min_odds]
            
            if max_odds is not None:
                filtered_df = filtered_df[filtered_df['odds'] <= max_odds]
            
            logger.info(f"Filtered bets: {len(value_bets_df)} -> {len(filtered_df)}")
            return filtered_df
        
        except Exception as e:
            logger.error(f"Error filtering bets: {e}")
            return value_bets_df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    calculator = EdgeCalculator(kelly_fraction=0.25, max_stake_pct=5.0)
    
    # Example: Calculate implied probability
    odds = 2.5
    implied_prob = calculator.calculate_implied_probability(odds)
    print(f"Odds: {odds}, Implied Probability: {implied_prob:.2%}")
    
    # Example: Calculate edge
    model_prob = 0.50
    edge = calculator.calculate_edge(model_prob, odds)
    print(f"Model Prob: {model_prob:.2%}, Edge: {edge:.2%}")
    
    # Example: Calculate Kelly stake
    bankroll = 1000
    stake_info = calculator.calculate_kelly_stake(model_prob, odds, bankroll)
    print(f"Recommended stake: ${stake_info['recommended_stake']:.2f} ({stake_info['recommended_stake_pct']:.2f}%)")
    
    # Example: Find value bets
    predictions = {
        'home_team': 'Team A',
        'away_team': 'Team B',
        'probabilities': {
            'home_win': 0.50,
            'draw': 0.25,
            'away_win': 0.25,
            'over_2.5': 0.60,
            'under_2.5': 0.40
        }
    }
    
    bookmaker_odds = {
        'home_odds': 2.5,
        'draw_odds': 3.2,
        'away_odds': 3.0,
        'over_2.5_odds': 1.8,
        'under_2.5_odds': 2.1
    }
    
    value_bets = calculator.find_value_bets(predictions, bookmaker_odds, min_edge=0.03, bankroll=1000)
    print(f"\nValue Bets:\n{value_bets}")
    
    if not value_bets.empty:
        portfolio = calculator.calculate_portfolio_metrics(value_bets, bankroll)
        print(f"\nPortfolio Metrics:\n{portfolio}")
