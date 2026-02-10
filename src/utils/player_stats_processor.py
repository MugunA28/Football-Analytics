"""
Player Statistics Processor

This module provides utility functions for processing player statistics,
calculating probabilities, and applying various adjustments for player analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


def calculate_rolling_average(data: List[float], window: int = 5) -> float:
    """
    Calculate rolling average for a given data series.
    
    Args:
        data: List of numerical values
        window: Rolling window size (default: 5)
    
    Returns:
        Rolling average value, or 0.0 if insufficient data
    
    Examples:
        >>> calculate_rolling_average([1, 2, 3, 4, 5], window=3)
        4.0
        >>> calculate_rolling_average([0.5, 1.0, 1.5], window=5)
        1.0
    """
    if not data:
        logger.warning("Empty data provided for rolling average calculation")
        return 0.0
    
    if len(data) == 0:
        return 0.0
    
    # Use only the most recent 'window' values
    recent_data = data[-window:] if len(data) >= window else data
    
    try:
        return float(np.mean(recent_data))
    except Exception as e:
        logger.error(f"Error calculating rolling average: {e}")
        return 0.0


def normalize_probability(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to a probability between 0 and 1.
    
    Args:
        value: Value to normalize
        min_val: Minimum value in the range
        max_val: Maximum value in the range
    
    Returns:
        Normalized probability between 0 and 1
    
    Examples:
        >>> normalize_probability(5, 0, 10)
        0.5
        >>> normalize_probability(7.5, 5, 10)
        0.5
    """
    if max_val == min_val:
        logger.warning(f"Max and min values are equal: {max_val}")
        return 0.5  # Return middle value if range is zero
    
    try:
        normalized = (value - min_val) / (max_val - min_val)
        # Clamp between 0 and 1
        return max(0.0, min(1.0, normalized))
    except Exception as e:
        logger.error(f"Error normalizing probability: {e}")
        return 0.0


def apply_home_advantage_factor(probability: float, is_home: bool, 
                                 home_boost: float = 0.1) -> float:
    """
    Apply home advantage factor to a probability.
    
    Args:
        probability: Base probability (0-1)
        is_home: Whether the player's team is playing at home
        home_boost: Boost factor for home games (default: 0.1 or 10%)
    
    Returns:
        Adjusted probability
    
    Examples:
        >>> apply_home_advantage_factor(0.5, True)
        0.55
        >>> apply_home_advantage_factor(0.5, False)
        0.45
    """
    try:
        if is_home:
            adjusted = probability * (1 + home_boost)
        else:
            adjusted = probability * (1 - home_boost * 0.5)  # Less penalty for away
        
        # Ensure probability stays between 0 and 1
        return max(0.0, min(1.0, adjusted))
    except Exception as e:
        logger.error(f"Error applying home advantage factor: {e}")
        return probability


def adjust_for_opponent_strength(probability: float, opponent_rating: float,
                                  league_avg_rating: float = 70.0) -> float:
    """
    Adjust probability based on opponent strength.
    
    Args:
        probability: Base probability (0-1)
        opponent_rating: Opponent's strength rating (0-100)
        league_avg_rating: Average rating in the league (default: 70.0)
    
    Returns:
        Adjusted probability
    
    Examples:
        >>> adjust_for_opponent_strength(0.5, 50, 70)  # Weak opponent
        0.6
        >>> adjust_for_opponent_strength(0.5, 90, 70)  # Strong opponent
        0.4
    """
    try:
        # Calculate adjustment factor based on opponent strength
        # Negative for weak opponents (easier), positive for strong opponents (harder)
        strength_diff = (opponent_rating - league_avg_rating) / league_avg_rating
        
        # Apply adjustment (max ±20% change)
        adjustment_factor = 1 - (strength_diff * 0.2)
        adjusted = probability * adjustment_factor
        
        # Ensure probability stays between 0 and 1
        return max(0.0, min(1.0, adjusted))
    except Exception as e:
        logger.error(f"Error adjusting for opponent strength: {e}")
        return probability


def calculate_form_score(recent_matches: List[Dict], 
                         metric: str = 'goals',
                         weight_recent: bool = True) -> float:
    """
    Calculate a form score from recent match performances.
    
    Args:
        recent_matches: List of match dictionaries with performance metrics
        metric: Metric to evaluate (e.g., 'goals', 'assists', 'clean_sheets')
        weight_recent: Whether to weight recent matches more heavily
    
    Returns:
        Form score (0-10 scale)
    
    Examples:
        >>> matches = [{'goals': 2}, {'goals': 1}, {'goals': 0}]
        >>> calculate_form_score(matches, metric='goals')
        6.0
    """
    if not recent_matches:
        logger.warning("No recent matches provided for form calculation")
        return 0.0
    
    try:
        scores = []
        weights = []
        
        for i, match in enumerate(recent_matches):
            if metric in match:
                value = match[metric]
                scores.append(value)
                
                if weight_recent:
                    # More recent matches get higher weight
                    weight = (i + 1) / len(recent_matches)
                    weights.append(weight)
                else:
                    weights.append(1.0)
        
        if not scores:
            return 0.0
        
        # Calculate weighted average
        weighted_avg = np.average(scores, weights=weights)
        
        # Scale to 0-10 range (assuming max value of 3 per match for most metrics)
        max_expected = 3.0
        form_score = min(10.0, (weighted_avg / max_expected) * 10.0)
        
        return float(form_score)
    except Exception as e:
        logger.error(f"Error calculating form score: {e}")
        return 0.0


def calculate_weighted_probability(metrics: Dict[str, float], 
                                   weights: Dict[str, float]) -> float:
    """
    Calculate a weighted probability from multiple metrics.
    
    Args:
        metrics: Dictionary of metric names to values (0-1 scale)
        weights: Dictionary of metric names to weights (should sum to 1.0)
    
    Returns:
        Weighted probability (0-1)
    
    Examples:
        >>> metrics = {'xg': 0.8, 'form': 0.6, 'conversion': 0.7}
        >>> weights = {'xg': 0.5, 'form': 0.3, 'conversion': 0.2}
        >>> calculate_weighted_probability(metrics, weights)
        0.73
    """
    try:
        # Validate that weights sum to approximately 1.0
        weight_sum = sum(weights.values())
        if not (0.99 <= weight_sum <= 1.01):
            logger.warning(f"Weights do not sum to 1.0: {weight_sum}")
            # Normalize weights
            weights = {k: v / weight_sum for k, v in weights.items()}
        
        # Calculate weighted sum
        weighted_prob = 0.0
        for metric, value in metrics.items():
            if metric in weights:
                weighted_prob += value * weights[metric]
            else:
                logger.warning(f"No weight defined for metric: {metric}")
        
        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, weighted_prob))
    except Exception as e:
        logger.error(f"Error calculating weighted probability: {e}")
        return 0.0


def calculate_minutes_adjustment(minutes_played: int, 
                                 total_minutes: int = 450,
                                 min_threshold: int = 180) -> float:
    """
    Calculate adjustment factor based on minutes played.
    Players with fewer minutes get reduced probability.
    
    Args:
        minutes_played: Total minutes played in analysis period
        total_minutes: Total possible minutes (default: 450 for 5 matches)
        min_threshold: Minimum minutes for full credit (default: 180)
    
    Returns:
        Adjustment factor (0-1)
    
    Examples:
        >>> calculate_minutes_adjustment(450, 450)
        1.0
        >>> calculate_minutes_adjustment(90, 450, 180)
        0.5
    """
    try:
        if minutes_played >= min_threshold:
            # Full credit if above threshold
            adjustment = min(1.0, minutes_played / total_minutes)
        else:
            # Reduced credit below threshold
            adjustment = minutes_played / min_threshold * 0.5
        
        return max(0.0, min(1.0, adjustment))
    except Exception as e:
        logger.error(f"Error calculating minutes adjustment: {e}")
        return 0.5


def exponential_moving_average(data: List[float], alpha: float = 0.3) -> float:
    """
    Calculate exponential moving average (EMA) for time series data.
    More recent values have higher weight.
    
    Args:
        data: List of numerical values (oldest to newest)
        alpha: Smoothing factor (0-1), higher = more weight on recent values
    
    Returns:
        EMA value
    
    Examples:
        >>> exponential_moving_average([1, 2, 3, 4, 5], alpha=0.3)
        3.6
    """
    if not data:
        return 0.0
    
    try:
        ema = data[0]  # Start with first value
        
        for value in data[1:]:
            ema = alpha * value + (1 - alpha) * ema
        
        return float(ema)
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        return 0.0


def scale_to_percentage(probability: float) -> float:
    """
    Scale probability (0-1) to percentage (0-100).
    
    Args:
        probability: Probability value (0-1)
    
    Returns:
        Percentage value (0-100)
    
    Examples:
        >>> scale_to_percentage(0.754)
        75.4
    """
    return max(0.0, min(100.0, probability * 100.0))


def get_form_stars(form_score: float) -> str:
    """
    Convert form score to star rating string.
    
    Args:
        form_score: Form score (0-10)
    
    Returns:
        Star rating string (e.g., "⭐⭐⭐⭐⭐")
    
    Examples:
        >>> get_form_stars(8.0)
        '⭐⭐⭐⭐'
        >>> get_form_stars(10.0)
        '⭐⭐⭐⭐⭐'
    """
    num_stars = int(np.ceil(form_score / 2))  # 0-10 scale to 0-5 stars
    num_stars = max(0, min(5, num_stars))  # Clamp to 0-5
    return '⭐' * num_stars


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test rolling average
    data = [1.2, 0.8, 1.5, 0.9, 1.1]
    avg = calculate_rolling_average(data, window=3)
    print(f"Rolling average (window=3): {avg:.2f}")
    
    # Test probability normalization
    norm_prob = normalize_probability(7.5, 5.0, 10.0)
    print(f"Normalized probability: {norm_prob:.2f}")
    
    # Test home advantage
    adj_prob = apply_home_advantage_factor(0.5, is_home=True)
    print(f"Adjusted probability (home): {adj_prob:.2f}")
    
    # Test form score
    matches = [
        {'goals': 2, 'assists': 1},
        {'goals': 1, 'assists': 0},
        {'goals': 1, 'assists': 2},
        {'goals': 0, 'assists': 1},
    ]
    form = calculate_form_score(matches, metric='goals')
    print(f"Form score: {form:.2f} {get_form_stars(form)}")
