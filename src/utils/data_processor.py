"""
Data Processing Utilities

This module provides utilities for cleaning, normalizing, and processing scraped data.
Includes functions for data validation, feature engineering, and data export/import.
"""

import pandas as pd
import numpy as np
import json
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import os

logger = logging.getLogger(__name__)


def clean_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize match data.
    
    Args:
        df: DataFrame with raw match data
    
    Returns:
        Cleaned DataFrame
    """
    try:
        logger.info(f"Cleaning match data with {len(df)} rows")
        
        df_clean = df.copy()
        
        # Remove duplicates
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if len(df_clean) < initial_len:
            logger.info(f"Removed {initial_len - len(df_clean)} duplicate rows")
        
        # Handle missing values in goals
        if 'home_goals' in df_clean.columns:
            df_clean['home_goals'] = df_clean['home_goals'].fillna(0).astype(int)
        if 'away_goals' in df_clean.columns:
            df_clean['away_goals'] = df_clean['away_goals'].fillna(0).astype(int)
        
        # Handle missing values in xG
        if 'home_xg' in df_clean.columns:
            df_clean['home_xg'] = pd.to_numeric(df_clean['home_xg'], errors='coerce')
        if 'away_xg' in df_clean.columns:
            df_clean['away_xg'] = pd.to_numeric(df_clean['away_xg'], errors='coerce')
        
        # Clean team names (strip whitespace, standardize)
        if 'home_team' in df_clean.columns:
            df_clean['home_team'] = df_clean['home_team'].str.strip()
        if 'away_team' in df_clean.columns:
            df_clean['away_team'] = df_clean['away_team'].str.strip()
        
        # Convert date column if present
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        
        logger.info(f"Cleaned data: {len(df_clean)} rows remaining")
        return df_clean
    
    except Exception as e:
        logger.error(f"Error cleaning match data: {e}")
        return df


def merge_data_sources(sofascore_df: pd.DataFrame, fotmob_df: pd.DataFrame,
                      merge_on: List[str] = ['home_team', 'away_team', 'date']) -> pd.DataFrame:
    """
    Merge data from multiple sources.
    
    Args:
        sofascore_df: DataFrame from SofaScore
        fotmob_df: DataFrame from FotMob
        merge_on: Columns to merge on
    
    Returns:
        Merged DataFrame
    """
    try:
        logger.info("Merging data from multiple sources")
        
        # Clean both dataframes
        sofascore_clean = clean_match_data(sofascore_df)
        fotmob_clean = clean_match_data(fotmob_df)
        
        # Merge dataframes
        merged_df = pd.merge(
            sofascore_clean,
            fotmob_clean,
            on=merge_on,
            how='outer',
            suffixes=('_sofascore', '_fotmob')
        )
        
        logger.info(f"Merged data: {len(merged_df)} rows")
        return merged_df
    
    except Exception as e:
        logger.error(f"Error merging data sources: {e}")
        return pd.DataFrame()


def validate_data(df: pd.DataFrame, required_columns: List[str],
                 numeric_columns: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """
    Validate data structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        numeric_columns: List of columns that should be numeric
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    try:
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        errors.append(f"Column '{col}' should be numeric")
        
        # Check for empty dataframe
        if len(df) == 0:
            errors.append("DataFrame is empty")
        
        # Check for all-null columns
        null_cols = [col for col in df.columns if df[col].isna().all()]
        if null_cols:
            errors.append(f"Columns with all null values: {null_cols}")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation failed: {errors}")
        
        return is_valid, errors
    
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return False, [str(e)]


def create_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features for match data.
    
    Args:
        df: DataFrame with match data
    
    Returns:
        DataFrame with additional features
    """
    try:
        logger.info("Creating match features")
        
        df_features = df.copy()
        
        # Total goals
        if 'home_goals' in df.columns and 'away_goals' in df.columns:
            df_features['total_goals'] = df['home_goals'] + df['away_goals']
            df_features['goal_difference'] = df['home_goals'] - df['away_goals']
            
            # Match result (0: home win, 1: draw, 2: away win)
            df_features['result'] = df.apply(
                lambda row: 0 if row['home_goals'] > row['away_goals'] 
                           else (1 if row['home_goals'] == row['away_goals'] else 2),
                axis=1
            )
            
            # Over/Under 2.5
            df_features['over_2.5'] = (df_features['total_goals'] > 2.5).astype(int)
            
            # BTTS (Both Teams To Score)
            df_features['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
        
        # xG features
        if 'home_xg' in df.columns and 'away_xg' in df.columns:
            df_features['total_xg'] = df['home_xg'] + df['away_xg']
            df_features['xg_difference'] = df['home_xg'] - df['away_xg']
            
            # xG overperformance (goals - xG)
            if 'home_goals' in df.columns:
                df_features['home_xg_diff'] = df['home_goals'] - df['home_xg']
            if 'away_goals' in df.columns:
                df_features['away_xg_diff'] = df['away_goals'] - df['away_xg']
        
        logger.info(f"Created {len(df_features.columns) - len(df.columns)} new features")
        return df_features
    
    except Exception as e:
        logger.error(f"Error creating match features: {e}")
        return df


def calculate_team_form(df: pd.DataFrame, team: str, num_matches: int = 5) -> Dict:
    """
    Calculate recent form for a team.
    
    Args:
        df: DataFrame with match data
        team: Team name
        num_matches: Number of recent matches to consider
    
    Returns:
        Dictionary with form statistics
    """
    try:
        # Get matches involving the team
        team_matches = df[
            (df['home_team'] == team) | (df['away_team'] == team)
        ].copy()
        
        if len(team_matches) == 0:
            logger.warning(f"No matches found for team: {team}")
            return {}
        
        # Sort by date if available
        if 'date' in team_matches.columns:
            team_matches = team_matches.sort_values('date', ascending=False)
        
        # Get recent matches
        recent_matches = team_matches.head(num_matches)
        
        # Calculate form statistics
        wins = 0
        draws = 0
        losses = 0
        goals_scored = 0
        goals_conceded = 0
        
        for _, match in recent_matches.iterrows():
            is_home = match['home_team'] == team
            
            if is_home:
                goals_scored += match['home_goals']
                goals_conceded += match['away_goals']
                
                if match['home_goals'] > match['away_goals']:
                    wins += 1
                elif match['home_goals'] == match['away_goals']:
                    draws += 1
                else:
                    losses += 1
            else:
                goals_scored += match['away_goals']
                goals_conceded += match['home_goals']
                
                if match['away_goals'] > match['home_goals']:
                    wins += 1
                elif match['away_goals'] == match['home_goals']:
                    draws += 1
                else:
                    losses += 1
        
        # Calculate points (3 for win, 1 for draw)
        points = wins * 3 + draws
        
        form = {
            'team': team,
            'matches_played': len(recent_matches),
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'points': points,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'goal_difference': goals_scored - goals_conceded,
            'avg_goals_scored': round(goals_scored / len(recent_matches), 2),
            'avg_goals_conceded': round(goals_conceded / len(recent_matches), 2),
            'points_per_game': round(points / len(recent_matches), 2)
        }
        
        logger.debug(f"Form calculated for {team}: {points} points in {len(recent_matches)} matches")
        return form
    
    except Exception as e:
        logger.error(f"Error calculating team form: {e}")
        return {}


def save_to_csv(df: pd.DataFrame, filepath: str):
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Path to save the file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")


def load_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded from {filepath}: {len(df)} rows")
        return df
    
    except Exception as e:
        logger.error(f"Error loading from CSV: {e}")
        return pd.DataFrame()


def save_to_json(data: Any, filepath: str):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save (dict, list, etc.)
        filepath: Path to save the file
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Data saved to {filepath}")
    
    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")


def load_from_json(filepath: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        Loaded data
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Data loaded from {filepath}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading from JSON: {e}")
        return None


def save_to_pickle(data: Any, filepath: str):
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save the file
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Data saved to {filepath}")
    
    except Exception as e:
        logger.error(f"Error saving to pickle: {e}")


def load_from_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Path to the pickle file
    
    Returns:
        Loaded data
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Data loaded from {filepath}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading from pickle: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Sample data
    sample_data = pd.DataFrame({
        'home_team': ['Team A', 'Team B', 'Team C'],
        'away_team': ['Team B', 'Team C', 'Team A'],
        'home_goals': [2, 1, 3],
        'away_goals': [1, 1, 2],
        'home_xg': [1.8, 1.2, 2.5],
        'away_xg': [1.1, 1.0, 1.9],
        'date': pd.date_range('2023-01-01', periods=3)
    })
    
    # Clean data
    cleaned = clean_match_data(sample_data)
    print("Cleaned data:\n", cleaned)
    
    # Create features
    featured = create_match_features(cleaned)
    print("\nWith features:\n", featured)
    
    # Calculate form
    form = calculate_team_form(featured, 'Team A', num_matches=3)
    print(f"\nTeam A form: {form}")
