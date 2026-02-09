"""
Machine Learning Match Predictor

This module implements ML-based prediction models using Random Forest and Gradient Boosting
for predicting football match outcomes with feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Machine Learning predictor for football match outcomes.
    
    Uses Random Forest or Gradient Boosting with engineered features
    to predict match results and provide probability estimates.
    
    Attributes:
        model_type (str): Type of model ('random_forest' or 'gradient_boosting')
        model: Trained ML model
        scaler: StandardScaler for feature normalization
        feature_names (list): Names of features used for training
        trained (bool): Whether the model has been trained
    """
    
    def __init__(self, model_type: str = "random_forest", n_estimators: int = 100,
                 max_depth: int = 10, random_state: int = 42):
        """
        Initialize the ML predictor.
        
        Args:
            model_type: Type of model ('random_forest' or 'gradient_boosting')
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.scaler = StandardScaler()
        self.feature_names = []
        self.trained = False
        
        logger.info(f"ML Predictor initialized with {model_type}")
    
    def create_features(self, matches_df: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
        """
        Engineer features from raw match data.
        
        Args:
            matches_df: DataFrame with match data (home_team, away_team, home_goals, away_goals, etc.)
            rolling_window: Window size for rolling statistics (default 5)
        
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info(f"Creating features with rolling window of {rolling_window}")
            
            features_list = []
            
            # Get unique teams
            teams = set(matches_df['home_team'].unique()) | set(matches_df['away_team'].unique())
            
            # Calculate rolling statistics for each team
            team_stats = {}
            
            for team in teams:
                # Get all matches for this team
                home_matches = matches_df[matches_df['home_team'] == team].copy()
                away_matches = matches_df[matches_df['away_team'] == team].copy()
                
                # Combine and sort by date if available
                if 'date' in matches_df.columns:
                    home_matches['goals_scored'] = home_matches['home_goals']
                    home_matches['goals_conceded'] = home_matches['away_goals']
                    away_matches['goals_scored'] = away_matches['away_goals']
                    away_matches['goals_conceded'] = away_matches['home_goals']
                    
                    all_matches = pd.concat([
                        home_matches[['date', 'goals_scored', 'goals_conceded']],
                        away_matches[['date', 'goals_scored', 'goals_conceded']]
                    ]).sort_values('date')
                    
                    # Calculate rolling averages
                    all_matches['rolling_goals_scored'] = all_matches['goals_scored'].rolling(
                        window=rolling_window, min_periods=1
                    ).mean()
                    all_matches['rolling_goals_conceded'] = all_matches['goals_conceded'].rolling(
                        window=rolling_window, min_periods=1
                    ).mean()
                    all_matches['rolling_goal_diff'] = (
                        all_matches['rolling_goals_scored'] - all_matches['rolling_goals_conceded']
                    )
                    
                    team_stats[team] = all_matches
            
            # Create features for each match
            for idx, match in matches_df.iterrows():
                home_team = match['home_team']
                away_team = match['away_team']
                
                features = {
                    'match_id': idx
                }
                
                # Home team features
                if home_team in team_stats and len(team_stats[home_team]) > 0:
                    home_recent = team_stats[home_team].iloc[-rolling_window:]
                    features['home_avg_goals_scored'] = home_recent['rolling_goals_scored'].mean()
                    features['home_avg_goals_conceded'] = home_recent['rolling_goals_conceded'].mean()
                    features['home_goal_diff'] = home_recent['rolling_goal_diff'].mean()
                else:
                    features['home_avg_goals_scored'] = 1.5
                    features['home_avg_goals_conceded'] = 1.5
                    features['home_goal_diff'] = 0.0
                
                # Away team features
                if away_team in team_stats and len(team_stats[away_team]) > 0:
                    away_recent = team_stats[away_team].iloc[-rolling_window:]
                    features['away_avg_goals_scored'] = away_recent['rolling_goals_scored'].mean()
                    features['away_avg_goals_conceded'] = away_recent['rolling_goals_conceded'].mean()
                    features['away_goal_diff'] = away_recent['rolling_goal_diff'].mean()
                else:
                    features['away_avg_goals_scored'] = 1.5
                    features['away_avg_goals_conceded'] = 1.5
                    features['away_goal_diff'] = 0.0
                
                # Combined features
                features['attack_vs_defense'] = (
                    features['home_avg_goals_scored'] - features['away_avg_goals_conceded']
                )
                features['defense_vs_attack'] = (
                    features['home_avg_goals_conceded'] - features['away_avg_goals_scored']
                )
                features['goal_diff_difference'] = (
                    features['home_goal_diff'] - features['away_goal_diff']
                )
                
                # Include xG data if available
                if 'home_xg' in match and 'away_xg' in match:
                    features['home_xg'] = match['home_xg'] if pd.notna(match['home_xg']) else 1.5
                    features['away_xg'] = match['away_xg'] if pd.notna(match['away_xg']) else 1.5
                    features['xg_difference'] = features['home_xg'] - features['away_xg']
                
                features_list.append(features)
            
            features_df = pd.DataFrame(features_list)
            logger.info(f"Created {len(features_df.columns)} features for {len(features_df)} matches")
            
            return features_df
        
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def prepare_training_data(self, matches_df: pd.DataFrame, 
                            features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with features and labels.
        
        Args:
            matches_df: DataFrame with match results
            features_df: DataFrame with engineered features
        
        Returns:
            Tuple of (X, y) where X is features array and y is labels array
        """
        try:
            # Create target variable (0: home win, 1: draw, 2: away win)
            y = []
            for idx, match in matches_df.iterrows():
                home_goals = match['home_goals']
                away_goals = match['away_goals']
                
                if home_goals > away_goals:
                    y.append(0)  # Home win
                elif home_goals == away_goals:
                    y.append(1)  # Draw
                else:
                    y.append(2)  # Away win
            
            # Remove match_id column if present
            X = features_df.drop(['match_id'], axis=1, errors='ignore')
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            logger.info(f"Class distribution - Home wins: {y.count(0)}, Draws: {y.count(1)}, Away wins: {y.count(2)}")
            
            return X.values, np.array(y)
        
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """
        Train the ML model with cross-validation.
        
        Args:
            X: Feature array
            y: Labels array
            test_size: Proportion of data to use for testing
        
        Returns:
            Dictionary with training metrics
        """
        try:
            logger.info(f"Training {self.model_type} model")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            
            # Predictions
            y_pred = self.model.predict(X_test_scaled)
            
            self.trained = True
            
            metrics = {
                'train_accuracy': round(train_score, 4),
                'test_accuracy': round(test_score, 4),
                'cv_mean': round(cv_scores.mean(), 4),
                'cv_std': round(cv_scores.std(), 4),
                'classification_report': classification_report(y_test, y_pred, 
                                                              target_names=['Home Win', 'Draw', 'Away Win'])
            }
            
            logger.info(f"Model trained - Test accuracy: {test_score:.4f}, CV mean: {cv_scores.mean():.4f}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {}
    
    def predict_match(self, home_team_features: Dict, away_team_features: Dict) -> Dict:
        """
        Predict match outcome with probability estimates.
        
        Args:
            home_team_features: Dictionary with home team features
            away_team_features: Dictionary with away team features
        
        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.trained:
            logger.error("Model has not been trained yet")
            return {}
        
        try:
            # Combine features
            features = {
                'home_avg_goals_scored': home_team_features.get('avg_goals_scored', 1.5),
                'home_avg_goals_conceded': home_team_features.get('avg_goals_conceded', 1.5),
                'home_goal_diff': home_team_features.get('goal_diff', 0.0),
                'away_avg_goals_scored': away_team_features.get('avg_goals_scored', 1.5),
                'away_avg_goals_conceded': away_team_features.get('avg_goals_conceded', 1.5),
                'away_goal_diff': away_team_features.get('goal_diff', 0.0),
            }
            
            # Calculate combined features
            features['attack_vs_defense'] = (
                features['home_avg_goals_scored'] - features['away_avg_goals_conceded']
            )
            features['defense_vs_attack'] = (
                features['home_avg_goals_conceded'] - features['away_avg_goals_scored']
            )
            features['goal_diff_difference'] = (
                features['home_goal_diff'] - features['away_goal_diff']
            )
            
            # Include xG if available
            if 'home_xg' in home_team_features and 'away_xg' in away_team_features:
                features['home_xg'] = home_team_features['home_xg']
                features['away_xg'] = away_team_features['away_xg']
                features['xg_difference'] = features['home_xg'] - features['away_xg']
            
            # Create feature vector
            X = pd.DataFrame([features])[self.feature_names]
            X_scaled = self.scaler.transform(X.values)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            outcome_map = {0: 'home_win', 1: 'draw', 2: 'away_win'}
            
            result = {
                'prediction': outcome_map[prediction],
                'probabilities': {
                    'home_win': round(probabilities[0], 4),
                    'draw': round(probabilities[1], 4),
                    'away_win': round(probabilities[2], 4)
                }
            }
            
            logger.info(f"Prediction: {result['prediction']} with probabilities {result['probabilities']}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error predicting match: {e}")
            return {}
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importances
        """
        if not self.trained or not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model not trained or does not support feature importance")
            return pd.DataFrame()
        
        try:
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            logger.info(f"Top {top_n} important features extracted")
            return feature_importance_df
        
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.trained:
            logger.error("Cannot save untrained model")
            return
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
        
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.trained = True
            logger.info(f"Model loaded from {filepath}")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Sample data
    sample_data = pd.DataFrame({
        'home_team': ['Team A', 'Team B', 'Team C'] * 10,
        'away_team': ['Team B', 'Team C', 'Team A'] * 10,
        'home_goals': np.random.randint(0, 4, 30),
        'away_goals': np.random.randint(0, 4, 30),
        'date': pd.date_range('2023-01-01', periods=30)
    })
    
    predictor = MLPredictor(model_type="random_forest")
    features = predictor.create_features(sample_data)
    X, y = predictor.prepare_training_data(sample_data, features)
    
    if len(X) > 0:
        metrics = predictor.train(X, y)
        print(f"Training metrics: {metrics}")
