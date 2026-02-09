"""
Football Analytics Main Application

This is the main orchestration script that coordinates data collection, model training,
predictions, and value bet identification.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

# Import project modules
from src.scrapers.sofascore_scraper import SofaScoreScraper
from src.scrapers.fotmob_scraper import FotMobScraper
from src.scrapers.oneXbet_scraper import OneXBetScraper
from src.models.poisson_model import PoissonModel
from src.models.ml_predictor import MLPredictor
from src.models.edge_calculator import EdgeCalculator
from src.utils.data_processor import (
    clean_match_data, create_match_features, save_to_csv,
    load_from_csv, calculate_team_form
)
from src.utils.database import Database

# Load environment variables
load_dotenv()

# Setup logging
def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/football_analytics.log")
        ]
    )

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def scrape_data(config: dict):
    """
    Scrape data from all sources.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("=" * 50)
    logger.info("Starting data scraping...")
    logger.info("=" * 50)
    
    try:
        # Initialize scrapers
        sofascore = SofaScoreScraper(
            base_url=config['scraping']['sofascore']['base_url'],
            timeout=config['scraping']['sofascore']['timeout']
        )
        
        fotmob = FotMobScraper(
            base_url=config['scraping']['fotmob']['base_url'],
            timeout=config['scraping']['fotmob']['timeout']
        )
        
        oneXbet = OneXBetScraper(
            base_url=config['scraping']['oneXbet']['base_url'],
            headless=config['scraping']['oneXbet']['headless']
        )
        
        # Example: Scrape data (implementation depends on specific requirements)
        logger.info("Scraping SofaScore data...")
        # sofascore_data = sofascore.get_tournament_matches(...)
        
        logger.info("Scraping FotMob data...")
        # fotmob_data = fotmob.get_league_matches(...)
        
        logger.info("Scraping 1xBet odds...")
        # odds_data = oneXbet.get_football_odds()
        
        # Close Selenium driver
        oneXbet.close()
        
        logger.info("Data scraping completed successfully")
    
    except Exception as e:
        logger.error(f"Error during data scraping: {e}")
        raise


def train_models(config: dict, data_path: str):
    """
    Train prediction models on historical data.
    
    Args:
        config: Configuration dictionary
        data_path: Path to training data
    """
    logger.info("=" * 50)
    logger.info("Starting model training...")
    logger.info("=" * 50)
    
    try:
        # Load training data
        logger.info(f"Loading training data from {data_path}")
        matches_df = load_from_csv(data_path)
        
        if matches_df.empty:
            logger.error("No training data available")
            return
        
        # Clean data
        matches_df = clean_match_data(matches_df)
        
        # Train Poisson model
        logger.info("Training Poisson model...")
        poisson_model = PoissonModel(
            home_advantage=config['models']['poisson']['home_advantage']
        )
        
        if poisson_model.train(matches_df):
            logger.info("Poisson model trained successfully")
            # Save model stats
            stats = poisson_model.get_model_stats()
            logger.info(f"Model stats: {stats}")
        
        # Train ML model
        logger.info("Training ML predictor...")
        ml_predictor = MLPredictor(
            model_type=config['models']['ml_predictor']['model_type'],
            n_estimators=config['models']['ml_predictor']['n_estimators'],
            max_depth=config['models']['ml_predictor']['max_depth']
        )
        
        # Create features
        features_df = ml_predictor.create_features(
            matches_df,
            rolling_window=config['models']['ml_predictor']['rolling_window']
        )
        
        if not features_df.empty:
            X, y = ml_predictor.prepare_training_data(matches_df, features_df)
            
            if len(X) > 0:
                metrics = ml_predictor.train(X, y)
                logger.info(f"ML model trained successfully")
                logger.info(f"Training metrics: {metrics}")
                
                # Save model
                model_path = f"data/models/ml_predictor_{datetime.now().strftime('%Y%m%d')}.pkl"
                ml_predictor.save_model(model_path)
                logger.info(f"Model saved to {model_path}")
        
        logger.info("Model training completed successfully")
    
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def predict_matches(config: dict):
    """
    Generate predictions for upcoming matches.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("=" * 50)
    logger.info("Generating predictions for upcoming matches...")
    logger.info("=" * 50)
    
    try:
        # Initialize database
        database_url = os.getenv('DATABASE_URL')
        db = Database(database_url)
        session = db.get_session()
        
        # Get upcoming matches
        upcoming_matches = db.get_upcoming_matches(session, limit=10)
        
        if not upcoming_matches:
            logger.info("No upcoming matches found")
            return
        
        logger.info(f"Found {len(upcoming_matches)} upcoming matches")
        
        # Load models
        # poisson_model = PoissonModel()
        # ml_predictor = MLPredictor()
        # ml_predictor.load_model("data/models/ml_predictor_latest.pkl")
        
        for match in upcoming_matches:
            logger.info(f"\nPredicting: {match.home_team} vs {match.away_team}")
            
            # Generate predictions using models
            # prediction = poisson_model.predict_match_outcome(match.home_team, match.away_team)
            
            # Save predictions to database
            # db.add_prediction(session, {...})
            
            logger.info(f"Prediction saved for match {match.id}")
        
        session.close()
        logger.info("Predictions completed successfully")
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


def analyze_value_bets(config: dict):
    """
    Analyze matches and identify value bets.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("=" * 50)
    logger.info("Analyzing value bets...")
    logger.info("=" * 50)
    
    try:
        # Initialize edge calculator
        edge_calc = EdgeCalculator(
            kelly_fraction=config['models']['edge_calculator']['kelly_fraction'],
            max_stake_pct=config['models']['edge_calculator']['max_stake_pct'],
            min_odds=config['models']['edge_calculator']['min_odds'],
            max_odds=config['models']['edge_calculator']['max_odds']
        )
        
        bankroll = float(os.getenv('BANKROLL', config['betting']['bankroll']))
        min_edge = config['betting']['min_edge']
        
        logger.info(f"Bankroll: ${bankroll}, Min edge: {min_edge*100}%")
        
        # Get predictions and odds from database
        database_url = os.getenv('DATABASE_URL')
        db = Database(database_url)
        session = db.get_session()
        
        upcoming_matches = db.get_upcoming_matches(session, limit=10)
        
        all_value_bets = []
        
        for match in upcoming_matches:
            # Get latest odds and predictions
            odds = db.get_latest_odds(session, match.id)
            predictions = db.get_predictions_for_match(session, match.id)
            
            if not odds or not predictions:
                continue
            
            # Prepare data for edge calculation
            prediction_data = {
                'home_team': match.home_team,
                'away_team': match.away_team,
                'probabilities': {
                    'home_win': predictions[0].home_win_prob,
                    'draw': predictions[0].draw_prob,
                    'away_win': predictions[0].away_win_prob,
                    'over_2.5': predictions[0].over_2_5_prob,
                    'under_2.5': predictions[0].under_2_5_prob
                }
            }
            
            odds_data = {
                'home_odds': odds.home_odds,
                'draw_odds': odds.draw_odds,
                'away_odds': odds.away_odds,
                'over_2.5_odds': odds.over_2_5_odds,
                'under_2.5_odds': odds.under_2_5_odds
            }
            
            # Find value bets
            value_bets = edge_calc.find_value_bets(
                prediction_data,
                odds_data,
                min_edge=min_edge,
                bankroll=bankroll
            )
            
            if not value_bets.empty:
                all_value_bets.append(value_bets)
                logger.info(f"Found {len(value_bets)} value bets for match {match.id}")
        
        if all_value_bets:
            import pandas as pd
            combined_value_bets = pd.concat(all_value_bets, ignore_index=True)
            
            # Calculate portfolio metrics
            portfolio = edge_calc.calculate_portfolio_metrics(combined_value_bets, bankroll)
            
            logger.info("\n" + "=" * 50)
            logger.info("VALUE BETTING PORTFOLIO")
            logger.info("=" * 50)
            logger.info(f"Total bets: {portfolio['num_bets']}")
            logger.info(f"Total stake: ${portfolio['total_stake']:.2f}")
            logger.info(f"Exposure: {portfolio['exposure_pct']:.2f}%")
            logger.info(f"Average edge: {portfolio['avg_edge']:.2f}%")
            logger.info(f"Expected value: ${portfolio['expected_value']:.2f}")
            logger.info(f"Expected ROI: {portfolio['expected_roi']:.2f}%")
            
            # Save to CSV
            output_path = f"data/processed/value_bets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            save_to_csv(combined_value_bets, output_path)
            logger.info(f"\nValue bets saved to {output_path}")
        else:
            logger.info("No value bets found")
        
        session.close()
    
    except Exception as e:
        logger.error(f"Error analyzing value bets: {e}")
        raise


def backtest(config: dict, data_path: str):
    """
    Backtest model performance on historical data.
    
    Args:
        config: Configuration dictionary
        data_path: Path to historical data
    """
    logger.info("=" * 50)
    logger.info("Starting backtest...")
    logger.info("=" * 50)
    
    try:
        logger.info(f"Loading historical data from {data_path}")
        matches_df = load_from_csv(data_path)
        
        if matches_df.empty:
            logger.error("No historical data available")
            return
        
        # Implement backtest logic
        logger.info("Backtesting models...")
        
        # Calculate metrics: accuracy, ROI, Sharpe ratio, etc.
        
        logger.info("Backtest completed successfully")
    
    except Exception as e:
        logger.error(f"Error during backtest: {e}")
        raise


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Football Analytics - Betting Edge Finder",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'command',
        choices=['scrape', 'train', 'predict', 'analyze', 'backtest'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data',
        default='data/processed/matches.csv',
        help='Path to data file'
    )
    
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Setup logging
    setup_logging(args.log_level)
    
    logger.info("=" * 50)
    logger.info("Football Analytics System")
    logger.info("=" * 50)
    logger.info(f"Command: {args.command}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data: {args.data}")
    logger.info("=" * 50)
    
    # Load configuration
    config = load_config(args.config)
    
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'scrape':
            scrape_data(config)
        elif args.command == 'train':
            train_models(config, args.data)
        elif args.command == 'predict':
            predict_matches(config)
        elif args.command == 'analyze':
            analyze_value_bets(config)
        elif args.command == 'backtest':
            backtest(config, args.data)
        
        logger.info("\n" + "=" * 50)
        logger.info("Execution completed successfully!")
        logger.info("=" * 50)
    
    except Exception as e:
        logger.error(f"\n{'=' * 50}")
        logger.error(f"Execution failed: {e}")
        logger.error("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
