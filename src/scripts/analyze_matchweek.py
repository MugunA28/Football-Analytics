#!/usr/bin/env python3
"""
Matchweek Analysis Script

Command-line script to analyze English Premier League matchweeks and generate
probability rankings for goal scorers, assist providers, and clean sheets.

Usage:
    python src/scripts/analyze_matchweek.py --matchweek 26 --league "Premier League" --output console
    python src/scripts/analyze_matchweek.py --matchweek 26 --output json --file results.json
    python src/scripts/analyze_matchweek.py --matchweek 26 --output csv --file results.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.player_analyzer import PlayerAnalyzer
from src.scrapers.fotmob_scraper import FotMobScraper
from src.scrapers.sofascore_scraper import SofaScoreScraper


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze Premier League matchweek player probabilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Console output
  python src/scripts/analyze_matchweek.py --matchweek 26 --output console
  
  # JSON output to file
  python src/scripts/analyze_matchweek.py --matchweek 26 --output json --file data/processed/matchweek_26.json
  
  # CSV output to file
  python src/scripts/analyze_matchweek.py --matchweek 26 --output csv --file data/processed/matchweek_26.csv
  
  # Verbose output
  python src/scripts/analyze_matchweek.py --matchweek 26 --output console --verbose
        """
    )
    
    parser.add_argument(
        '--matchweek',
        type=int,
        required=True,
        help='Matchweek number to analyze'
    )
    
    parser.add_argument(
        '--league',
        type=str,
        default='Premier League',
        help='League name (default: Premier League)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        choices=['console', 'json', 'csv'],
        default='console',
        help='Output format (default: console)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Output file path (required for json/csv output)'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top players to display (default: 20)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (YAML)'
    )
    
    return parser.parse_args()


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from file."""
    if not config_path:
        return {}
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        return {}


def save_json_output(report: Dict, filepath: str):
    """Save report as JSON file."""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logging.info(f"JSON output saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving JSON output: {e}")
        raise


def save_csv_output(report: Dict, filepath: str):
    """Save report as CSV file."""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Create separate CSV files for each category
        base_path = Path(filepath)
        base_name = base_path.stem
        base_dir = base_path.parent
        
        # Save goal scorers
        if report['top_goal_scorers']:
            goal_scorers_df = pd.DataFrame(report['top_goal_scorers'])
            goal_scorers_path = base_dir / f"{base_name}_goal_scorers.csv"
            goal_scorers_df.to_csv(goal_scorers_path, index=False)
            logging.info(f"Goal scorers CSV saved to {goal_scorers_path}")
        
        # Save assist providers
        if report['top_assist_providers']:
            assist_providers_df = pd.DataFrame(report['top_assist_providers'])
            assist_providers_path = base_dir / f"{base_name}_assist_providers.csv"
            assist_providers_df.to_csv(assist_providers_path, index=False)
            logging.info(f"Assist providers CSV saved to {assist_providers_path}")
        
        # Save clean sheet candidates
        if report['clean_sheet_candidates']:
            clean_sheets_df = pd.DataFrame(report['clean_sheet_candidates'])
            clean_sheets_path = base_dir / f"{base_name}_clean_sheets.csv"
            clean_sheets_df.to_csv(clean_sheets_path, index=False)
            logging.info(f"Clean sheets CSV saved to {clean_sheets_path}")
        
        # Save fixtures
        if report['fixtures']:
            fixtures_df = pd.DataFrame(report['fixtures'])
            fixtures_path = base_dir / f"{base_name}_fixtures.csv"
            fixtures_df.to_csv(fixtures_path, index=False)
            logging.info(f"Fixtures CSV saved to {fixtures_path}")
        
        logging.info(f"CSV output saved to {base_dir / base_name}_*.csv")
    
    except Exception as e:
        logging.error(f"Error saving CSV output: {e}")
        raise


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting matchweek {args.matchweek} analysis for {args.league}")
    
    # Validate output file argument
    if args.output in ['json', 'csv'] and not args.file:
        logger.error(f"--file argument is required for {args.output} output")
        sys.exit(1)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize scrapers
        logger.info("Initializing scrapers...")
        fotmob = FotMobScraper()
        sofascore = SofaScoreScraper()
        
        # Initialize analyzer
        logger.info("Initializing player analyzer...")
        analyzer = PlayerAnalyzer(
            fotmob_scraper=fotmob,
            sofascore_scraper=sofascore,
            config=config.get('analysis', {})
        )
        
        # Generate analysis report
        logger.info(f"Generating analysis for matchweek {args.matchweek}...")
        report = analyzer.generate_analysis_report(
            matchweek=args.matchweek,
            top_n=args.top_n
        )
        
        # Check for errors
        if 'error' in report:
            logger.error(f"Analysis error: {report['error']}")
            sys.exit(1)
        
        # Output results
        if args.output == 'console':
            console_output = analyzer.format_console_output(report, output_top_n=args.top_n)
            print(console_output)
        
        elif args.output == 'json':
            save_json_output(report, args.file)
            print(f"Analysis results saved to {args.file}")
        
        elif args.output == 'csv':
            save_csv_output(report, args.file)
            print(f"Analysis results saved to CSV files")
        
        logger.info("Analysis completed successfully")
        
        # Also save to default location if console output
        if args.output == 'console':
            default_path = Path('data/processed') / f'matchweek_{args.matchweek}_analysis.json'
            try:
                save_json_output(report, str(default_path))
                logger.info(f"Report also saved to {default_path}")
            except Exception as e:
                logger.warning(f"Could not save to default location: {e}")
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
