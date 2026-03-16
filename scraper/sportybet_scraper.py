#!/usr/bin/env python3
"""
SPORTYBET REAL-TIME ODDS SCRAPER
Scrapes Over 1.5 odds from sportybet.com and feeds directly into EnsemblePredictor
"""

import sys
import json
import time
import logging
from datetime import datetime
from collections import defaultdict
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, 'src')
from prediction.ensemble_predictor import EnsemblePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)  

class SportybetScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        }
        self.base_url = 'https://www.sportybet.com/'
        self.predictor = EnsemblePredictor()
        self.matches_cache = defaultdict(dict)
    
    def scrape_matches(self):
        """
        Scrape all available football matches with Over 1.5 odds from sportybet
        Returns list of matches in format compatible with EnsemblePredictor
        """
        try:
            response = requests.get(self.base_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            matches = []
            match_elements = soup.find_all('div', class_='match-item')
            
            if not match_elements:
                logger.warning("No matches found. Selectors may need updating.")
                return []
            
            for match in match_elements:
                try:
                    teams_elem = match.find('div', class_='match-teams')
                    if not teams_elem:
                        continue
                    
                    team_links = teams_elem.find_all('a')
                    if len(team_links) < 2:
                        continue
                    
                    home_team = team_links[0].text.strip()
                    away_team = team_links[1].text.strip()
                    
                    odds_elem = match.find('div', class_='odds-over-1-5')
                    if not odds_elem:
                        continue
                    
                    over_1_5_odds = float(odds_elem.text.strip())
                    
                    odds_container = match.find('div', class_='odds-container')
                    odds_values = odds_container.find_all('span', class_='odd-value') if odds_container else []
                    
                    if len(odds_values) >= 3:
                        home_odds = float(odds_values[0].text.strip())
                        draw_odds = float(odds_values[1].text.strip())
                        away_odds = float(odds_values[2].text.strip())
                    else:
                        home_odds = 1.8
                        draw_odds = 3.5
                        away_odds = 4.0
                    
                    league = self._extract_league(match)
                    
                    match_data = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'odds': {
                            'home': home_odds,
                            'draw': draw_odds,
                            'away': away_odds,
                            'over_1_5': over_1_5_odds
                        },
                        'league': league,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    matches.append(match_data)
                    logger.info(f"✓ Scraped: {home_team} vs {away_team} [{league}]")
                    
                except Exception as e:
                    logger.debug(f"Error parsing match element: {e}")
                    continue
            
            return matches
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch sportybet.com: {e}")
            return []
    
    def _extract_league(self, match_elem):
        """Extract league name from match element"""
        league_elem = match_elem.find('span', class_='league-name')
        if league_elem:
            return league_elem.text.strip()
        return "UNKNOWN"
    
    def format_for_predictor(self, match_data):
        """Convert scraped data to format compatible with EnsemblePredictor"""
        odds = match_data['odds']
        formatted = f"{match_data['home_team']} vs {match_data['away_team']} ({odds['home']} {odds['draw']} {odds['away']})"
        return formatted
    
    def run_realtime_monitor(self, interval=30, verbose=False):
        """
        Run continuous real-time monitoring
        interval: seconds between scrapes (default 30)
        """
        print("="*90)
        print("🚀 SPORTYBET REAL-TIME ODDS SCRAPER")
        print("="*90)
        print(f"Monitoring: sportybet.com for Over 1.5 odds")
        print(f"Interval: {interval} seconds")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        iteration = 0
        try:
            while True:
                iteration += 1
                print(f"\n[{iteration}] Scraping at {datetime.now().strftime('%H:%M:%S')}...")
                
                matches = self.scrape_matches()
                
                if matches:
                    print(f"✓ Found {len(matches)} matches\n")
                    
                    by_league = defaultdict(list)
                    for match in matches:
                        by_league[match['league']].append(match)
                    
                    all_results = []
                    for league, league_matches in by_league.items():
                        print(f"\n📊 {league.upper()} - {len(league_matches)} matches")
                        print("─" * 90)
                        
                        formatted_matches = [self.format_for_predictor(m) for m in league_matches]
                        
                        try:
                            results = self.predictor.analyze_matches(formatted_matches, league)
                            all_results.extend(results)
                            
                            high_conf = [r for r in results if r['over_1_5']['confidence'] == 'High']
                            if high_conf:
                                print(f"\n⭐ HIGH CONFIDENCE OVER 1.5 ({len(high_conf)} picks):")
                                for r in high_conf:
                                    print(f"   • {r['home_team']} vs {r['away_team']}: {r['over_1_5']['probability']}%")
                        except Exception as e:
                            logger.error(f"Error analyzing {league} matches: {e}")
                    
                    if verbose:
                        print(f"\n✓ Processed {len(all_results)} total predictions")
                else:
                    print("⚠ No matches found in this cycle")
                
                print(f"\n⏳ Next scrape in {interval}s...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n\n✅ Scraper stopped at {datetime.now().strftime('%H:%M:%S')}")
            sys.exit(0)
    
    def export_matches(self, output_file='data/scraped_matches.json'):
        """Export scraped matches to JSON file"""  
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        matches = self.scrape_matches()
        with open(output_file, 'w') as f:
            json.dump(matches, f, indent=2)
        
        print(f"✓ Exported {len(matches)} matches to {output_file}")
        return output_file

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sportybet Real-Time Odds Scraper')
    parser.add_argument('--mode', choices=['realtime', 'export'], default='realtime',
                       help='Operation mode')
    parser.add_argument('--interval', type=int, default=30,
                       help='Scraping interval in seconds (for realtime mode)')
    parser.add_argument('--output', type=str, default='data/scraped_matches.json',
                       help='Output file path (for export mode)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    scraper = SportybetScraper()
    
    if args.mode == 'realtime':
        scraper.run_realtime_monitor(interval=args.interval, verbose=args.verbose)
    elif args.mode == 'export':
        scraper.export_matches(args.output)