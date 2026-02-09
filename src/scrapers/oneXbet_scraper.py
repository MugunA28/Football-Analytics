"""
1xBet Odds Scraper

This module provides functionality to scrape betting odds from 1xBet using Selenium.
It extracts odds for various betting markets and calculates implied probabilities and house edge.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OneXBetScraper:
    """
    A scraper class for extracting betting odds from 1xBet using Selenium.
    
    Attributes:
        base_url (str): Base URL for 1xBet website
        driver: Selenium WebDriver instance
        timeout (int): Page load timeout in seconds
        implicit_wait (int): Implicit wait time in seconds
        headless (bool): Whether to run browser in headless mode
    """
    
    def __init__(self, base_url: str = "https://1xbet.co.ke",
                 timeout: int = 60, page_load_timeout: int = 30,
                 implicit_wait: int = 10, headless: bool = True):
        """
        Initialize the 1xBet scraper with Selenium WebDriver.
        
        Args:
            base_url: Base URL for 1xBet website
            timeout: Maximum time to wait for elements
            page_load_timeout: Maximum time to wait for page load
            implicit_wait: Implicit wait time for elements
            headless: Whether to run browser in headless mode
        """
        self.base_url = base_url
        self.timeout = timeout
        self.page_load_timeout = page_load_timeout
        self.implicit_wait = implicit_wait
        self.headless = headless
        self.driver = None
        
        self._initialize_driver()
        logger.info("1xBet scraper initialized")
    
    def _initialize_driver(self):
        """Initialize the Selenium WebDriver with Chrome options."""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--disable-gpu")
            
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Set user agent
            chrome_options.add_argument(
                "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            # Initialize driver with webdriver-manager
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            self.driver.set_page_load_timeout(self.page_load_timeout)
            self.driver.implicitly_wait(self.implicit_wait)
            
            logger.info("Chrome WebDriver initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def get_football_odds(self, league: str = "Premier League") -> Optional[List[Dict]]:
        """
        Get football odds for a specific league.
        
        Args:
            league: Name of the league to scrape odds for
        
        Returns:
            List of dictionaries containing match odds or None if scraping fails
        """
        try:
            logger.info(f"Fetching football odds for {league}")
            
            # Navigate to football section
            url = f"{self.base_url}/en/line/Football/"
            self.driver.get(url)
            
            # Wait for page to load
            time.sleep(3)
            
            matches_data = []
            
            # This is a simplified example - actual implementation would need
            # to adapt to 1xBet's specific HTML structure
            try:
                # Wait for match elements to load
                WebDriverWait(self.driver, self.timeout).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "c-events"))
                )
                
                # Note: Actual selectors would need to be updated based on 1xBet's current structure
                matches = self.driver.find_elements(By.CLASS_NAME, "c-events__item")
                
                for match in matches[:10]:  # Limit to first 10 matches
                    try:
                        match_data = self._extract_match_odds(match)
                        if match_data:
                            matches_data.append(match_data)
                    except Exception as e:
                        logger.warning(f"Failed to extract odds from match element: {e}")
                        continue
                
                logger.info(f"Successfully scraped odds for {len(matches_data)} matches")
                return matches_data if matches_data else None
            
            except TimeoutException:
                logger.error("Timeout waiting for match elements")
                return None
        
        except Exception as e:
            logger.error(f"Error fetching football odds: {e}")
            return None
    
    def _extract_match_odds(self, match_element) -> Optional[Dict]:
        """
        Extract odds from a single match element.
        
        Args:
            match_element: Selenium WebElement for a match
        
        Returns:
            Dictionary with match odds or None if extraction fails
        """
        try:
            # This is a template - actual implementation needs real selectors
            match_data = {
                'home_team': 'Team A',
                'away_team': 'Team B',
                'home_odds': 0.0,
                'draw_odds': 0.0,
                'away_odds': 0.0,
                'timestamp': pd.Timestamp.now()
            }
            
            # Example: Extract team names and odds
            # Note: These selectors are placeholders
            # teams = match_element.find_elements(By.CLASS_NAME, "team-name")
            # if len(teams) >= 2:
            #     match_data['home_team'] = teams[0].text
            #     match_data['away_team'] = teams[1].text
            
            # odds = match_element.find_elements(By.CLASS_NAME, "coef")
            # if len(odds) >= 3:
            #     match_data['home_odds'] = float(odds[0].text)
            #     match_data['draw_odds'] = float(odds[1].text)
            #     match_data['away_odds'] = float(odds[2].text)
            
            return match_data
        
        except Exception as e:
            logger.error(f"Error extracting match odds: {e}")
            return None
    
    def get_match_odds_detailed(self, match_url: str) -> Optional[Dict]:
        """
        Get detailed odds for a specific match including multiple markets.
        
        Args:
            match_url: URL of the specific match
        
        Returns:
            Dictionary containing detailed odds for various markets or None if scraping fails
        """
        try:
            logger.info(f"Fetching detailed odds from {match_url}")
            self.driver.get(match_url)
            
            time.sleep(3)
            
            odds_data = {
                '1X2': {},
                'over_under': {},
                'btts': {},
                'timestamp': pd.Timestamp.now()
            }
            
            # Wait for odds to load
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.CLASS_NAME, "c-bets"))
            )
            
            # Extract odds for different markets
            # This is a template - needs real implementation
            
            logger.info("Detailed odds extracted successfully")
            return odds_data
        
        except Exception as e:
            logger.error(f"Error fetching detailed match odds: {e}")
            return None
    
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
    
    def calculate_house_edge(self, odds_list: List[float], odds_format: str = "decimal") -> float:
        """
        Calculate the bookmaker's house edge (margin) from a set of odds.
        
        Args:
            odds_list: List of odds for all outcomes in a market
            odds_format: Format of odds ('decimal' or 'american')
        
        Returns:
            House edge as a percentage
        """
        try:
            implied_probs = [self.calculate_implied_probability(odd, odds_format) for odd in odds_list]
            total_implied_prob = sum(implied_probs)
            
            house_edge = (total_implied_prob - 1.0) * 100
            
            logger.debug(f"House edge calculated: {house_edge:.2f}%")
            return house_edge
        
        except Exception as e:
            logger.error(f"Error calculating house edge: {e}")
            return 0.0
    
    def get_odds_dataframe(self, matches_data: List[Dict]) -> pd.DataFrame:
        """
        Convert matches odds data to a pandas DataFrame.
        
        Args:
            matches_data: List of dictionaries containing match odds
        
        Returns:
            DataFrame with odds data
        """
        if not matches_data:
            logger.warning("No matches data to convert")
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(matches_data)
            
            # Calculate implied probabilities
            if 'home_odds' in df.columns:
                df['home_implied_prob'] = df['home_odds'].apply(
                    lambda x: self.calculate_implied_probability(x) if x > 0 else 0
                )
                df['draw_implied_prob'] = df['draw_odds'].apply(
                    lambda x: self.calculate_implied_probability(x) if x > 0 else 0
                )
                df['away_implied_prob'] = df['away_odds'].apply(
                    lambda x: self.calculate_implied_probability(x) if x > 0 else 0
                )
                
                # Calculate house edge for each match
                df['house_edge'] = df.apply(
                    lambda row: self.calculate_house_edge([row['home_odds'], row['draw_odds'], row['away_odds']]),
                    axis=1
                )
            
            logger.info(f"Created DataFrame with {len(df)} matches")
            return df
        
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close the Selenium WebDriver and cleanup."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("WebDriver closed successfully")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {e}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    try:
        scraper = OneXBetScraper(headless=True)
        
        # Example: Get football odds
        # odds = scraper.get_football_odds("Premier League")
        # if odds:
        #     df = scraper.get_odds_dataframe(odds)
        #     print(df)
        
        # Example: Calculate implied probability
        decimal_odds = 2.50
        implied_prob = scraper.calculate_implied_probability(decimal_odds)
        print(f"Odds: {decimal_odds}, Implied Probability: {implied_prob:.2%}")
        
        # Example: Calculate house edge
        odds_list = [2.10, 3.50, 3.20]
        house_edge = scraper.calculate_house_edge(odds_list)
        print(f"House Edge: {house_edge:.2f}%")
        
        scraper.close()
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
