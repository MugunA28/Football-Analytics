import requests
from bs4 import BeautifulSoup
import pandas as pd

class SportybetScraper:
    def __init__(self):
        self.url = 'https://www.sportybet.com/'

    def get_odds(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Assuming there's a specific pattern for odds on the website
        odds = []

        # Scraping logic for odds goes here (update according to actual site structure)
        matches = soup.find_all('div', class_='match-container')
        for match in matches:
            teams = match.find('div', class_='teams').text.strip()
            odds_value = match.find('div', class_='odds').text.strip()
            if odds_value:  # Check if odds value is present
                odds.append({'teams': teams, 'odds': odds_value})

        return odds

    def save_to_model(self, odds):
        df = pd.DataFrame(odds)
        # Implement your model saving logic here
        # For example, save to a CSV file or database
        df.to_csv('odds_data.csv', index=False)

if __name__ == '__main__':
    scraper = SportybetScraper()
    odds = scraper.get_odds()
    scraper.save_to_model(odds)
