# Football Analytics - Betting Edge Finder

A sophisticated football betting analytics system that scrapes data from multiple sources (SofaScore, FotMob, and 1xBet), employs statistical and machine learning models to analyze matches, and identifies value bets to gain an edge over bookmakers.

## 🎯 Project Overview

This system combines data science, statistical modeling, and sports analytics to:
- Scrape real-time match data and statistics from multiple sources
- Calculate team strengths using Poisson distribution models
- Train machine learning models to predict match outcomes
- Identify value bets by comparing model predictions to bookmaker odds
- Calculate optimal stake sizes using Kelly Criterion
- Manage betting portfolios with risk management

## ⚠️ Legal Disclaimer

**IMPORTANT**: Please read carefully before using this system.

1. **Web Scraping**: This system scrapes data from third-party websites. Always:
   - Check the website's `robots.txt` and Terms of Service
   - Respect rate limits and implement appropriate delays
   - Be aware that scraping may violate the website's ToS
   - Use the data responsibly and legally

2. **Betting Risks**: 
   - **No Guarantee of Profit**: This system does NOT guarantee profits
   - **Responsible Gambling**: Only bet what you can afford to lose
   - **Addiction Warning**: Seek help if gambling becomes problematic
   - **Legal Compliance**: Ensure betting is legal in your jurisdiction
   - **Age Restrictions**: Must be of legal gambling age

3. **Data Accuracy**:
   - Models are based on historical data and probabilities
   - Past performance does not guarantee future results
   - Unexpected events can affect match outcomes

4. **Use at Your Own Risk**: The creators assume no liability for any losses incurred using this system.

## 🌟 Key Features

### Data Collection
- **SofaScore Scraper**: Match statistics, team data, head-to-head records
- **FotMob Scraper**: Expected Goals (xG) data, detailed match analysis
- **1xBet Scraper**: Live odds for multiple betting markets

### Prediction Models
- **Poisson Model**: Statistical approach using team attack/defense strengths
- **ML Predictor**: Random Forest/Gradient Boosting with feature engineering
- **Ensemble Methods**: Combine multiple models for robust predictions

### Value Betting
- **Edge Calculator**: Identify positive expected value bets
- **Kelly Criterion**: Optimal stake sizing with fractional Kelly for safety
- **Portfolio Management**: Track exposure and expected returns
- **Risk Management**: Maximum stake limits and minimum edge thresholds

### Player Analysis (NEW!)
- **PlayerAnalyzer**: Comprehensive player performance analysis system
- **Probability Calculations**: Goal scoring, assist, and clean sheet probabilities
- **xG-Based Analysis**: Expected Goals (xG) and Expected Assists (xA) metrics
- **Matchweek Reports**: Generate detailed analysis for any Premier League matchweek
- **Interactive Visualizations**: Jupyter notebooks with charts and graphs
- **Multiple Output Formats**: Console, JSON, and CSV outputs

### Database & Storage
- SQLAlchemy ORM for data persistence
- PostgreSQL/SQLite support
- Historical data storage for model training and backtesting

## 🛠️ Technology Stack

- **Python 3.8+**
- **Web Scraping**: Requests, BeautifulSoup, Selenium
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Statistics**: SciPy (Poisson distribution)
- **Database**: SQLAlchemy, PostgreSQL
- **Visualization**: Matplotlib, Plotly, Streamlit
- **Testing**: Pytest

## 📁 Project Structure

```
Football-Analytics/
├── data/
│   ├── raw/              # Raw scraped data
│   ├── processed/        # Cleaned and processed data
│   └── models/           # Saved ML models
├── src/
│   ├── scrapers/         # Web scrapers for data collection
│   │   ├── sofascore_scraper.py
│   │   ├── fotmob_scraper.py
│   │   └── oneXbet_scraper.py
│   ├── analysis/         # Player analysis module (NEW!)
│   │   └── player_analyzer.py
│   ├── models/           # Prediction models
│   │   ├── poisson_model.py
│   │   ├── ml_predictor.py
│   │   └── edge_calculator.py
│   ├── scripts/          # Command-line scripts (NEW!)
│   │   └── analyze_matchweek.py
│   ├── utils/            # Utility functions
│   │   ├── data_processor.py
│   │   ├── database.py
│   │   └── player_stats_processor.py  # NEW!
│   └── main.py           # Main application entry point
├── notebooks/            # Jupyter notebooks for analysis
│   └── player_analysis_matchweek26.ipynb  # NEW!
├── tests/                # Unit tests
│   ├── test_scrapers.py
│   └── test_player_analyzer.py  # NEW!
├── config/               # Configuration files
│   ├── config.yaml
│   ├── player_analysis_config.yaml  # NEW!
│   └── .env.example
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- PostgreSQL (optional, SQLite can be used for development)
- Chrome/Chromium browser (for Selenium)

### Step 1: Clone the Repository
```bash
git clone https://github.com/MugunA28/Football-Analytics.git
cd Football-Analytics
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install as Package (Optional)
```bash
pip install -e .
```

### Step 5: Setup Environment Variables
```bash
cp config/.env.example .env
# Edit .env file with your configuration
```

### Step 6: Configure Database
Edit `.env` file:
```
DATABASE_URL=postgresql://user:password@localhost:5432/football_analytics
```

Or use SQLite for development:
```
DATABASE_URL=sqlite:///football_analytics.db
```

### Step 7: Create Database Tables
```python
from src.utils.database import Database

db = Database()
db.create_tables()
```

## 📖 Usage

### Command Line Interface

The system provides a CLI with several commands:

#### 1. Scrape Data
```bash
python src/main.py scrape --config config/config.yaml
```

#### 2. Train Models
```bash
python src/main.py train --data data/processed/matches.csv
```

#### 3. Generate Predictions
```bash
python src/main.py predict
```

#### 4. Analyze Value Bets
```bash
python src/main.py analyze
```

#### 5. Backtest Strategy
```bash
python src/main.py backtest --data data/processed/historical_matches.csv
```

#### 6. Player Analysis (NEW!)
```bash
# Analyze Premier League matchweek
python src/scripts/analyze_matchweek.py --matchweek 26 --output console

# Export to JSON
python src/scripts/analyze_matchweek.py --matchweek 26 --output json --file data/processed/matchweek_26.json

# Export to CSV
python src/scripts/analyze_matchweek.py --matchweek 26 --output csv --file data/processed/matchweek_26.csv

# Verbose output
python src/scripts/analyze_matchweek.py --matchweek 26 --output console --verbose
```

### Python API Usage

#### Example 1: Scraping Data
```python
from src.scrapers.sofascore_scraper import SofaScoreScraper

scraper = SofaScoreScraper()

# Get match statistics
match_stats = scraper.get_match_statistics(match_id=12345)

# Get team statistics
team_stats = scraper.get_team_statistics(
    team_id=42,
    tournament_id=17,  # Premier League
    season_id=52760
)

# Parse to DataFrame
df = scraper.parse_statistics_to_dataframe(match_stats)
```

#### Example 2: Poisson Model Prediction
```python
from src.models.poisson_model import PoissonModel
import pandas as pd

# Load training data
matches_df = pd.read_csv('data/processed/matches.csv')

# Train model
model = PoissonModel(home_advantage=1.3)
model.train(matches_df)

# Predict match
prediction = model.predict_match_outcome('Manchester City', 'Arsenal')
print(f"Home win: {prediction['probabilities']['home_win']:.2%}")
print(f"Draw: {prediction['probabilities']['draw']:.2%}")
print(f"Away win: {prediction['probabilities']['away_win']:.2%}")
```

#### Example 3: Finding Value Bets
```python
from src.models.edge_calculator import EdgeCalculator

calc = EdgeCalculator(kelly_fraction=0.25, max_stake_pct=5.0)

# Model predictions
predictions = {
    'home_team': 'Manchester City',
    'away_team': 'Arsenal',
    'probabilities': {
        'home_win': 0.55,
        'draw': 0.25,
        'away_win': 0.20
    }
}

# Bookmaker odds
bookmaker_odds = {
    'home_odds': 2.1,
    'draw_odds': 3.5,
    'away_odds': 3.2
}

# Find value bets
value_bets = calc.find_value_bets(
    predictions, 
    bookmaker_odds, 
    min_edge=0.03,
    bankroll=1000
)

print(value_bets)
```

#### Example 4: Machine Learning Predictor
```python
from src.models.ml_predictor import MLPredictor
import pandas as pd

# Load and prepare data
matches_df = pd.read_csv('data/processed/matches.csv')

predictor = MLPredictor(model_type='random_forest')

# Create features
features_df = predictor.create_features(matches_df, rolling_window=5)

# Prepare training data
X, y = predictor.prepare_training_data(matches_df, features_df)

# Train
metrics = predictor.train(X, y)
print(f"Test Accuracy: {metrics['test_accuracy']:.2%}")

# Save model
predictor.save_model('data/models/ml_predictor.pkl')
```

#### Example 5: Player Analysis (NEW!)
```python
from src.analysis.player_analyzer import PlayerAnalyzer
from src.scrapers.fotmob_scraper import FotMobScraper
from src.scrapers.sofascore_scraper import SofaScoreScraper

# Initialize scrapers and analyzer
fotmob = FotMobScraper()
sofascore = SofaScoreScraper()
analyzer = PlayerAnalyzer(fotmob_scraper=fotmob, sofascore_scraper=sofascore)

# Get fixtures for matchweek 26
fixtures = analyzer.get_premier_league_fixtures(matchweek=26)
print(f"Found {len(fixtures)} fixtures")

# Calculate goal probability for a player
player_stats = {
    'xg_per_90': 0.85,
    'recent_goals': [2, 1, 0, 1, 2],  # Last 5 matches
    'shots_total': 20,
    'shots_on_target': 12,
    'minutes_played': 450,
    'is_home': True,
    'opponent_rating': 68.0
}

goal_prob = analyzer.calculate_goal_probability(player_stats)
print(f"Goal probability: {goal_prob * 100:.1f}%")

# Generate full analysis report
report = analyzer.generate_analysis_report(matchweek=26, top_n=20)

# Display console output
console_output = analyzer.format_console_output(report, output_top_n=20)
print(console_output)

# Rank players by goal probability
from src.utils.player_stats_processor import scale_to_percentage

top_scorers = analyzer.rank_players_by_metric(
    players=report['top_goal_scorers'],
    metric='goal_probability',
    top_n=10
)

for player in top_scorers:
    print(f"{player['rank']}. {player['name']} - {scale_to_percentage(player['goal_probability']):.1f}%")
```


## 📊 Data Sources

### SofaScore API
- Base URL: `https://api.sofascore.com/api/v1`
- Data: Match statistics, team data, fixtures, head-to-head
- Rate Limit: Recommended 1.5s delay between requests

### FotMob API
- Base URL: `https://www.fotmob.com/api`
- Data: Expected Goals (xG), match details, player statistics
- Rate Limit: Recommended 1.5s delay between requests

### 1xBet
- URL: `https://1xbet.co.ke`
- Data: Live betting odds for various markets
- Method: Selenium WebDriver (requires ChromeDriver)

## 🧠 Model Explanations

### Poisson Model

The Poisson model is a statistical approach that:
1. Calculates team attack and defense strengths from historical data
2. Uses Poisson distribution to predict goal probabilities
3. Incorporates home advantage factor
4. Generates probabilities for all match outcomes

**Formula**:
```
Expected Home Goals = League Avg × Home Attack × Away Defense × Home Advantage
Expected Away Goals = League Avg × Away Attack × Home Defense
```

### Machine Learning Predictor

The ML predictor uses ensemble methods (Random Forest or Gradient Boosting) with:
- **Features**: Rolling averages, form, xG, head-to-head, and more
- **Target**: Match result (Home Win, Draw, Away Win)
- **Training**: Cross-validation with stratified splits
- **Output**: Probability estimates for each outcome

### Edge Calculator

Identifies value bets using:
1. **Implied Probability**: Convert odds to probabilities
2. **Edge Calculation**: Model Probability - Implied Probability
3. **Kelly Criterion**: Optimal stake sizing
4. **Fractional Kelly**: Conservative approach (default 25%)

**Formula**:
```
Kelly % = (bp - q) / b
where:
  b = net odds (decimal odds - 1)
  p = probability of winning
  q = probability of losing (1 - p)
```

### Player Analyzer (NEW!)

The PlayerAnalyzer provides comprehensive player performance analysis:

**Key Features:**
- **Goal Probability Calculation**: Uses xG per 90, recent form, and shot conversion
- **Assist Probability Calculation**: Uses xA per 90, key passes, and recent assists
- **Clean Sheet Probability**: Uses defensive xG, recent clean sheets, and opponent strength
- **Multiple Adjustments**: Home advantage, opponent strength, minutes played

**Probability Components:**

1. **Goal Scoring Probability**:
   - xG per 90 (50% weight): Expected goals normalized per 90 minutes
   - Recent Form (30% weight): Rolling average of goals in last 5 matches
   - Shot Conversion (20% weight): Shots on target / total shots

2. **Assist Probability**:
   - xA per 90 (50% weight): Expected assists normalized per 90 minutes
   - Recent Form (30% weight): Rolling average of assists in last 5 matches
   - Key Passes (20% weight): Number of key passes per match

3. **Clean Sheet Probability**:
   - Defensive xG (40% weight): Expected goals conceded per 90
   - Recent Clean Sheets (30% weight): Clean sheets in last 5 matches
   - Opponent Strength (30% weight): Opponent's attacking rating

**Adjustments Applied:**
- **Minutes Adjustment**: Players with fewer minutes get reduced probabilities
- **Home Advantage**: +10% boost for home teams, -5% for away teams
- **Opponent Strength**: Adjusted based on opponent's defensive/attacking rating

**Example Output**:
```
=== PREMIER LEAGUE MATCHWEEK 26 ANALYSIS ===

TOP 20 PLAYERS - GOAL SCORING PROBABILITY:
Rank | Player Name        | Team          | vs Opponent    | Probability | xG/90 | Form
1    | Erling Haaland    | Man City      | vs Brighton    | 68.5%      | 1.12  | ⭐⭐⭐⭐⭐
2    | Mohamed Salah     | Liverpool     | vs Burnley     | 64.2%      | 0.98  | ⭐⭐⭐⭐⭐
```

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
scraping:
  sofascore:
    base_url: "https://api.sofascore.com/api/v1"
    timeout: 30
    rate_limit_delay: 1.5

models:
  poisson:
    home_advantage: 1.3
  
  edge_calculator:
    min_edge_threshold: 0.03  # 3%
    kelly_fraction: 0.25
    max_stake_pct: 5.0

betting:
  bankroll: 1000
  min_edge: 0.03
  max_stake_pct: 5.0
```

Edit `.env`:
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/football_analytics
MIN_EDGE_THRESHOLD=0.03
KELLY_FRACTION=0.25
MAX_STAKE_PCT=5.0
BANKROLL=1000
```

## 🧪 Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_scrapers.py -v

# Run with detailed output
pytest -v -s
```

## 📈 Performance Metrics

Track these metrics for model evaluation:
- **Accuracy**: Percentage of correct predictions
- **ROI**: Return on Investment
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of winning bets
- **Average Edge**: Average edge across all bets

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write tests for new features

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SofaScore for comprehensive football statistics
- FotMob for Expected Goals (xG) data
- Scientific Python community (NumPy, Pandas, Scikit-learn)
- Sports analytics research community

## 📧 Contact

For questions, suggestions, or issues:
- GitHub Issues: [Create an issue](https://github.com/MugunA28/Football-Analytics/issues)
- Email: info@football-analytics.com

## 🔗 Resources

### Sports Analytics
- [Expected Goals Philosophy](https://statsbomb.com/articles/soccer/expected-goals-philosophy/)
- [Poisson Distribution in Football](https://www.pinnacle.com/en/betting-articles/Soccer/how-to-calculate-poisson-distribution/)

### Betting Theory
- [Kelly Criterion Explained](https://en.wikipedia.org/wiki/Kelly_criterion)
- [Value Betting Guide](https://www.sportsbettingdime.com/guides/strategy/value-betting/)

### Python Libraries
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [SQLAlchemy Documentation](https://www.sqlalchemy.org/)

---

**Remember**: Bet responsibly. This is a tool for educational and analytical purposes. Always gamble within your means and seek help if needed.

**Gambling Help Resources**:
- 🇺🇸 National Council on Problem Gambling: 1-800-522-4700
- 🇬🇧 BeGambleAware: 0808-8020-133
- 🇪🇺 GamCare: www.gamcare.org.uk
