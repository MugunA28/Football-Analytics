# ⚽ Football Analytics - Professional Betting Prediction System

Advanced football match prediction system using **Dixon-Coles statistical model**, ensemble machine learning, and league-aware data analysis.

## 🎯 Features

- **Dixon-Coles Model**: Industry-standard model used by professional betting syndicates
- **Ensemble Prediction**: Combines multiple models for higher accuracy
- **League-Aware Estimates**: Uses real historical league data when live data unavailable
- **Over 1.5 Goals Prediction**: High-accuracy predictions for total match goals
- **Team Scoring Predictions**: Individual team Over 0.5 goals analysis
- **BTTS Analysis**: Both Teams To Score probability calculations
- **Real-time xG Calculation**: Expected Goals for each team

## 🔥 Value Bet Detector (NEW)

The newest and most powerful tool — finds **value bets** by comparing the model's probability estimates against bookmaker odds.

### How to Run

```bash
python value_bet.py
```

### Input Format

```
Team A vs Team B (home_o0.5 away_o0.5 over_1.5)
```

Where:
- `home_o0.5` = Bookmaker odds for home team Over 0.5 Goals
- `away_o0.5` = Bookmaker odds for away team Over 0.5 Goals
- `over_1.5` = Bookmaker odds for Over 1.5 Match Goals

### Usage Example

```
> Manchester City vs Arsenal (1.12 1.45 1.40)
> Liverpool vs Sheffield Utd (1.08 1.60 1.18)
> Burnley vs Everton (1.35 1.55 1.65)
> DONE
```

### Example Output

```
📊 VALUE BET ANALYSIS (Threshold: >15%)
==================================================================================

MATCH 1: Manchester City vs Arsenal  [Premier League]
──────────────────────────────────────────────────────────────────
  🏠 Man City Over 0.5 Goals
     Model: 91.2%  |  Bookie: 89.3% (1.12)  |  Edge: +1.9%    ❌ NO VALUE

  ✈️  Arsenal Over 0.5 Goals
     Model: 75.4%  |  Bookie: 69.0% (1.45)  |  Edge: +6.4%    ❌ NO VALUE

  ⚽ Over 1.5 Match Goals
     Model: 89.3%  |  Bookie: 71.4% (1.40)  |  Edge: +17.9%   🔥 VALUE BET!

==================================================================================
🔥 RECOMMENDED BETS (Edge > 15%):
==================================================================================

  1. Man City vs Arsenal — Over 1.5 Match Goals
     Edge: +17.9%  |  Model: 89.3%  |  Bookie Odds: 1.40
     Confidence: 🔥🔥🔥

==================================================================================
📋 SUMMARY: 1 value bet(s) found out of 3 markets analyzed
==================================================================================
```

### Edge Verdicts

| Edge | Verdict | Meaning |
|------|---------|---------|
| > 15% | 🔥 VALUE BET! | Strong edge — recommended bet |
| 10% – 15% | ⚠️ MARGINAL | Close but below threshold |
| 0% – 10% | ❌ NO VALUE | Not worth it |
| < 0% | 🚫 AVOID | Bookie odds are tighter than model |

### How It Works

The Value Bet Detector uses a **combined probability approach**:

1. **Odds-Implied Probability** — converted directly from bookmaker odds
2. **Dixon-Coles Model** — using league-specific xG estimates
3. **Ensemble Weighting**:
   - Fallback mode: 50% model / 50% odds-implied
   - With real data: 80% model / 20% odds-implied
4. **Auto-detects league** from 60+ team names across 8+ leagues

### Markets Analyzed Per Match

- 🏠 Home team Over 0.5 Goals
- ✈️ Away team Over 0.5 Goals
- ⚽ Over 1.5 Match Goals

### Results Auto-Save

Results are automatically saved to `results/value_YYYYMMDD_HHMMSS.json`.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MugunA28/Football-Analytics.git
cd Football-Analytics

# Install dependencies
pip install requests beautifulsoup4 pandas numpy scipy

# Run the analyzer
python bet_pro.py
```

### Usage

```bash
# Start the professional analyzer
python bet_pro.py

# Enter league name
> Premier League

# Enter matches (format: Team A vs Team B (home_odds draw_odds away_odds))
> Manchester City vs Arsenal (1.50 3.50 6.00)
> Liverpool vs Chelsea (2.10 3.40 3.50)
> DONE

# View predictions immediately!
```

## 📊 Example Output

```
🎯 PREMIER LEAGUE - PREDICTIONS
==================================================================================

MATCH 1: Manchester City vs Arsenal [✓ REAL DATA]
──────────────────────────────────────────────────────────────────────────────

   ⚽ Expected Goals: Manchester City 2.1 - 1.4 Arsenal
      Total Expected: 3.5 goals

   ⭐⭐⭐⭐⭐ OVER 1.5 MATCH GOALS
      Probability: 89.3%
      Confidence: Very High
      Prediction: Over 1.5

   🏠 Manchester City TO SCORE (Over 0.5 Goals)
      Probability: 91.2%
      Confidence: Very High
      Prediction: YES

   ✈️  Arsenal TO SCORE (Over 0.5 Goals)
      Probability: 75.4%
      Confidence: High
      Prediction: YES

==================================================================================
💎 BEST BETS - PREMIER LEAGUE
==================================================================================

🏆 TOP PICK:
   Match: Manchester City vs Arsenal
   Bet: Manchester City Over 0.5 Goals
   ⭐⭐⭐⭐⭐ 91.2% [Very High]

📋 ALL BEST BETS:

   🔥 Best Over 1.5 Match Goals:
      Manchester City vs Arsenal
      ⭐⭐⭐⭐⭐ 89.3% (Expected: 3.5 goals)

   ⚽ Best Team to Score:
      Manchester City (Manchester City vs Arsenal)
      ⭐⭐⭐⭐⭐ 91.2% (xG: 2.1)
```

## 🧠 How It Works

### 1. **Dixon-Coles Model**
- Accounts for low-scoring correlation in football
- Uses Poisson distribution with correlation parameter (ρ ≈ -0.13)
- Adjusts for score dependencies (0-0, 1-0, 0-1, 1-1)

### 2. **Ensemble Prediction**
Combines three models:
- **Model 1**: Dixon-Coles with home/away splits
- **Model 2**: Form-weighted expected goals (recent 5 matches)
- **Model 3**: Odds-implied probability

Weights adjust based on data availability:
- **Real data available**: 45% Dixon-Coles, 35% Form, 20% Odds
- **Using estimates**: 30% Dixon-Coles, 25% Form, 45% Odds

### 3. **League-Aware Estimates**
When live team data is unavailable, uses historical league averages:

| League | Avg Goals/Game | Home Scoring | Away Scoring |
|--------|---------------|--------------|--------------|
| Premier League | 2.75 | 1.55 | 1.15 |
| Bundesliga | 3.10 | 1.65 | 1.25 |
| Serie A | 2.65 | 1.40 | 1.00 |
| La Liga | 2.70 | 1.45 | 1.10 |
| A-League | 2.55 | 1.35 | 1.10 |

### 4. **Confidence Levels**
- **Very High**: ≥85% probability
- **High**: 75-84% probability
- **Medium**: 60-74% probability
- **Low-Medium**: 50-59% probability
- **Low**: <50% probability

## 📁 Project Structure

```
Football-Analytics/
├── value_bet.py                        # 🔥 Value Bet Detector (NEW)
├── bet_pro.py                          # Professional analyzer
├── bet_simple.py                       # Simple interface
├── src/
│   ├── prediction/
│   │   ├── value_detector.py           # Value detection engine (NEW)
│   │   ├── ensemble_predictor.py       # Ensemble model
│   │   ├── dixon_coles.py             # Dixon-Coles implementation
│   │   └── betting_analyzer.py        # Legacy analyzer
│   └── scrapers/
│       ├── free_data_fetcher.py       # Free data sources
│       ├── league_estimator.py        # League-based estimates
│       └── fotmob_scraper.py          # FotMob data scraper
├── results/                            # Saved predictions (auto-generated)
└── README.md
```

## 🔬 Statistical Models

### Dixon-Coles Formula

```
P(X=x, Y=y) = τ(x,y,λ₁,λ₂) × Poisson(λ₁,x) × Poisson(λ₂,y)

where:
λ₁ = home team expected goals
λ₂ = away team expected goals
τ = correlation adjustment for low scores
```

### Expected Goals Calculation

```
λ_home = (home_attack / league_avg) × (away_defense / league_avg) × league_avg × 1.1
λ_away = (away_attack / league_avg) × (home_defense / league_avg) × league_avg × 0.95
```

## 📈 Accuracy

**Target Accuracy**: 70-80% on Over 1.5 predictions

Based on:
- Dixon-Coles statistical model (proven in academic research)
- Ensemble approach reducing individual model variance
- League-specific historical data
- Odds market intelligence

## 🎲 Supported Leagues

- ✅ Premier League
- ✅ Championship
- ✅ La Liga
- ✅ Serie A
- ✅ Bundesliga
- ✅ Ligue 1
- ✅ Eredivisie
- ✅ A-League (Australian)
- ✅ MLS
- ✅ Scottish Premiership
- ✅ Any league (uses default estimates)

## 💡 Tips for Best Results

1. **Use accurate odds**: More recent odds = better predictions
2. **Multiple leagues**: Diversify across different competitions
3. **High confidence picks**: Focus on 70%+ probability bets
4. **Combine predictions**: Use both Over 1.5 AND team totals
5. **Track results**: Save predictions and monitor accuracy

## 🛠️ Advanced Usage

### Save Results

```python
# Results auto-save to results/ directory
# Format: results/pro_YYYYMMDD_HHMMSS.json
```

### Batch Analysis

```bash
# Analyze multiple leagues in one session
python bet_pro.py

> Premier League
> Man City vs Arsenal (1.50 3.50 6.00)
> DONE

> La Liga  
> Barcelona vs Real Madrid (2.20 3.30 3.10)
> DONE

> QUIT  # See overall summary
```

## 📊 Data Sources

- **Primary**: FotMob API (free, real-time team statistics)
- **Fallback**: League historical averages (2023-24 season data)
- **Odds**: User-provided bookmaker odds

## 🔄 Updates & Maintenance

**Last Updated**: February 17, 2026

**Recent Changes**:
- ✅ Added Dixon-Coles model implementation
- ✅ Enhanced ensemble prediction system
- ✅ Added league-aware fallback estimates
- ✅ Improved confidence calibration
- ✅ Single best team selection logic
- ✅ Logical consistency checks

## 📝 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 🐛 Known Issues

- FotMob API occasionally returns 404 (using league estimates as fallback)
- Some team name variations may not match (trying multiple variations)

## 📧 Contact

**GitHub**: [@MugunA28](https://github.com/MugunA28)

---

⭐ **Star this repo if it helps your betting strategy!** ⭐

**Disclaimer**: This tool is for educational and entertainment purposes. Bet responsibly.