"""
Value Bet Detector
Combines Dixon-Coles statistical model with bookmaker odds analysis
to identify value bets with an edge threshold of >15%.
"""

import re
import json
import os
from math import exp
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, 'src')

from prediction.dixon_coles import DixonColesModel
from scrapers.league_estimator import LeagueEstimator

# ---------------------------------------------------------------------------
# League auto-detection: known team → league mapping
# ---------------------------------------------------------------------------
TEAM_LEAGUE_MAP: Dict[str, str] = {
    # Premier League
    "Arsenal": "Premier League",
    "Aston Villa": "Premier League",
    "Bournemouth": "Premier League",
    "Brentford": "Premier League",
    "Brighton": "Premier League",
    "Burnley": "Premier League",
    "Chelsea": "Premier League",
    "Crystal Palace": "Premier League",
    "Everton": "Premier League",
    "Fulham": "Premier League",
    "Liverpool": "Premier League",
    "Luton Town": "Premier League",
    "Luton": "Premier League",
    "Man City": "Premier League",
    "Manchester City": "Premier League",
    "Man Utd": "Premier League",
    "Manchester United": "Premier League",
    "Newcastle": "Premier League",
    "Newcastle United": "Premier League",
    "Nottm Forest": "Premier League",
    "Nottingham Forest": "Premier League",
    "Sheffield Utd": "Premier League",
    "Sheffield United": "Premier League",
    "Tottenham": "Premier League",
    "Spurs": "Premier League",
    "West Ham": "Premier League",
    "West Ham United": "Premier League",
    "Wolves": "Premier League",
    "Wolverhampton": "Premier League",
    # Championship
    "Leeds United": "Championship",
    "Leeds": "Championship",
    "Leicester City": "Championship",
    "Leicester": "Championship",
    "Ipswich Town": "Championship",
    "Southampton": "Championship",
    "Sunderland": "Championship",
    "Norwich City": "Championship",
    "Middlesbrough": "Championship",
    "Watford": "Championship",
    "Birmingham City": "Championship",
    "Swansea City": "Championship",
    "Stoke City": "Championship",
    "Preston North End": "Championship",
    # La Liga
    "Real Madrid": "La Liga",
    "Barcelona": "La Liga",
    "Atletico Madrid": "La Liga",
    "Atletico": "La Liga",
    "Sevilla": "La Liga",
    "Real Betis": "La Liga",
    "Real Sociedad": "La Liga",
    "Villarreal": "La Liga",
    "Valencia": "La Liga",
    "Athletic Bilbao": "La Liga",
    "Athletic Club": "La Liga",
    "Girona": "La Liga",
    "Getafe": "La Liga",
    "Osasuna": "La Liga",
    "Rayo Vallecano": "La Liga",
    "Cadiz": "La Liga",
    "Almeria": "La Liga",
    "Granada": "La Liga",
    "Celta Vigo": "La Liga",
    "Mallorca": "La Liga",
    "Las Palmas": "La Liga",
    # Bundesliga
    "Bayern Munich": "Bundesliga",
    "Bayern": "Bundesliga",
    "Borussia Dortmund": "Bundesliga",
    "Dortmund": "Bundesliga",
    "RB Leipzig": "Bundesliga",
    "Bayer Leverkusen": "Bundesliga",
    "Leverkusen": "Bundesliga",
    "Eintracht Frankfurt": "Bundesliga",
    "Frankfurt": "Bundesliga",
    "Wolfsburg": "Bundesliga",
    "Freiburg": "Bundesliga",
    "Hoffenheim": "Bundesliga",
    "Borussia Monchengladbach": "Bundesliga",
    "Augsburg": "Bundesliga",
    "Mainz": "Bundesliga",
    "Union Berlin": "Bundesliga",
    "Werder Bremen": "Bundesliga",
    "Bochum": "Bundesliga",
    "Darmstadt": "Bundesliga",
    "Cologne": "Bundesliga",
    # Serie A
    "Inter Milan": "Serie A",
    "Inter": "Serie A",
    "AC Milan": "Serie A",
    "Milan": "Serie A",
    "Juventus": "Serie A",
    "Napoli": "Serie A",
    "AS Roma": "Serie A",
    "Roma": "Serie A",
    "Lazio": "Serie A",
    "Atalanta": "Serie A",
    "Fiorentina": "Serie A",
    "Torino": "Serie A",
    "Bologna": "Serie A",
    "Udinese": "Serie A",
    "Monza": "Serie A",
    "Genoa": "Serie A",
    "Cagliari": "Serie A",
    "Verona": "Serie A",
    "Lecce": "Serie A",
    "Empoli": "Serie A",
    "Frosinone": "Serie A",
    "Salernitana": "Serie A",
    # Ligue 1
    "PSG": "Ligue 1",
    "Paris Saint-Germain": "Ligue 1",
    "Paris SG": "Ligue 1",
    "Monaco": "Ligue 1",
    "Marseille": "Ligue 1",
    "Lille": "Ligue 1",
    "Lens": "Ligue 1",
    "Lyon": "Ligue 1",
    "Nice": "Ligue 1",
    "Rennes": "Ligue 1",
    "Toulouse": "Ligue 1",
    "Reims": "Ligue 1",
    "Montpellier": "Ligue 1",
    "Nantes": "Ligue 1",
    "Brest": "Ligue 1",
    "Strasbourg": "Ligue 1",
    "Lorient": "Ligue 1",
    "Metz": "Ligue 1",
    "Clermont": "Ligue 1",
    # Eredivisie
    "Ajax": "Eredivisie",
    "PSV": "Eredivisie",
    "PSV Eindhoven": "Eredivisie",
    "Feyenoord": "Eredivisie",
    "AZ Alkmaar": "Eredivisie",
    "AZ": "Eredivisie",
    "Utrecht": "Eredivisie",
    "Twente": "Eredivisie",
    "FC Groningen": "Eredivisie",
    # MLS
    "LA Galaxy": "MLS",
    "LAFC": "MLS",
    "Seattle Sounders": "MLS",
    "Portland Timbers": "MLS",
    "Atlanta United": "MLS",
    "New York City FC": "MLS",
    "New York Red Bulls": "MLS",
    "Inter Miami": "MLS",
    "Chicago Fire": "MLS",
    "Columbus Crew": "MLS",
    # Scottish Premiership
    "Celtic": "Scottish Premiership",
    "Rangers": "Scottish Premiership",
    "Hearts": "Scottish Premiership",
    "Hibernian": "Scottish Premiership",
    "Aberdeen": "Scottish Premiership",
    "Dundee": "Scottish Premiership",
    "Motherwell": "Scottish Premiership",
    "St Mirren": "Scottish Premiership",
}


# ---------------------------------------------------------------------------
# Confidence stars helper
# ---------------------------------------------------------------------------
def _confidence_stars(edge_pct: float) -> str:
    """Return flame/star string based on edge magnitude."""
    if edge_pct >= 25:
        return "🔥🔥🔥🔥🔥"
    elif edge_pct >= 20:
        return "🔥🔥🔥🔥"
    elif edge_pct >= 15:
        return "🔥🔥🔥"
    elif edge_pct >= 10:
        return "🔥🔥"
    else:
        return "🔥"


# ---------------------------------------------------------------------------
# Main ValueDetector class
# ---------------------------------------------------------------------------
class ValueDetector:
    """
    Detects value bets by combining Dixon-Coles model probabilities with
    bookmaker implied probabilities.

    Input format:
        Team A vs Team B (home_o0.5 away_o0.5 over_1.5)

    Edge threshold: >15%
    """

    EDGE_THRESHOLD = 15.0  # percent

    # Pre-built lowercase lookup for O(1) league detection
    _TEAM_LEAGUE_LOWER: Dict[str, str] = {
        k.lower(): v for k, v in TEAM_LEAGUE_MAP.items()
    }

    def __init__(self):
        self.dixon_coles = DixonColesModel(rho=-0.13)
        self.league_estimator = LeagueEstimator()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def parse_match(self, match_string: str) -> Optional[Dict]:
        """
        Parse a match string in the format:
            Team A vs Team B (home_o0.5 away_o0.5 over_1.5)

        Returns a dict with keys: home_team, away_team, odds
        (home_o05, away_o05, over_15) or None on failure.
        """
        pattern = (
            r'(.+?)\s+vs\s+(.+?)\s*'
            r'\(\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*\)'
        )
        m = re.search(pattern, match_string, re.IGNORECASE)
        if not m:
            return None
        return {
            'home_team': m.group(1).strip(),
            'away_team': m.group(2).strip(),
            'odds': {
                'home_o05': float(m.group(3)),
                'away_o05': float(m.group(4)),
                'over_15': float(m.group(5)),
            },
        }

    # ------------------------------------------------------------------
    # League auto-detection
    # ------------------------------------------------------------------
    def detect_league(self, home_team: str, away_team: str) -> str:
        """
        Auto-detect league from team names using the pre-built lowercase map.
        Falls back to 'Default' if neither team is recognised.
        """
        for team in (home_team, away_team):
            league = self._TEAM_LEAGUE_LOWER.get(team.lower())
            if league:
                return league
        return 'Default'

    # ------------------------------------------------------------------
    # Implied probability
    # ------------------------------------------------------------------
    @staticmethod
    def implied_probability(decimal_odds: float) -> float:
        """Convert decimal odds to implied probability (as percentage)."""
        if decimal_odds <= 0:
            return 0.0
        return (1.0 / decimal_odds) * 100.0

    # ------------------------------------------------------------------
    # Model probabilities via Dixon-Coles
    # ------------------------------------------------------------------
    def _xg_from_league(
        self, home_team: str, away_team: str, league: str
    ) -> Tuple[float, float]:
        """
        Derive expected goals from league statistics using the
        LeagueEstimator (identical to what free_data_fetcher uses as
        its fallback).
        """
        home_stats = self.league_estimator.get_league_estimate(
            home_team, league, is_home=True
        )
        away_stats = self.league_estimator.get_league_estimate(
            away_team, league, is_home=False
        )

        home_xg = (
            home_stats['home_avg_scored'] + away_stats['away_avg_conceded']
        ) / 2
        away_xg = (
            away_stats['away_avg_scored'] + home_stats['home_avg_conceded']
        ) / 2

        home_xg = round(max(0.3, min(4.0, home_xg)), 3)
        away_xg = round(max(0.3, min(4.0, away_xg)), 3)
        return home_xg, away_xg

    def model_probabilities(
        self, home_team: str, away_team: str, league: str
    ) -> Dict[str, float]:
        """
        Calculate Dixon-Coles model probabilities for the three markets.

        Returns dict with keys:
            home_o05  – P(home scores ≥ 1)
            away_o05  – P(away scores ≥ 1)
            over_15   – P(total goals ≥ 2)
        All as percentages.
        """
        home_xg, away_xg = self._xg_from_league(home_team, away_team, league)

        # P(home scores at least 1) = 1 − P(home scores 0)
        home_o05 = (1 - exp(-home_xg)) * 100.0
        away_o05 = (1 - exp(-away_xg)) * 100.0

        # Over 1.5 via Dixon-Coles
        over_15_result = self.dixon_coles.over_under_probability(
            home_xg, away_xg, threshold=1.5
        )
        over_15 = over_15_result['over']

        return {
            'home_o05': round(home_o05, 2),
            'away_o05': round(away_o05, 2),
            'over_15': round(over_15, 2),
            'home_xg': home_xg,
            'away_xg': away_xg,
        }

    # ------------------------------------------------------------------
    # Ensemble weighting
    # ------------------------------------------------------------------
    def ensemble_probability(
        self,
        dixon_coles_probs: Dict[str, float],
        implied_probs: Dict[str, float],
        data_available: bool = False,
    ) -> Dict[str, float]:
        """
        Combine Dixon-Coles model probabilities with odds-implied
        probabilities.

        Per the spec, the target weights are:
            real data  → Dixon-Coles 45% | Form 35% | Odds-Implied 20%
            fallback   → Dixon-Coles 25% | Form 25% | Odds-Implied 50%

        Form data is folded into the Dixon-Coles estimate (both derive
        from the same league statistics), giving effective blends of:
            real data  → DC+Form = 80%,  Odds-Implied = 20%
            fallback   → DC+Form = 50%,  Odds-Implied = 50%
        """
        if data_available:
            dc_weight = 0.80
            odds_weight = 0.20
        else:
            dc_weight = 0.50
            odds_weight = 0.50

        result = {}
        for market in ('home_o05', 'away_o05', 'over_15'):
            dc_p = dixon_coles_probs.get(market, 0.0)
            odds_p = implied_probs.get(market, 0.0)
            blended = dc_weight * dc_p + odds_weight * odds_p
            result[market] = round(blended, 2)

        return result

    # ------------------------------------------------------------------
    # Verdict helpers
    # ------------------------------------------------------------------
    def _verdict(self, edge: float) -> str:
        if edge > self.EDGE_THRESHOLD:
            return "🔥 VALUE BET!"
        elif edge >= 10.0:
            return "⚠️ MARGINAL"
        elif edge >= 0.0:
            return "❌ NO VALUE"
        else:
            return "🚫 AVOID"

    # ------------------------------------------------------------------
    # Analyse a single match
    # ------------------------------------------------------------------
    def analyze_match(self, match_data: Dict) -> Dict:
        """
        Full pipeline for a single parsed match.

        Returns a dict with all model, bookie and edge values for the
        three markets, plus verdict strings and recommended_bet flag.
        """
        home_team = match_data['home_team']
        away_team = match_data['away_team']
        odds = match_data['odds']

        # League auto-detection
        league = self.detect_league(home_team, away_team)

        # Implied probabilities from bookie odds
        implied = {
            'home_o05': round(self.implied_probability(odds['home_o05']), 2),
            'away_o05': round(self.implied_probability(odds['away_o05']), 2),
            'over_15': round(self.implied_probability(odds['over_15']), 2),
        }

        # Dixon-Coles model probabilities
        dc_probs = self.model_probabilities(home_team, away_team, league)

        # No live data fetch in this path – always "fallback" weighting
        data_available = False
        ensemble = self.ensemble_probability(dc_probs, implied, data_available)

        # Edge per market
        markets = {}
        for market_key, label, icon in (
            ('home_o05', f'{home_team} Over 0.5 Goals', '🏠'),
            ('away_o05', f'{away_team} Over 0.5 Goals', '✈️ '),
            ('over_15', 'Over 1.5 Match Goals', '⚽'),
        ):
            bookie_prob = implied[market_key]
            model_prob = ensemble[market_key]
            edge = round(model_prob - bookie_prob, 2)
            verdict = self._verdict(edge)
            markets[market_key] = {
                'label': label,
                'icon': icon,
                'model_prob': model_prob,
                'bookie_prob': bookie_prob,
                'bookie_odds': odds[market_key],
                'edge': edge,
                'verdict': verdict,
                'is_value': edge > self.EDGE_THRESHOLD,
            }

        return {
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'home_xg': dc_probs['home_xg'],
            'away_xg': dc_probs['away_xg'],
            'markets': markets,
            'data_available': data_available,
        }

    # ------------------------------------------------------------------
    # Analyse multiple matches
    # ------------------------------------------------------------------
    def analyze_matches(self, match_strings: List[str]) -> List[Dict]:
        """Parse and analyse a list of match strings."""
        results = []
        for s in match_strings:
            parsed = self.parse_match(s)
            if parsed:
                results.append(self.analyze_match(parsed))
            else:
                print(f"   ⚠️  Could not parse: {s!r}")
        return results

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------
    def format_output(self, results: List[Dict]) -> str:
        """
        Generate the formatted output string as specified in the
        problem statement.
        """
        lines: List[str] = []
        sep = "=" * 82
        thin_sep = "─" * 66

        lines.append("")
        lines.append("📊 VALUE BET ANALYSIS (Threshold: >15%)")
        lines.append(sep)

        value_bets: List[Dict] = []

        for i, r in enumerate(results, 1):
            lines.append("")
            lines.append(
                f"MATCH {i}: {r['home_team']} vs {r['away_team']}"
                f"  [{r['league']}]"
            )
            lines.append(thin_sep)

            for market_key in ('home_o05', 'away_o05', 'over_15'):
                mkt = r['markets'][market_key]
                edge_str = (
                    f"+{mkt['edge']:.1f}%"
                    if mkt['edge'] >= 0
                    else f"{mkt['edge']:.1f}%"
                )
                lines.append(
                    f"  {mkt['icon']} {mkt['label']}"
                )
                lines.append(
                    f"     Model: {mkt['model_prob']:.1f}%"
                    f"  |  Bookie: {mkt['bookie_prob']:.1f}%"
                    f" ({mkt['bookie_odds']})"
                    f"  |  Edge: {edge_str:<8}  {mkt['verdict']}"
                )
                lines.append("")

                if mkt['is_value']:
                    value_bets.append(
                        {
                            'match': f"{r['home_team']} vs {r['away_team']}",
                            'label': mkt['label'],
                            'edge': mkt['edge'],
                            'model_prob': mkt['model_prob'],
                            'bookie_odds': mkt['bookie_odds'],
                        }
                    )

            lines.append(sep)

        # Recommended bets section
        if value_bets:
            lines.append("🔥 RECOMMENDED BETS (Edge > 15%):")
            lines.append(sep)
            lines.append("")
            for idx, vb in enumerate(value_bets, 1):
                edge_str = f"+{vb['edge']:.1f}%"
                lines.append(
                    f"  {idx}. {vb['match']} — {vb['label']}"
                )
                lines.append(
                    f"     Edge: {edge_str}"
                    f"  |  Model: {vb['model_prob']:.1f}%"
                    f"  |  Bookie Odds: {vb['bookie_odds']}"
                )
                lines.append(
                    f"     Confidence: {_confidence_stars(vb['edge'])}"
                )
                lines.append("")
        else:
            lines.append("No value bets found above the 15% threshold.")
            lines.append("")

        total_markets = len(results) * 3
        lines.append(sep)
        lines.append(
            f"📋 SUMMARY: {len(value_bets)} value bet(s) found"
            f" out of {total_markets} markets analyzed"
        )
        lines.append(sep)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Save to JSON
    # ------------------------------------------------------------------
    def save_results(self, results: List[Dict], filename: Optional[str] = None) -> str:
        """Auto-save analysis results to results/ directory as JSON."""
        if not os.path.exists('results'):
            os.makedirs('results')

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'results/value_{timestamp}.json'

        # Serialise – convert any non-serialisable types
        payload = {
            'timestamp': datetime.now().isoformat(),
            'threshold_pct': self.EDGE_THRESHOLD,
            'matches': results,
        }
        with open(filename, 'w') as fh:
            json.dump(payload, fh, indent=2)

        return filename
