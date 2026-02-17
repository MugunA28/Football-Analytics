#!/usr/bin/env python3
"""
PROFESSIONAL BETTING ANALYZER
Shows best single team to score pick
"""

import sys
import json
import os
from datetime import datetime

sys.path.insert(0, 'src')
from prediction.ensemble_predictor import EnsemblePredictor

def save_results(results, filename=None):
    """Optional save to file."""
    if not os.path.exists('results'):
        os.makedirs('results')
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/pro_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filename

def display_match_predictions(results, league):
    """Display predictions IMMEDIATELY after analysis."""
    print(f"\n{'='*90}")
    print(f"📊 {league.upper()} - PREDICTIONS")
    print(f"{'='*90}")
    
    for i, m in enumerate(results, 1):
        xg = m['expected_goals']
        o15 = m['over_1_5']
        ht = m['home_total']
        at = m['away_total']
        btts = m['btts']
        
        data_source = "✓ REAL DATA" if m['using_real_data'] else "~ ESTIMATES"
        
        print(f"\n{'─'*90}")
        print(f"MATCH {i}: {m['home_team']} vs {m['away_team']} [{data_source}]")
        print(f"{'─'*90}")
        
        print(f"\n   ⚽ Expected Goals: {m['home_team']} {xg['home']} - {xg['away']} {m['away_team']}")
        print(f"      Total Expected: {xg['total']} goals")
        
        print(f"\n   {o15['stars']} OVER 1.5 MATCH GOALS")
        print(f"      Probability: {o15['over_1.5']}%")
        print(f"      Confidence: {o15['confidence']}")
        print(f"      Prediction: {o15['prediction']}")
        
        print(f"\n   🏠 {m['home_team']} TO SCORE (Over 0.5 Goals)")
        print(f"      Probability: {ht['over_0.5']}%")
        print(f"      Confidence: {ht['confidence']}")
        print(f"      Prediction: {ht['prediction']}")
        
        print(f"\n   ✈️  {m['away_team']} TO SCORE (Over 0.5 Goals)")
        print(f"      Probability: {at['over_0.5']}%")
        print(f"      Confidence: {at['confidence']}")
        print(f"      Prediction: {at['prediction']}")
        
        print(f"\n   🤝 BOTH TEAMS TO SCORE (BTTS)")
        print(f"      Probability: {btts['probability']}%")
        print(f"      Confidence: {btts['confidence']}")
        print(f"      Prediction: {btts['prediction']}")
    
    # Best bets for this league
    print(f"\n{'='*90}")
    print(f"💎 BEST BETS - {league.upper()}")
    print(f"{'='*90}")
    
    best_bets = find_best_bets(results)
    display_best_bets(best_bets, league)

def find_best_bets(results):
    """Find top bets - SINGLE best team to score."""
    best_bets = {
        'over_1_5': None,
        'team_total': None,
        'btts': None
    }
    
    max_scores = {
        'over_1_5': 0,
        'team_total': 0,
        'btts': 0
    }
    
    for r in results:
        # Over 1.5
        o15 = r['over_1_5']
        if o15['prediction'] == 'Over 1.5':
            score = o15['over_1.5']
            if o15['confidence'] == 'Very High':
                score += 10
            elif o15['confidence'] == 'High':
                score += 5
            
            if score > max_scores['over_1_5']:
                max_scores['over_1_5'] = score
                best_bets['over_1_5'] = {
                    'match': f"{r['home_team']} vs {r['away_team']}",
                    'prediction': 'Over 1.5 Goals',
                    'probability': o15['over_1.5'],
                    'confidence': o15['confidence'],
                    'stars': o15['stars'],
                    'xg': r['expected_goals']['total'],
                    'league': r['league']
                }
        
        # CHANGED: Compare both teams and pick the SINGLE best one
        ht = r['home_total']
        at = r['away_total']
        
        # Check home team
        if ht['prediction'] in ['YES', 'YES (Moderate)']:
            score = ht['over_0.5']
            if ht['confidence'] == 'Very High':
                score += 8
            elif ht['confidence'] == 'High':
                score += 4
            
            if score > max_scores['team_total']:
                max_scores['team_total'] = score
                best_bets['team_total'] = {
                    'match': f"{r['home_team']} vs {r['away_team']}",
                    'prediction': f"{r['home_team']} Over 0.5 Goals",
                    'probability': ht['over_0.5'],
                    'confidence': ht['confidence'],
                    'stars': '⭐⭐⭐⭐⭐' if ht['over_0.5'] > 90 else '⭐⭐⭐⭐' if ht['over_0.5'] > 80 else '⭐⭐⭐',
                    'xg': r['expected_goals']['home'],
                    'team': r['home_team'],
                    'league': r['league']
                }
        
        # Check away team
        if at['prediction'] in ['YES', 'YES (Moderate)']:
            score = at['over_0.5']
            if at['confidence'] == 'Very High':
                score += 8
            elif at['confidence'] == 'High':
                score += 4
            
            if score > max_scores['team_total']:
                max_scores['team_total'] = score
                best_bets['team_total'] = {
                    'match': f"{r['home_team']} vs {r['away_team']}",
                    'prediction': f"{r['away_team']} Over 0.5 Goals",
                    'probability': at['over_0.5'],
                    'confidence': at['confidence'],
                    'stars': '⭐⭐⭐⭐⭐' if at['over_0.5'] > 90 else '⭐⭐⭐⭐' if at['over_0.5'] > 80 else '⭐⭐⭐',
                    'xg': r['expected_goals']['away'],
                    'team': r['away_team'],
                    'league': r['league']
                }
        
        # BTTS
        btts = r['btts']
        if btts['prediction'] == 'YES':
            score = btts['probability']
            if btts['confidence'] == 'High':
                score += 5
            
            if score > max_scores['btts']:
                max_scores['btts'] = score
                best_bets['btts'] = {
                    'match': f"{r['home_team']} vs {r['away_team']}",
                    'prediction': 'Both Teams to Score - YES',
                    'probability': btts['probability'],
                    'confidence': btts['confidence'],
                    'stars': '⭐⭐⭐⭐' if btts['probability'] > 75 else '⭐⭐⭐',
                    'xg': r['expected_goals']['total'],
                    'league': r['league']
                }
    
    return best_bets

def display_best_bets(best_bets, league):
    """Display best bets - SINGLE team to score."""
    
    # Overall top pick
    all_bets = [(k, v) for k, v in best_bets.items() if v is not None]
    if all_bets:
        top = max(all_bets, key=lambda x: x[1]['probability'])
        
        print(f"\n🏆 TOP PICK:")
        print(f"   Match: {top[1]['match']}")
        print(f"   Bet: {top[1]['prediction']}")
        print(f"   {top[1]['stars']} {top[1]['probability']}% [{top[1]['confidence']}]")
    
    # Show all categories
    print(f"\n📋 ALL BEST BETS:")
    
    if best_bets['over_1_5']:
        b = best_bets['over_1_5']
        print(f"\n   🔥 Best Over 1.5 Match Goals:")
        print(f"      {b['match']}")
        print(f"      {b['stars']} {b['probability']}% (Expected: {b['xg']} goals)")
    else:
        print(f"\n   🔥 Best Over 1.5 Match Goals: None (low confidence)")
    
    # CHANGED: Single best team to score
    if best_bets['team_total']:
        b = best_bets['team_total']
        print(f"\n   ⚽ Best Team to Score:")
        print(f"      {b['team']} ({b['match']})")
        print(f"      {b['stars']} {b['probability']}% (xG: {b['xg']})")
    else:
        print(f"\n   ⚽ Best Team to Score: None (low confidence)")
    
    if best_bets['btts']:
        b = best_bets['btts']
        print(f"\n   🤝 Best BTTS:")
        print(f"      {b['match']}")
        print(f"      {b['stars']} {b['probability']}%")
    else:
        print(f"\n   🤝 Best BTTS: None (low confidence)")

def display_final_summary(all_results):
    """Display complete final summary after QUIT."""
    
    if not all_results:
        print("\n⚠️  No matches analyzed")
        return
    
    print(f"\n\n{'#'*90}")
    print(f"{'#'*90}")
    print(f"##{'':^86}##")
    print(f"##{'FINAL SUMMARY - ALL LEAGUES':^86}##")
    print(f"##{'':^86}##")
    print(f"{'#'*90}")
    print(f"{'#'*90}")
    
    # Find overall best bets
    overall_best = find_best_bets(all_results)
    
    print(f"\n🥇 TOP PICKS ACROSS ALL LEAGUES:")
    
    if overall_best['over_1_5']:
        b = overall_best['over_1_5']
        print(f"\n   1️⃣  OVER 1.5 MATCH GOALS:")
        print(f"      {b['match']} [{b['league']}]")
        print(f"      {b['stars']} {b['probability']}% [{b['confidence']}] - {b['xg']} xG")
    
    # CHANGED: Single best team to score
    if overall_best['team_total']:
        b = overall_best['team_total']
        print(f"\n   2️⃣  TEAM TO SCORE:")
        print(f"      {b['team']} [{b['league']}]")
        print(f"      {b['stars']} {b['probability']}% [{b['confidence']}] - {b['xg']} xG")
    
    # High confidence lists
    print(f"\n{'='*90}")
    print("🔥 ALL HIGH CONFIDENCE PICKS")
    print(f"{'='*90}")
    
    # Over 1.5
    high_over = [r for r in all_results if r['over_1_5']['confidence'] in ['Very High', 'High'] 
                 and r['over_1_5']['prediction'] == 'Over 1.5']
    
    if high_over:
        print(f"\n📊 OVER 1.5 MATCH GOALS ({len(high_over)} picks):")
        for r in sorted(high_over, key=lambda x: x['over_1_5']['over_1.5'], reverse=True):
            data_mark = "✓" if r['using_real_data'] else "~"
            print(f"   {r['over_1_5']['stars']} {r['home_team']} vs {r['away_team']}: "
                  f"{r['over_1_5']['over_1.5']}% [{r['league']}] {data_mark}")
    else:
        print(f"\n📊 OVER 1.5 MATCH GOALS: No high confidence picks")
    
    # Team totals - ALL high confidence teams
    high_teams = []
    for r in all_results:
        if r['home_total']['confidence'] in ['Very High', 'High'] and r['home_total']['prediction'] in ['YES', 'YES (Moderate)']:
            high_teams.append({
                'team': r['home_team'],
                'prob': r['home_total']['over_0.5'],
                'match': f"{r['home_team']} vs {r['away_team']}",
                'league': r['league'],
                'xg': r['expected_goals']['home']
            })
        if r['away_total']['confidence'] in ['Very High', 'High'] and r['away_total']['prediction'] in ['YES', 'YES (Moderate)']:
            high_teams.append({
                'team': r['away_team'],
                'prob': r['away_total']['over_0.5'],
                'match': f"{r['home_team']} vs {r['away_team']}",
                'league': r['league'],
                'xg': r['expected_goals']['away']
            })
    
    if high_teams:
        print(f"\n⚽ TEAM TO SCORE OVER 0.5 ({len(high_teams)} picks):")
        for t in sorted(high_teams, key=lambda x: x['prob'], reverse=True):
            stars = '⭐⭐⭐⭐⭐' if t['prob'] > 90 else '⭐⭐⭐⭐' if t['prob'] > 85 else '⭐⭐⭐'
            print(f"   {stars} {t['team']}: {t['prob']}% (xG: {t['xg']}) [{t['match']}]")
    else:
        print(f"\n⚽ TEAM TO SCORE OVER 0.5: No high confidence picks")
    
    # BTTS
    btts_picks = [r for r in all_results if r['btts']['confidence'] == 'High' and r['btts']['prediction'] == 'YES']
    if btts_picks:
        print(f"\n🤝 BOTH TEAMS TO SCORE ({len(btts_picks)} picks):")
        for r in sorted(btts_picks, key=lambda x: x['btts']['probability'], reverse=True):
            print(f"   ⭐⭐⭐ {r['home_team']} vs {r['away_team']}: {r['btts']['probability']}% [{r['league']}]")
    else:
        print(f"\n🤝 BOTH TEAMS TO SCORE: No high confidence picks")
    
    # Statistics
    print(f"\n{'='*90}")
    print("📈 ANALYSIS STATISTICS")
    print(f"{'='*90}")
    
    total_matches = len(all_results)
    real_data_count = sum(1 for r in all_results if r['using_real_data'])
    leagues = set(r['league'] for r in all_results)
    
    print(f"\nTotal Matches Analyzed: {total_matches}")
    print(f"Leagues Covered: {len(leagues)}")
    print(f"Real Data Used: {real_data_count}/{total_matches} ({real_data_count/total_matches*100:.1f}%)")
    print(f"High Confidence Over 1.5: {len(high_over)}")
    print(f"High Confidence Team Totals: {len(high_teams)}")
    print(f"High Confidence BTTS: {len(btts_picks)}")
    
    print(f"\n{'='*90}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*90}\n")

def main():
    print("="*90)
    print("🚀 PROFESSIONAL BETTING ANALYZER")
    print("Dixon-Coles Model + Ensemble Prediction + League Data")
    print("="*90)
    print("\nShows: Over 1.5 Match Goals + Best Single Team Total")
    print("Format: Team A vs Team B (home draw away)")
    print("Type QUIT to see final summary\n")
    
    predictor = EnsemblePredictor()
    all_results = []
    
    while True:
        league = input("\nLeague (or QUIT): ").strip()
        
        if league.upper() == 'QUIT':
            break
        
        matches = []
        print(f"Matches (DONE when finished):")
        while True:
            line = input().strip()
            if line.upper() == 'DONE':
                break
            if line:
                matches.append(line)
        
        if matches:
            results = predictor.analyze_matches(matches, league)
            all_results.extend(results)
            
            # Display predictions IMMEDIATELY
            display_match_predictions(results, league)
    
    # Display final summary after QUIT
    if all_results:
        display_final_summary(all_results)
        
        # Optional save
        save_choice = input("\nSave results to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            filename = save_results(all_results)
            print(f"💾 Saved to: {filename}")

if __name__ == '__main__':
    main()

