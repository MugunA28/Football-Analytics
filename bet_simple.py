<<<<<<< HEAD
#!/usr/bin/env python3
"""
SIMPLE BETTING PREDICTOR
Clean, actionable predictions only
"""

import sys
sys.path.insert(0, 'src')

from prediction.ensemble_predictor import EnsemblePredictor

def display_simple_predictions(results, league):
    """Display simple, clean predictions."""
    print(f"\n{'='*90}")
    print(f"🎯 {league.upper()} - PREDICTIONS")
    print(f"{'='*90}\n")
    
    for i, m in enumerate(results, 1):
        xg = m['expected_goals']
        o15 = m['over_1_5']
        ht = m['home_total']
        at = m['away_total']
        
        print(f"MATCH {i}: {m['home_team']} vs {m['away_team']}")
        print(f"{'─'*90}")
        
        # Main prediction
        over_decision = "✅ YES" if o15['prediction'] == 'Over 1.5' else "❌ NO"
        print(f"\n   📊 OVER 1.5 GOALS: {over_decision} ({o15['over_1.5']}% confidence)")
        
        if o15['prediction'] == 'Over 1.5':
            print(f"\n   ⚽ TEAMS TO SCORE OVER 0.5:")
            
            # Show which teams should score
            if ht['prediction'] in ['YES', 'YES (Moderate)']:
                print(f"      ✅ {m['home_team']}: YES ({ht['over_0.5']}%)")
            else:
                print(f"      ⚠️  {m['home_team']}: MAYBE ({ht['over_0.5']}%)")
            
            if at['prediction'] in ['YES', 'YES (Moderate)']:
                print(f"      ✅ {m['away_team']}: YES ({at['over_0.5']}%)")
            else:
                print(f"      ⚠️  {m['away_team']}: MAYBE ({at['over_0.5']}%)")
        else:
            print(f"\n   ❌ Under 1.5 predicted - low-scoring match expected")
        
        print(f"\n   💡 Expected Score: {xg['home']:.1f} - {xg['away']:.1f}\n")
    
    # Summary
    print(f"{'='*90}")
    print(f"💎 RECOMMENDED BETS")
    print(f"{'='*90}\n")
    
    over_picks = [r for r in results if r['over_1_5']['prediction'] == 'Over 1.5' 
                  and r['over_1_5']['over_1.5'] >= 65]
    
    if over_picks:
        for r in sorted(over_picks, key=lambda x: x['over_1_5']['over_1.5'], reverse=True):
            print(f"✅ {r['home_team']} vs {r['away_team']}")
            print(f"   BET: Over 1.5 Goals ({r['over_1_5']['over_1.5']}%)")
            
            teams_to_score = []
            if r['home_total']['prediction'] in ['YES', 'YES (Moderate)']:
                teams_to_score.append(f"{r['home_team']} ({r['home_total']['over_0.5']}%)")
            if r['away_total']['prediction'] in ['YES', 'YES (Moderate)']:
                teams_to_score.append(f"{r['away_team']} ({r['away_total']['over_0.5']}%)")
            
            if teams_to_score:
                print(f"   ALSO: {' & '.join(teams_to_score)} to score")
            print()
    else:
        print("⚠️  No high-confidence Over 1.5 picks\n")

def main():
    print("="*90)
    print("🎯 SIMPLE BETTING PREDICTOR")
    print("="*90)
    print("\nFormat: Team A vs Team B (home draw away)\n")
    
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
            print(f"\n⏳ Analyzing...")
            results = predictor.analyze_matches(matches, league)
            display_simple_predictions(results, league)
            all_results.extend(results)
    
    # Final summary
    if all_results:
        print(f"\n{'='*90}")
        print(f"📊 FINAL SUMMARY - {len(all_results)} MATCHES ANALYZED")
        print(f"{'='*90}\n")
        
        over_count = sum(1 for r in all_results if r['over_1_5']['prediction'] == 'Over 1.5')
        high_conf = sum(1 for r in all_results if r['over_1_5']['over_1.5'] >= 70)
        
        print(f"Over 1.5 predictions: {over_count}/{len(all_results)}")
        print(f"High confidence (70%+): {high_conf}")
        print(f"\n{'='*90}\n")

if __name__ == '__main__':
    main()

=======
# bet_simple.py content here... (actual content needs to be read) 
# This file contains simple betting strategies.


# further implementation here...
>>>>>>> 4cab15094e8a14ad692f4b9d37983ad988eb16a1
