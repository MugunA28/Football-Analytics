#!/usr/bin/env python3
"""
VALUE BET DETECTOR
Combines Dixon-Coles statistical model with bookmaker odds analysis
to identify value bets with an edge threshold of >15%.

Input format:
    Team A vs Team B (home_o0.5 away_o0.5 over_1.5)

Where:
    home_o0.5  = Bookmaker odds for home team Over 0.5 Goals
    away_o0.5  = Bookmaker odds for away team Over 0.5 Goals
    over_1.5   = Bookmaker odds for Over 1.5 Match Goals

Example:
    > Manchester City vs Arsenal (1.12 1.45 1.40)
    > Liverpool vs Sheffield Utd (1.08 1.60 1.18)
    > DONE
"""

import sys
import os

sys.path.insert(0, 'src')

from prediction.value_detector import ValueDetector


def main():
    print("=" * 82)
    print("🔥 VALUE BET DETECTOR")
    print("Dixon-Coles Model + Bookmaker Odds Analysis  |  Threshold: >15% Edge")
    print("=" * 82)
    print()
    print("Enter matches (format: Team A vs Team B (home_o0.5 away_o0.5 over_1.5))")
    print("Type DONE to analyse  |  Type QUIT to exit")
    print()

    detector = ValueDetector()
    match_strings = []

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.upper() == "QUIT":
            print("\n👋 Goodbye!")
            return

        if line.upper() == "DONE":
            if not match_strings:
                print("   ⚠️  No matches entered yet. Add a match or type QUIT.\n")
                continue

            print("\n⏳ Analysing matches…")
            results = detector.analyze_matches(match_strings)

            if results:
                output = detector.format_output(results)
                print(output)

                # Auto-save
                saved_file = detector.save_results(results)
                print(f"\n💾 Results saved to: {saved_file}\n")

            # Reset for another round
            match_strings = []
            print("─" * 82)
            print("Enter more matches or type QUIT to exit.")
            print()
        else:
            # Basic validation: check it looks like a valid match string
            if " vs " not in line.lower():
                print(
                    "   ⚠️  Unrecognised input. "
                    "Expected format: Team A vs Team B (h_odds a_odds o1.5_odds)\n"
                )
                continue
            match_strings.append(line)
            print(f"   ✅ Added: {line}")


if __name__ == "__main__":
    main()
