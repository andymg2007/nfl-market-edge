"""
load_data.py

Phase 1: pull historical NFL game data and clean it up so later
steps (features, model, backtest) have something reliable to use.

Data source: nflverse's public games.csv — a community-maintained,
free NFL data project. Includes every game since 1999: final scores,
week, date, and the betting lines available at kickoff.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def load_games(min_season: int = 2010) -> pd.DataFrame:
    """Load NFL game-level data and keep only played games from min_season on."""
    df = pd.read_csv(GAMES_URL)

    # Keep only games that have actually been played (drop future/blank scores)
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["season"] >= min_season].copy()

    # Convenience columns we'll want later for features/labels
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["point_diff"] = df["home_score"] - df["away_score"]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    games = load_games()
    out_path = DATA_DIR / "games_clean.csv"
    games.to_csv(out_path, index=False)

    print(f"Loaded {len(games)} played games from {games['season'].min()}"
          f" to {games['season'].max()}")
    print(f"Saved to {out_path}")
    print(games[["season", "week", "home_team", "away_team",
                 "home_score", "away_score", "home_win"]].tail())