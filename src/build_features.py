import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

START_ELO = 1500
K = 20
HOME_ADVANTAGE = 65
SEASON_REGRESSION = 1/3

def expected_win_probability(elo_a, elo_b):
    probability = 1 / (1+10**((elo_b - elo_a) / 400))
    return probability

def mov_multiplier(point_diff, elo_diff_winner):
    multiplier = np.log(abs(point_diff) + 1) * (2.2 / (elo_diff_winner * 0.001 + 2.2))
    return multiplier

def build_elo_features(games):
    games = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    elo = {}  # dictionary: team abbreviation -> current rating
    current_season = None

    home_elo_pre = []
    away_elo_pre = []

    for row in games.itertuples():
        team_h = row.home_team
        team_a = row.away_team

        if row.season != current_season:
            current_season = row.season
            for team in elo:
                elo[team] = elo[team] + (SEASON_REGRESSION) * (START_ELO - elo[team])
        elo.setdefault(team_h, START_ELO)
        elo.setdefault(team_a, START_ELO)

        h_pre = elo[team_h]
        a_pre = elo[team_a]
        home_elo_pre.append(h_pre)
        away_elo_pre.append(a_pre)

        exp_home = expected_win_probability(h_pre+HOME_ADVANTAGE, a_pre)
        if row.home_score > row.away_score:
            actual_home = 1.0
        elif row.home_score < row.away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        point_diff = row.home_score - row.away_score
        if point_diff > 0:
            winner_elo_diff = (h_pre + HOME_ADVANTAGE) - a_pre
        elif point_diff < 0:
            winner_elo_diff = (a_pre - (h_pre + HOME_ADVANTAGE))
        else:
            winner_elo_diff = 0
        mult = mov_multiplier(point_diff, winner_elo_diff)

        shift = K * mult * (actual_home - exp_home)
        elo[team_h] = shift + h_pre
        elo[team_a] = a_pre - shift

    games["home_elo_pre"] = home_elo_pre
    games["away_elo_pre"] = away_elo_pre
    games["elo_diff"] = games["home_elo_pre"] - games["away_elo_pre"]
    games["home_win_prob_elo"] = expected_win_probability(games["home_elo_pre"]+HOME_ADVANTAGE, games["away_elo_pre"])
    return games

if __name__ == "__main__":
    games = pd.read_csv(DATA_DIR / "games_clean.csv")
    games = build_elo_features(games)
    games.to_csv(DATA_DIR / "games_features.csv", index=False)
    print(f"Saved {len(games)} games with Elo features")
    print(games[["gameday", "home_team", "away_team", "home_elo_pre", "away_elo_pre", "home_win_prob_elo"]].tail())
    games["prob_bucket"] = pd.cut(games["home_win_prob_elo"], bins=np.arange(0, 1.1, 0.1))
    calibration = games.groupby("prob_bucket", observed=True)["home_win"].agg(["mean", "count"])
    print(calibration)