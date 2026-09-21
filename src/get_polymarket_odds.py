import requests
import json

TEAM_QUERY = "Rams Giants"
NOISE_WORDS = ["Props", "Spread", "Total", "Touchdown", "Quarter", "Scorer"]


def is_moneyline_event(title):
    for word in NOISE_WORDS:
        if word in title:
            return False
    return True


if __name__ == "__main__":
    url = "https://gamma-api.polymarket.com/public-search"
    params = {"q": TEAM_QUERY}

    response = requests.get(url, params=params)
    data = response.json()
    events = data["events"]
    game_event = None
    for e in events:
        if " vs. " in e["title"] and is_moneyline_event(e["title"]) and not e["closed"]:
            game_event = e

    moneyline_market = None
    for m in game_event["markets"]:
        if m["question"] == game_event["title"]:
            moneyline_market = m

    outcomes = json.loads(moneyline_market["outcomes"])
    prices = json.loads(moneyline_market["outcomePrices"])

    print(f"{game_event['title']}:")
    for team, price in zip(outcomes, prices):
        print(f"  {team}: {float(price):.1%} implied probability")