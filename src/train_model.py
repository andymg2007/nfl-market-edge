import pandas as pd
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.metrics import accuracy_score, brier_score_loss

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
df = pd.read_csv(DATA_DIR / "games_features.csv")

df["rest_diff"] = df["home_rest"] - df["away_rest"]
train = df[df["season"] < 2025]
test = df[df["season"] >= 2025]

features = ["elo_diff", "rest_diff"]

X_train = train[features]
y_train = train["home_win"]

X_test = test[features]
y_test = test["home_win"]

model = LogisticRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, preds)
brier = brier_score_loss(y_test, probs)

print("accuracy:", acc)
print("brier score:", brier)
elo_brier = brier_score_loss(y_test, test["home_win_prob_elo"])
print("elo-only brier score:", elo_brier)