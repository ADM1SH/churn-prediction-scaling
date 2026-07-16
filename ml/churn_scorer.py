"""Scores aggregated player session rows from the telemetry log with the
trained XGBoost model, and fires a webhook for any player whose predicted
churn probability crosses the threshold, carrying a suggested action.

Usage: .venv10projects/bin/python ml/churn_scorer.py [telemetry_csv] [webhook_url] [threshold]
"""
import json
import sys
import urllib.request

import pandas as pd
from xgboost import XGBClassifier

TELEMETRY_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/telemetry_log.csv"
WEBHOOK_URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8090/webhook"
THRESHOLD = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
MODEL_PATH = "models/churn_model.json"

FEATURE_COLS = [
    "session_duration",
    "movement_events",
    "deaths",
    "menu_opens",
    "menu_time",
    "time_since_last_death",
]


def suggest_action(row, prob):
    # ponytail: simple rule over the same features, not a second model.
    # Add a bandit/policy model when more than two actions are needed.
    if row["deaths"] >= 4 or row["menu_time"] > row["session_duration"] * 0.3:
        return "lower_difficulty"
    return "grant_reward"


def fire_webhook(player_id, prob, action):
    payload = {
        "player_id": player_id,
        "churn_probability": round(float(prob), 4),
        "suggested_action": action,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        WEBHOOK_URL, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        resp_body = resp.read().decode()
    print(f"[churn_scorer] fired webhook for {player_id}: {json.dumps(payload)} "
          f"-> response={resp_body}")


def main():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    df = pd.read_csv(TELEMETRY_PATH)
    X = df[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]
    df["churn_probability"] = probs

    print(f"[churn_scorer] scored {len(df)} rows from {TELEMETRY_PATH}, "
          f"threshold={THRESHOLD}")

    fired = 0
    for _, row in df.iterrows():
        prob = row["churn_probability"]
        if prob >= THRESHOLD:
            action = suggest_action(row, prob)
            fire_webhook(row["player_id"], prob, action)
            fired += 1

    print(f"[churn_scorer] done: {fired}/{len(df)} rows crossed threshold and fired webhooks")


if __name__ == "__main__":
    main()
