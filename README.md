# Player Churn Prediction & Dynamic Scaling

[![C++ / Python / XGBoost](https://img.shields.io/ML-XGBoost_%7C_C%2B%2B_Telemetry-orange.svg)](https://github.com/ADM1SH/churn-prediction-scaling)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/ADM1SH/churn-prediction-scaling)](https://github.com/ADM1SH/churn-prediction-scaling/issues)


A minimal but real pipeline: a C++ TCP server ingests player telemetry, an
XGBoost model predicts churn probability from the aggregated session
features, and a scorer fires a webhook with a suggested action for
players about to quit.

## What it does

1. **C++ telemetry server** (`server/telemetry_server.cpp`) : plain POSIX-socket
   TCP server. Accepts newline-delimited JSON events per connection
   (`session_start`, `movement`, `death`, `menu_open`, `menu_close`,
   `session_end`), aggregates them in memory per session (deaths, movement
   count, menu-idle time, session duration, time since last death), and
   appends one CSV feature row per session on `session_end` (or on
   disconnect, if the client drops mid-session).
2. **Event-generator client** (`client/generator.py`) : simulates 6 players
   over real sockets: 3 "at-risk" players whose sessions get shorter, whose
   menu-idle time rises, and whose deaths cluster right before they quit
   across 4 sequential sessions; 3 "healthy" players with stable session
   length, low menu time, and few well-spread deaths.
3. **Churn model + webhook trigger** (`ml/`) : trains an XGBoost classifier
   on a larger synthetic labeled batch (same feature schema as the live
   log), reports held-out accuracy/AUC, then scores rows from the real
   telemetry log and POSTs a webhook (`webhook/receiver.py`, a local
   `http.server` mock) for any player whose churn probability crosses 0.7,
   carrying a suggested action (`lower_difficulty` or `grant_reward`).

## Architecture

```
client/generator.py  --TCP JSON lines-->  server/telemetry_server (C++)
                                                |
                                                v
                                     data/telemetry_log.csv
                                                |
                                                v
ml/synthetic_data.py -> ml/train_model.py -> models/churn_model.json
                                                |
                                                v
                                      ml/churn_scorer.py
                                                |
                                     HTTP POST (churn_prob >= 0.7)
                                                v
                                     webhook/receiver.py (mock endpoint)
```

## Build & run

Requires g++ (C++17) and the shared venv
`../.venv10projects` (numpy, pandas, scikit-learn, xgboost already installed).

One command, correct order baked in:

```bash
./run_demo.sh
```

Or step by step (from this directory):

```bash
PY=../.venv10projects/bin/python

# 1. build the server
g++ -O2 -std=c++17 -pthread server/telemetry_server.cpp -o server/telemetry_server

# 2. train the model (synthetic batch -> XGBoost -> models/churn_model.json)
$PY ml/synthetic_data.py
$PY ml/train_model.py

# 3. start the mock webhook receiver (separate terminal / background)
$PY webhook/receiver.py 8090 &

# 4. start the telemetry server (separate terminal / background)
./server/telemetry_server 9000 data/telemetry_log.csv &

# 5. run the event generator (feeds the live server over a real socket)
$PY client/generator.py 127.0.0.1 9000

# 6. score the captured telemetry log and fire webhooks for at-risk players
$PY ml/churn_scorer.py data/telemetry_log.csv http://127.0.0.1:8090/webhook 0.7
```

## Captured output (actual run)

Model, held out on a 20% split of the 4000-row synthetic batch:

```
[train_model] held-out accuracy=0.8638 auc=0.9121 (train=3200, test=800)
```

Live socket demo produced 24 session rows (6 players x 4 sessions) in
`data/telemetry_log.csv`. Scoring that log against the trained model fired
two real webhook calls:

```
[churn_scorer] fired webhook for churn_jax: {"player_id": "churn_jax", "churn_probability": 0.8461, "suggested_action": "lower_difficulty"} -> response={"status": "received"}
[churn_scorer] fired webhook for churn_kai: {"player_id": "churn_kai", "churn_probability": 0.8811, "suggested_action": "lower_difficulty"} -> response={"status": "received"}
[churn_scorer] done: 2/24 rows crossed threshold and fired webhooks
```

...received on the mock endpoint (`logs/webhook.log`):

```
[webhook-receiver] POST /webhook -> {"player_id": "churn_jax", "churn_probability": 0.8461, "suggested_action": "lower_difficulty"}
[webhook-receiver] POST /webhook -> {"player_id": "churn_kai", "churn_probability": 0.8811, "suggested_action": "lower_difficulty"}
```

## Simplified vs full spec

- **Synthetic telemetry, not real production traffic.** The generator scripts
  realistic-looking churn/healthy patterns instead of replaying real player
  logs. Add when: a real game client can stream production telemetry to the
  server's socket.
- **Threshold-based single-model scoring, not a full online-learning
  pipeline.** One XGBoost model retrained offline, scored in batch against a
  fixed 0.7 cutoff. Add when: model drift monitoring and periodic
  retraining on fresh labeled churn outcomes matter.
- **Local mock webhook, not a real game-server-side hook.** `webhook/receiver.py`
  just logs payloads on localhost. Add when: a live game backend exposes an
  authenticated endpoint to actually apply `lower_difficulty` /
  `grant_reward` to a player's session.

## Support
Submit issues, questions, or bug reports to the GitHub issue tracker:
https://github.com/ADM1SH/churn-prediction-scaling/issues


## Roadmap
* [x] Core architecture and baseline implementation.
* [x] Functional verification and test coverage.
* [ ] Add real-time feature store integration
* [ ] Implement automated retraining pipeline on drift detection


## Contributing
Contributions are welcome.
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/improvement`.
3. Commit your changes: `git commit -m "feat: enhance functionality"`.
4. Push to the branch: `git push origin feature/improvement`.
5. Open a Pull Request.


## Authors and Acknowledgment
* **Adam Anwar** (ADM1SH) - Lead architect and developer.
* Architected by Adam Anwar for player retention analytics in gaming.


## License
Licensed under the MIT License. See `LICENSE` for details.


## Project Status
Operational telemetry and predictive ML service.
