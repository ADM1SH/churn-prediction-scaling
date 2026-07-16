#!/usr/bin/env bash
# End-to-end demo: build the C++ server, train the churn model, run the live
# telemetry pipeline (server + webhook receiver + event generator), then
# score the captured telemetry log and fire webhooks for at-risk players.
set -euo pipefail
cd "$(dirname "$0")"

PY="../.venv10projects/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(dirname "$0")/../.venv10projects/bin/python"
fi

mkdir -p data models logs

echo "== 1. build C++ telemetry server =="
g++ -O2 -std=c++17 -pthread server/telemetry_server.cpp -o server/telemetry_server

echo "== 2. generate synthetic training batch + train XGBoost model =="
"$PY" ml/synthetic_data.py
"$PY" ml/train_model.py

echo "== 3. start webhook receiver (background) =="
rm -f logs/webhook.log
"$PY" webhook/receiver.py 8090 > logs/webhook.log 2>&1 &
WEBHOOK_PID=$!
trap 'kill $WEBHOOK_PID $SERVER_PID 2>/dev/null || true' EXIT

echo "== 4. start C++ telemetry server (background) =="
rm -f data/telemetry_log.csv logs/server.log
./server/telemetry_server 9000 data/telemetry_log.csv > logs/server.log 2>&1 &
SERVER_PID=$!
sleep 0.5

echo "== 5. run event-generator client (simulates 6 players over the socket) =="
"$PY" client/generator.py 127.0.0.1 9000
sleep 0.3

echo "== 6. score captured telemetry log, fire webhooks for at-risk players =="
"$PY" ml/churn_scorer.py data/telemetry_log.csv http://127.0.0.1:8090/webhook 0.7

echo "== done =="
echo "-- telemetry_log.csv --"
cat data/telemetry_log.csv
echo "-- webhook.log --"
cat logs/webhook.log
