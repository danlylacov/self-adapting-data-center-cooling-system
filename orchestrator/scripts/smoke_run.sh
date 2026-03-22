#!/usr/bin/env bash
set -uo pipefail
BASE="${ORCHESTRATOR_BASE:-http://127.0.0.1:8030}"
echo "GET $BASE/health"
curl -sS "$BASE/health" | head -c 500 || true
echo ""
echo "POST $BASE/run (short)"
curl -sS -X POST "$BASE/run" \
  -H 'Content-Type: application/json' \
  -d '{
    "scenario": {
      "steps": 30,
      "deltaTime": 10,
      "setpoint": 22,
      "fanSpeed": 65,
      "coolingMode": "mixed",
      "meanLoad": 0.55,
      "stdLoad": 0.12,
      "outsideTemp": 24,
      "useDatasetLoad": false
    },
    "controlIntervalSteps": 5,
    "safetyMaxPOverheat": 0.2,
    "failOnMlUnavailable": false
  }' | head -c 2000
echo ""
