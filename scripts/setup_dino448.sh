#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_myriad.sh
.venv/bin/python -m scripts.fetch_dino
.venv/bin/python -m scripts.fetch_mvtec
.venv/bin/python -m scripts.export_dino --size 448 --stop features
.venv/bin/python -m scripts.export_anomalydino448 --trials 3
.venv/bin/python -m scripts.export_anomalydino448 --trials 1
.venv/bin/python -m scripts.check_anomaly_metrics
