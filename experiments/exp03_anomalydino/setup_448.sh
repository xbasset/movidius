#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
bash scripts/setup_myriad.sh
.venv/bin/python -m experiments.exp03_anomalydino.scripts.fetch_dino
.venv/bin/python -m experiments.exp03_anomalydino.scripts.fetch_mvtec
.venv/bin/python -m experiments.exp03_anomalydino.scripts.export_dino --size 448 --stop features
.venv/bin/python -m experiments.exp03_anomalydino.scripts.export_anomalydino448 --trials 3
.venv/bin/python -m experiments.exp03_anomalydino.scripts.export_anomalydino448 --trials 1
.venv/bin/python -m experiments.exp03_anomalydino.scripts.check_anomaly_metrics
