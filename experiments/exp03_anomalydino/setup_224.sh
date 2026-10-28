#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
bash scripts/setup_myriad.sh
.venv/bin/python -m experiments.exp03_anomalydino.scripts.fetch_dino
.venv/bin/python -m experiments.exp03_anomalydino.scripts.export_dino --stop features
.venv/bin/python -m experiments.exp03_anomalydino.scripts.export_dino --stop anomaly
.venv/bin/python -m experiments.exp03_anomalydino.scripts.verify_dino_reference
