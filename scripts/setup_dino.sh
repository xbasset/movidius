#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_myriad.sh
.venv/bin/python -m scripts.fetch_dino
.venv/bin/python -m scripts.export_dino --stop features
.venv/bin/python -m scripts.export_dino --stop anomaly
.venv/bin/python -m scripts.verify_dino_reference
