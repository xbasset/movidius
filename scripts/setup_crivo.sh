#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_myriad.sh
.venv/bin/python -m scripts.fetch_crivo

.venv/bin/python -m scripts.export_crivo --tokens 64 --options 3 --output assets/crivo-ir/demo
.venv/bin/python -m scripts.export_crivo --tokens 960 --options 77 --output assets/crivo-ir/banking77
