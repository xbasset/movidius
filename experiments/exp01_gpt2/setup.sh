#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
bash scripts/setup.sh
.venv/bin/python -m experiments.exp01_gpt2.scripts.fetch_gpt2
if [ ! -f assets/wikitext2-test.txt ]; then
  curl -fL https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt \
    -o assets/wikitext2-test.txt
fi
