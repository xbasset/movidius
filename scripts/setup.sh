#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
mkdir -p third_party build results assets
if [ ! -d .venv ]; then uv venv --python 3.12 .venv; fi
uv pip install --python .venv/bin/python -r requirements-lock.txt
if [ ! -d third_party/ncsdk ]; then
  git clone --branch macos-V1 https://github.com/milosgajdos/ncsdk.git third_party/ncsdk
fi
git -C third_party/ncsdk checkout 38d0dfb9659fb0c4a1c5166471a9364d784262d1
if [ ! -d third_party/libusb ]; then
  git clone --depth 1 --branch v1.0.29 https://github.com/libusb/libusb.git third_party/libusb
fi
git -C third_party/libusb checkout 15a7ebb4d426c5ce196684347d2b7cafad862626
if [ ! -f .local/lib/libusb-1.0.a ]; then
  task_root="$PWD"
  (
    cd third_party/libusb
    ./bootstrap.sh
    # Avoid an unavailable weak pipe2 symbol with this macOS SDK/runtime pair.
    ./configure --prefix="$task_root/.local" --disable-shared --enable-static \
      --disable-tests-build ac_cv_func_pipe2=no
    make -j4
    make install
  )
fi
bash scripts/build_runtime.sh
.venv/bin/python -m scripts.fetch_gpt2
if [ ! -f assets/wikitext2-test.txt ]; then
  curl -fL https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt \
    -o assets/wikitext2-test.txt
fi
