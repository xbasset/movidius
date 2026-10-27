#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup.sh --runtime-only

ov_src="$PWD/third_party/openvino-2020.3"
ov_revision=2fe9b1523058e282ad374db7dc1b3538c7d2dd27
if [ ! -d "$ov_src" ]; then
  git clone --depth 1 --branch 2020.3.0 https://github.com/openvinotoolkit/openvino.git "$ov_src"
fi
test "$(git -C "$ov_src" rev-parse HEAD)" = "$ov_revision"
git -C "$ov_src" submodule update --init ngraph inference-engine/thirdparty/ade
ov_patch="$PWD/scripts/patches/openvino-macos-arm64.patch"
if git -C "$ov_src" apply --check "$ov_patch" 2>/dev/null; then
  git -C "$ov_src" apply "$ov_patch"
else
  git -C "$ov_src" apply --reverse --check "$ov_patch"
fi

cmake -S "$ov_src" -B build/openvino-arm64 \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/scripts/macos-arm64.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DENABLE_MKL_DNN=OFF -DENABLE_CLDNN=OFF -DENABLE_GNA=OFF \
  -DENABLE_OPENCV=OFF -DENABLE_TESTS=OFF -DENABLE_SAMPLES=OFF \
  -DENABLE_PYTHON=OFF -DENABLE_C=OFF -DENABLE_VPU=ON -DENABLE_MYRIAD=ON \
  -DTHREADING=SEQ -DTREAT_WARNING_AS_ERROR=OFF \
  -DNGRAPH_UNIT_TEST_ENABLE=OFF -DNGRAPH_UNIT_TEST_OPENVINO_ENABLE=OFF \
  -DNGRAPH_ONNX_IMPORT_ENABLE=OFF -DNGRAPH_TEST_UTIL_ENABLE=OFF \
  -DNGRAPH_INTERPRETER_ENABLE=OFF \
  -DLIBUSB_LIBRARY="$PWD/.local/lib/libusb-1.0.a" \
  -DLIBUSB_INCLUDE_DIR="$PWD/.local/include/libusb-1.0" \
  -DCMAKE_SHARED_LINKER_FLAGS='-framework IOKit -framework CoreFoundation -framework Security'
cmake --build build/openvino-arm64 --target myriadPlugin -j8
bash scripts/build_myriad_bridge.sh

.venv/bin/python - <<'PY'
from movidius_lab.crivo import ROOT, sha256
firmware = ROOT / 'third_party/openvino-2020.3/bin/aarch64/Release/lib/usb-ma2450.mvcmd'
assert sha256(firmware) == '6cf079e614d0fa68381ec29d0b30f44766cf3cf62355620a91912128f0bfdaac'
print('Myriad 2 firmware SHA256 verified.')
PY
