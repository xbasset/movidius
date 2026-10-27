#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
ov_src="$PWD/third_party/openvino-2020.3/inference-engine"
ov_lib="$PWD/third_party/openvino-2020.3/bin/aarch64/Release/lib"
clang++ -arch arm64 -std=c++14 -O2 -dynamiclib \
  -I"$ov_src/include" -I"$ov_src/src/legacy_api/include" \
  movidius_lab/myriad_bridge.cpp -L"$ov_lib" -linference_engine -linference_engine_legacy \
  -Wl,-rpath,"$ov_lib" -o build/libcrivo_myriad.dylib
file build/libcrivo_myriad.dylib
