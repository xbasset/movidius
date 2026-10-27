#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build/mvnc
src="$PWD/third_party/ncsdk/api/src"
clang -O2 -dynamiclib -fPIC -pthread -include string.h \
  -I"$src" -I"$src/../include" -I"$PWD/.local/include/libusb-1.0" \
  "$src/usb_boot.c" "$src/usb_link_vsc.c" "$src/mvnc_api.c" \
  "$PWD/.local/lib/libusb-1.0.a" \
  -framework IOKit -framework CoreFoundation -framework Security \
  -o build/libmvnc.dylib
cp "$src/mvnc/MvNCAPI.mvcmd" build/mvnc/
file build/libmvnc.dylib
