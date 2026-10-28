"""Transport to a MYRIAD-only graph. No host model operations or fallback."""

import ctypes as C
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from movidius_lab.paths import ROOT
LIBDIR = ROOT / "third_party/openvino-2020.3/bin/aarch64/Release/lib"


class Myriad:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads(self.path.with_suffix(".json").read_text())
        self.inputs = {x.attrib["name"] for x in ET.parse(self.path.with_suffix(".xml")).findall(".//layer")
                       if x.attrib["type"] == "Input"}
        # Keep transport code loaded until its worker threads have stopped.
        self.plugin = C.CDLL(str(LIBDIR / "libmyriadPlugin.dylib"))
        self.lib = C.CDLL(str(ROOT / "build/libcrivo_myriad.dylib"))
        self.lib.crivo_open.argtypes = [C.c_char_p, C.c_char_p, C.c_void_p, C.c_size_t]
        self.lib.crivo_open.restype = C.c_void_p
        self.lib.crivo_input.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
        self.lib.crivo_infer.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_double), C.c_void_p, C.c_size_t]
        self.lib.crivo_profile.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_size_t]
        self.lib.crivo_close.argtypes = [C.c_void_p]
        self.error = C.create_string_buffer(8192)
        start = time.perf_counter()
        self.handle = self.lib.crivo_open(str(LIBDIR / "plugins.xml").encode(),
                                          str(self.path.with_suffix(".xml")).encode(), self.error, len(self.error))
        if not self.handle:
            raise RuntimeError(self.error.value.decode())
        self.load_ms = (time.perf_counter() - start) * 1000
        self.last_infer_ms = None

    def check(self, status):
        if status:
            raise RuntimeError(self.error.value.decode())

    def set_input(self, name, value):
        value = np.ascontiguousarray(value)
        self.check(self.lib.crivo_input(self.handle, name.encode(), value.ctypes.data,
                                       value.nbytes, self.error, len(self.error)))

    def infer(self, ids, opt_positions):
        n, k = self.meta["tokens"], self.meta["options"]
        if len(ids) > n or len(opt_positions) != k:
            raise ValueError(f"Graph expects <= {n} tokens and exactly {k} options; no truncation is permitted")
        tokens = np.full((1, n), 50283, dtype=np.int32)
        tokens[0, :len(ids)] = ids
        self.set_input("input_ids", tokens)
        if "padding_mask" in self.inputs:
            mask = np.full((1, 1, 1, n), -10000, dtype=np.float16)
            mask[..., :len(ids)] = 0
            self.set_input("padding_mask", mask)
        if "opt_positions" in self.inputs:
            self.set_input("opt_positions", np.asarray(opt_positions, dtype=np.int32))
        return self.execute()

    def infer_tensors(self, inputs):
        if set(inputs) != self.inputs:
            raise ValueError(f"Expected graph inputs {sorted(self.inputs)}")
        for name, value in inputs.items():
            self.set_input(name, value)
        return self.execute()

    def execute(self):
        result = np.empty(self.meta["output_shape"], dtype=np.float32)
        duration = C.c_double()
        self.check(self.lib.crivo_infer(self.handle, result.ctypes.data, result.size,
                                       C.byref(duration), self.error, len(self.error)))
        self.last_infer_ms = duration.value
        return result

    def profile(self, path):
        self.check(self.lib.crivo_profile(self.handle, str(path).encode(), self.error, len(self.error)))

    def close(self):
        if self.handle:
            self.lib.crivo_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
