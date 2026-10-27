"""Typed bindings to the NCSDK v1 host API; firmware stays in device RAM."""

import ctypes as C
import plistlib
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def usb_snapshot():
    tree = plistlib.loads(subprocess.check_output(["ioreg", "-p", "IOUSB", "-a", "-l"]))
    found = []

    def visit(node):
        if node.get("idVendor") == 0x03e7:
            found.append({k: node.get(k) for k in ["USB Product Name", "idVendor", "idProduct", "UsbLinkSpeed", "bcdUSB"]})
        for child in node.get("IORegistryEntryChildren", []):
            visit(child)

    for node in (tree if isinstance(tree, list) else [tree]):
        visit(node)
    return found


class NCS:
    def __init__(self, verbose=False):
        self.lib = C.CDLL(str(ROOT / "build/libmvnc.dylib"))
        signatures = {
            "mvncGetDeviceName": [C.c_int, C.c_void_p, C.c_uint],
            "mvncOpenDevice": [C.c_char_p, C.POINTER(C.c_void_p)],
            "mvncCloseDevice": [C.c_void_p],
            "mvncAllocateGraph": [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p, C.c_uint],
            "mvncDeallocateGraph": [C.c_void_p],
            "mvncLoadTensor": [C.c_void_p, C.c_void_p, C.c_uint, C.c_void_p],
            "mvncGetResult": [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_uint), C.POINTER(C.c_void_p)],
            "mvncSetGlobalOption": [C.c_int, C.c_void_p, C.c_uint],
            "mvncGetGraphOption": [C.c_void_p, C.c_int, C.c_void_p, C.POINTER(C.c_uint)],
            "mvncGetDeviceOption": [C.c_void_p, C.c_int, C.c_void_p, C.POINTER(C.c_uint)],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = C.c_int
        self.handle = C.c_void_p()
        level = C.c_int(2 if verbose else 1)
        self.check(self.lib.mvncSetGlobalOption(0, C.byref(level), C.sizeof(level)), "log level")

    @staticmethod
    def check(status, operation):
        if status:
            raise RuntimeError(f"{operation}: NCSDK status {status}")

    def __enter__(self):
        name = C.create_string_buffer(28)
        self.check(self.lib.mvncGetDeviceName(0, name, len(name)), "enumerate")
        self.name = name.value.decode()
        self.check(self.lib.mvncOpenDevice(name.value, C.byref(self.handle)), "open")
        return self

    def __exit__(self, *_):
        if self.handle.value:
            self.check(self.lib.mvncCloseDevice(self.handle), "close")
            self.handle = C.c_void_p()

    def allocate(self, data):
        graph = C.c_void_p()
        raw = C.create_string_buffer(data)
        self.check(self.lib.mvncAllocateGraph(self.handle, C.byref(graph), raw, len(data)), "allocate graph")
        return graph

    def release(self, graph):
        self.check(self.lib.mvncDeallocateGraph(graph), "release graph")

    def run(self, graph, tensor):
        tensor = np.ascontiguousarray(tensor, dtype=np.float16)
        self.check(self.lib.mvncLoadTensor(graph, tensor.ctypes.data, tensor.nbytes, None), "load tensor")
        output, user, length = C.c_void_p(), C.c_void_p(), C.c_uint()
        self.check(self.lib.mvncGetResult(graph, C.byref(output), C.byref(length), C.byref(user)), "get result")
        return np.frombuffer(C.string_at(output, length.value), dtype=np.float16).copy()

    def timings(self, graph):
        output, length = C.c_void_p(), C.c_uint()
        self.check(self.lib.mvncGetGraphOption(graph, 1000, C.byref(output), C.byref(length)), "get timings")
        return np.frombuffer(C.string_at(output, length.value), dtype=np.float32).copy()
