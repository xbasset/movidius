"""Minimal NCSDK v2-format blob writer for a single FP16 dense operation.

The format is documented by the archived NCSDK 1.12 serializer. This writes
data for stock firmware; it does not compile or inject executable SHAVE code.
"""

import struct

import numpy as np


def dense_graph(weight, bias=None, shaves=12):
    """Serialize y = x @ weight + bias; weight has shape [inputs, outputs]."""
    weight = np.asarray(weight, dtype="<f2", order="C")
    if weight.ndim != 2 or not 1 <= shaves <= 12:
        raise ValueError("Expected a matrix and 1–12 SHAVEs")
    inputs, outputs = weight.shape
    if not inputs or not outputs:
        raise ValueError("Empty matrix")
    payload = bytearray(weight.tobytes())
    payload.extend(bytes((-len(payload)) % 64))
    bias_offset = len(payload)
    if bias is not None:
        bias = np.asarray(bias, dtype="<f2")
        if bias.shape != (outputs,):
            raise ValueError("Bias length differs from output width")
        payload.extend(bias.tobytes())
        payload.extend(bytes((-len(payload)) % 64))

    stage = bytearray(b"dense\0".ljust(100, b"\0"))
    stage.extend(struct.pack("<BIbbBBBBB", 4, 0x80000000, 1, 1, 1, 1, 0, 0, 0))
    dims = [1, 1, inputs, 1, inputs, outputs, 1, 1, outputs]
    strides = [2*inputs, 2*inputs, 2, 2*outputs, 2*outputs, 2, 2*outputs, 2*outputs, 2]
    stage.extend(struct.pack("<18I", *(dims + strides)))
    stage.extend(struct.pack("<BBB", 2, 2, 2))
    # Pointers are relative to their selected memory: input=1, output=2, blob=3.
    for pointer, index in [(0, 1), (0, 3), (bias_offset if bias is not None else 0, 3 if bias is not None else 0), (0, 0), (0, 2)]:
        stage.extend(struct.pack("<IH", pointer, index))
    stage.extend(struct.pack("<BBIHH", 5, 9 if bias is not None else 5, 0, 0, 0))
    assert len(stage) == 227
    start = (264 + len(stage) + 7) // 8 * 8
    header = bytearray(264)
    struct.pack_into("<II", header, 32, start + len(payload), 2)
    header[40:45] = b"dense"
    struct.pack_into("<IIHHIII", header, 240, 1, start, 0, shaves-1, 0, 0, 0)
    return bytes(header + stage + bytes(start - 264 - len(stage)) + payload)
