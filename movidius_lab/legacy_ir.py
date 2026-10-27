"""Small OpenVINO IR v7 writer; no model optimizer required."""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class Tensor:
    layer: int
    port: int
    shape: tuple
    precision: str = "FP16"


class IR:
    def __init__(self, name):
        self.root = ET.Element("net", name=name, version="7", batch="1", precision="FP16")
        self.layers = ET.SubElement(self.root, "layers")
        self.edges = ET.SubElement(self.root, "edges")
        self.weights = bytearray()
        self.constants = {}
        self.constant_values = {}

    @staticmethod
    def port(parent, number, shape, precision):
        port = ET.SubElement(parent, "port", id=str(number), precision=precision)
        for dim in shape:
            ET.SubElement(port, "dim").text = str(dim)

    def node(self, name, operation, inputs, shape, attrs=None, precision="FP16", blobs=None):
        number = len(self.layers)
        layer = ET.SubElement(self.layers, "layer", id=str(number), name=name,
                              type=operation, precision=precision)
        if attrs:
            ET.SubElement(layer, "data", **{key: str(value) for key, value in attrs.items()})
        if inputs:
            ports = ET.SubElement(layer, "input")
            for i, tensor in enumerate(inputs):
                self.port(ports, i, tensor.shape, tensor.precision)
                ET.SubElement(self.edges, "edge", **{"from-layer": str(tensor.layer),
                    "from-port": str(tensor.port), "to-layer": str(number), "to-port": str(i)})
        output = len(inputs)
        self.port(ET.SubElement(layer, "output"), output, shape, precision)
        if blobs:
            parent = ET.SubElement(layer, "blobs")
            for key, (offset, size) in blobs.items():
                ET.SubElement(parent, key, offset=str(offset), size=str(size))
        return Tensor(number, output, tuple(shape), precision)

    def input(self, name, shape, precision="FP16"):
        return self.node(name, "Input", [], shape, precision=precision)

    def const(self, name, value, precision="FP16"):
        dtype = "<i4" if precision == "I32" else "<f2"
        a = np.asarray(value, dtype=dtype, order="C")
        key = (a.shape, precision, a.tobytes())
        if key in self.constants:
            return self.constants[key]
        self.weights.extend(bytes((-len(self.weights)) % 64))
        offset = len(self.weights)
        self.weights.extend(a.tobytes())
        tensor = self.node(name, "Const", [], a.shape, precision=precision,
                           blobs={"custom": (offset, a.nbytes)})
        self.constants[key] = tensor
        self.constant_values[tensor.layer] = a
        return tensor

    def reshape(self, name, x, shape):
        return self.node(name, "Reshape", [x], shape, {"dim": ",".join(map(str, shape))}, x.precision)

    def permute(self, name, x, order):
        return self.node(name, "Permute", [x], [x.shape[i] for i in order],
                         {"order": ",".join(map(str, order))}, x.precision)

    def crop(self, name, x, axis, offset, size):
        shape = list(x.shape)
        shape[axis] = size
        return self.node(name, "Crop", [x], shape, {"axis": axis, "offset": offset, "dim": size})

    def power(self, name, x, power=1, scale=1, shift=0):
        return self.node(name, "Power", [x], x.shape, {"power": power, "scale": scale, "shift": shift})

    def eltwise(self, name, a, b, operation):
        shape = np.broadcast_shapes(a.shape, b.shape)
        # Work around tail corruption in the legacy MA2450 broadcast path.
        if getattr(self, "expand_vector_constants", False) and len(shape) == 3 and shape[1] > 512:
            if b.layer in self.constant_values and b.shape == (1, 1, shape[2]):
                b = self.const(name + "/expanded", np.broadcast_to(self.constant_values[b.layer], shape))
            if a.layer in self.constant_values and a.shape == (1, 1, shape[2]):
                a = self.const(name + "/expanded_left", np.broadcast_to(self.constant_values[a.layer], shape))
        return self.node(name, "Eltwise", [a, b], shape, {"operation": operation})

    def gemm(self, name, a, b, transpose_b=False, alpha=1):
        shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-2 if transpose_b else -1])
        return self.node(name, "GEMM", [a, b], shape,
                         {"alpha": alpha, "beta": 0, "transpose_a": 0, "transpose_b": int(transpose_b)})

    def linear(self, name, x, weight, bias=None):
        w = self.const(name + "/weight", weight.T[None])
        y = self.gemm(name, x, w)
        if bias is not None:
            y = self.eltwise(name + "/bias", y, self.const(name + "/bias_value", bias[None, None]), "sum")
        return y

    def convolution(self, name, x, weight, bias, stride):
        blobs = {}
        for key, value in [("weights", weight), ("biases", bias)]:
            array = np.asarray(value, dtype="<f2", order="C")
            self.weights.extend(bytes((-len(self.weights)) % 64))
            offset = len(self.weights)
            self.weights.extend(array.tobytes())
            blobs[key] = (offset, array.nbytes)
        channels, _, kh, kw = weight.shape
        shape = (1, channels, (x.shape[2] - kh) // stride + 1,
                 (x.shape[3] - kw) // stride + 1)
        return self.node(name, "Convolution", [x], shape,
                         {"kernel": f"{kh},{kw}", "strides": f"{stride},{stride}",
                          "pads_begin": "0,0", "pads_end": "0,0", "dilations": "1,1",
                          "output": channels, "group": 1}, blobs=blobs)

    def topk(self, name, x, k, axis):
        shape = list(x.shape)
        shape[axis] = k
        count = self.const(name + "/k", [k], "I32")
        value = self.node(name, "TopK", [x, count], shape,
                          {"axis": axis, "mode": "max", "sort": "value"})
        self.port(self.layers[-1].find("output"), value.port + 1, shape, "I32")
        return value

    def norm(self, name, x, weight, bias=None, eps=1e-5):
        axis = self.const(name + "/axes", [len(x.shape) - 1], "I32")
        shape = x.shape[:-1] + (1,)
        mean = self.node(name + "/mean", "ReduceMean", [x, axis], shape, {"keep_dims": "true"})
        centered = self.eltwise(name + "/center", x, mean, "sub")
        square = self.power(name + "/square", centered, power=2)
        variance = self.node(name + "/variance", "ReduceMean", [square, axis], shape, {"keep_dims": "true"})
        inv = self.power(name + "/invstd", variance, power=-0.5, shift=eps)
        y = self.eltwise(name + "/normalize", centered, inv, "prod")
        y = self.eltwise(name + "/scale", y, self.const(name + "/weight", weight[None, None]), "prod")
        if bias is not None:
            y = self.eltwise(name + "/bias", y, self.const(name + "/bias_value", bias[None, None]), "sum")
        return y

    def gelu(self, name, x):
        scaled = self.power(name + "/scale", x, scale=1 / np.sqrt(2))
        erf = self.node(name + "/erf", "Erf", [scaled], x.shape)
        cdf = self.power(name + "/cdf", erf, scale=0.5, shift=0.5)
        return self.eltwise(name, x, cdf, "prod")

    def save(self, path, output=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if output is not None:
            needed = {output.layer}
            for edge in reversed(list(self.edges)):
                if int(edge.attrib["to-layer"]) in needed:
                    needed.add(int(edge.attrib["from-layer"]))
            for layer in list(self.layers):
                if int(layer.attrib["id"]) not in needed:
                    self.layers.remove(layer)
            for edge in list(self.edges):
                if int(edge.attrib["to-layer"]) not in needed:
                    self.edges.remove(edge)
        ET.indent(self.root)
        ET.ElementTree(self.root).write(path.with_suffix(".xml"), encoding="utf-8", xml_declaration=True)
        path.with_suffix(".bin").write_bytes(self.weights)
