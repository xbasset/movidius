"""Validate device matrix arithmetic against FP32 and rounded-FP16 inputs."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from movidius_lab.device import NCS, ROOT
from movidius_lab.graph import dense_graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=int, default=32)
    p.add_argument("--outputs", type=int, default=64)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--gpt2", action="store_true")
    p.add_argument("--matrix", choices=["qkv", "mlp-up", "mlp-down", "lm-head"], default="qkv")
    p.add_argument("--output", default="experiments/exp01_gpt2/results/dense_device.json")
    args = p.parse_args()
    rng = np.random.default_rng(1337)
    if args.gpt2:
        from safetensors.numpy import load_file
        weights = load_file(ROOT / "assets/gpt2/model.safetensors")
        prefix = {"qkv": "h.0.attn.c_attn", "mlp-up": "h.0.mlp.c_fc", "mlp-down": "h.0.mlp.c_proj"}.get(args.matrix)
        if prefix:
            w = weights[prefix + ".weight"]
            b = weights[prefix + ".bias"]
        else:
            w = weights["wte.weight"].T
            b = None
        name = f"GPT-2 {args.matrix} projection"
    else:
        w = rng.normal(0, 0.1, (args.inputs, args.outputs)).astype(np.float32)
        b = rng.normal(0, 0.1, args.outputs).astype(np.float32)
        name = "seeded random dense"
    blob = dense_graph(w, b)
    report = {"operation": name, "shape": list(w.shape), "device_inference": True,
              "graph_sha256": hashlib.sha256(blob).hexdigest(), "firmware": "stock NCSDK v1",
              "weight_storage_bytes": w.size * 2, "seed": 1337, "measurements": []}
    with NCS(verbose=True) as device:
        start = time.perf_counter()
        graph = device.allocate(blob)
        report["allocation_seconds"] = time.perf_counter() - start
        try:
            for i in range(args.runs + 1):
                x = rng.normal(0, 1, w.shape[0]).astype(np.float16)
                start = time.perf_counter()
                y = device.run(graph, x).astype(np.float32)
                elapsed = time.perf_counter() - start
                reference = x.astype(np.float32) @ w.astype(np.float16).astype(np.float32)
                if b is not None:
                    reference += b.astype(np.float16).astype(np.float32)
                error = y - reference
                measurement = {"usb_wall_seconds": elapsed, "device_ms": float(device.timings(graph).sum()),
                               "max_abs_error": float(np.max(np.abs(error))),
                               "relative_l2_error": float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-12)),
                               "finite": bool(np.isfinite(y).all()), "sample": y[:8].tolist()}
                print(json.dumps(measurement), flush=True)
                if not measurement["finite"] or measurement["relative_l2_error"] > 0.02:
                    raise AssertionError(f"Dense numerical mismatch: {measurement}")
                if i:
                    report["measurements"].append(measurement)
        finally:
            device.release(graph)
    path = ROOT / args.output
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
