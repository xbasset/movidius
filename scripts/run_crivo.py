"""Run a real typed decision on the stick and compare it with the Mac."""

import argparse
import json
import time

import numpy as np
import torch

from movidius_lab.crivo import ROOT, CrivoTokenizer, CrivoReference, banking77
from movidius_lab.myriad import Myriad


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--graph")
    parser.add_argument("--text")
    parser.add_argument("--banking", action="store_true", help="Choose from all 77 BANKING77 intents")
    parser.add_argument("--no-reference", action="store_true", help="Run no model inference on the Mac")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    args.output = args.output or ("results/crivo_banking77_demo.json" if args.banking else "results/crivo_demo.json")
    torch.set_num_threads(4)
    if args.banking:
        labels, _ = banking77()
        options = {f"option_{i}": label for i, label in enumerate(labels)}
        state = {}
        text = args.text or "Where is my card? It has not arrived yet."
        question = "Classify the banking intent of this user request:\n" + text
    else:
        options = {"refund": "Money back for a duplicate charge", "cancel": "Cancel the order", "info": "Ask for information"}
        state = {"ticket": args.text or "My card was charged twice for the same order."}
        question = "What does the customer need?"
    ids, positions = CrivoTokenizer().encode(state, question, options)
    expected = None
    mac_times = []
    if not args.no_reference:
        reference = CrivoReference()
        reference.probabilities(ids, positions)
        for _ in range(args.runs):
            start = time.perf_counter()
            expected = reference.probabilities(ids, positions)
            mac_times.append((time.perf_counter() - start) * 1000)
        print("Mac choice:", list(options.values())[expected.argmax()], flush=True)
    graph = args.graph or ("assets/crivo-ir/banking77" if args.banking else "assets/crivo-ir/demo")
    with Myriad(ROOT / graph) as model:
        print("Graph loaded:", model.load_ms, "ms", flush=True)
        model.infer(ids, positions)
        results, times = [], []
        for _ in range(args.runs):
            start = time.perf_counter()
            probabilities = model.infer(ids, positions).flatten()
            times.append((time.perf_counter() - start) * 1000)
            results.append(probabilities.tolist())
        result = {"state": state, "question": question, "options": options, "tokens": len(ids),
                  "stick_probabilities": results,
                  "stick_latency_ms": times, "graph_load_ms": model.load_ms,
                  "graph": model.meta, "inference_backend": "MYRIAD only; no CPU fallback"}
        if expected is not None:
            result.update(mac_probabilities=expected.tolist(), mac_latency_ms=mac_times,
                          top1_equal=bool(expected.argmax() == probabilities.argmax()),
                          max_probability_error=float(np.max(np.abs(expected - probabilities))))
        path = ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        model.profile(path.with_suffix(".tsv"))
        top = np.argsort(-probabilities)[:5]
        print("Stick:", {list(options.values())[i]: float(probabilities[i]) for i in top}, flush=True)
        print("Median latency:", float(np.median(times)), "ms", flush=True)
        print("Saved:", args.output, flush=True)
    print("Device closed.", flush=True)


if __name__ == "__main__":
    main()
