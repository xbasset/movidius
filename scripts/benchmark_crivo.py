"""Single-request BANKING77 quality and latency; no truncation or label shortlist."""

import argparse
import json
import time
from contextlib import nullcontext

import numpy as np
import torch

from movidius_lab.crivo import (BANKING_DIR, ROOT, CrivoReference, CrivoTokenizer,
                                banking77, macro_f1, model_manifest, sha256)
from movidius_lab.myriad import Myriad


def save_result(path, result):
    # Keep one prediction per line so the public results remain reviewable.
    summary = {k: v for k, v in result.items() if k != "predictions"}
    header = json.dumps(summary, indent=2).rstrip()[:-1].rstrip()
    rows = ",\n".join("    " + json.dumps(r, separators=(",", ":")) for r in result["predictions"])
    path.write_text(header + ',\n  "predictions": [\n' + rows + "\n  ]\n}\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--device", choices=["cpu", "mps", "myriad"], default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=str)
    parser.add_argument("--graph", default="assets/crivo-ir/banking77")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    torch.set_num_threads(args.threads)
    tokenizer = CrivoTokenizer()
    labels, rows = banking77()
    if args.limit:
        rows = rows[:args.limit]
    encoded = [tokenizer.banking(row["text"], labels) for row in rows]
    predictions = []
    path = ROOT / (args.output or f"results/crivo_banking77_{args.device}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "results/raw" / (path.stem + ".jsonl")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.device == "myriad":
        meta = json.loads((ROOT / args.graph).with_suffix(".json").read_text())
        if meta["layers"] != 7 or meta["stop"] != "probabilities" or meta["options"] != 77:
            raise ValueError("BANKING77 requires all seven layers, the decision head, softmax and 77 options")
    context = Myriad(ROOT / args.graph) if args.device == "myriad" else nullcontext(CrivoReference(args.device))
    with context as model, checkpoint.open("w") as log:
        def infer(ids, positions):
            return model.infer(ids, positions).flatten() if args.device == "myriad" else model.probabilities(ids, positions)
        infer(*encoded[0])
        for row, (ids, positions) in zip(rows, encoded):
            start = time.perf_counter()
            probabilities = infer(ids, positions)
            if not np.isfinite(probabilities).all() or abs(float(probabilities.sum()) - 1) > 0.01:
                raise ValueError(f"Invalid probability distribution on {row['id']}")
            elapsed = (time.perf_counter() - start) * 1000
            prediction = {"id": row["id"], "gold": row["gold"], "prediction": int(probabilities.argmax()),
                          "tokens": len(ids), "latency_ms": elapsed, "probabilities": probabilities.tolist()}
            if args.device == "myriad":
                prediction["infer_api_ms"] = model.last_infer_ms
            predictions.append(prediction)
            log.write(json.dumps(prediction) + "\n")
            log.flush()
            if len(predictions) % 100 == 0 or len(predictions) == 1:
                print(f"{len(predictions)}/{len(rows)}, latest {elapsed:.1f} ms", flush=True)
        if args.device == "myriad":
            model.profile(path.with_suffix(".tsv"))
    gold = [r["gold"] for r in predictions]
    pred = [r["prediction"] for r in predictions]
    times = [r["latency_ms"] for r in predictions]
    result = {"model": model_manifest(), "backend": "openvino-MYRIAD-only" if args.device == "myriad" else f"pytorch-{args.device}", "host_torch_threads": args.threads,
              "benchmark": "BANKING77, Decision Index direct adapter", "cases": len(rows),
              "full_test_set": len(rows) == 3080, "options": 77, "truncation": "none", "batch_size": 1,
              "test_sha256": sha256(BANKING_DIR / "test.csv"),
              "labels_sha256": sha256(BANKING_DIR / "categories.json"),
              "accuracy": float(np.mean(np.equal(gold, pred))), "macro_f1": macro_f1(gold, pred),
              "latency": {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
                          "mean_ms": float(np.mean(times)),
                          "boundary": "pretokenized request to host probabilities, includes device synchronization",
                          "excludes": ["loading", "tokenization", "one warmup"]},
              "predictions": predictions}
    if args.device == "myriad":
        result["graph"] = model.meta
        result["graph_sha256"] = sha256(model.path.with_suffix(".xml"))
        result["weights_sha256"] = sha256(model.path.with_suffix(".bin"))
        result["graph_load_ms"] = model.load_ms
    save_result(path, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("predictions", "model")}, indent=2))


if __name__ == "__main__":
    main()
