"""Evaluate cached 448-square predictions against all six paper metrics."""

import argparse
import json
import time

import numpy as np

from experiments.exp03_anomalydino.paper import CATEGORIES, cases
from experiments.exp03_anomalydino.metrics import image_metrics, pixel_metrics
from movidius_lab.paths import sha256
from experiments.exp03_anomalydino.model import ROOT


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--device", choices=["cpu", "mps", "myriad"], required=True)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument("--wait-for-cache", action="store_true")
    args = parser.parse_args()
    cache = ROOT / "assets/mvtec448" / args.device
    reports = []
    for category in args.categories:
        path = cache / (category + ".npz")
        if args.wait_for_cache and not path.with_suffix('.json').exists():
            print(args.device, category, 'waiting for completed predictions', flush=True)
            while not path.with_suffix('.json').exists():
                time.sleep(5)
        fingerprint = sha256(path)
        report_path = cache / (category + "_metrics.json")
        if report_path.exists() and json.loads(report_path.read_text())["cache_sha256"] == fingerprint:
            reports.append(json.loads(report_path.read_text()))
            print(args.device, category, "metrics cached", flush=True)
            continue
        data = np.load(path)
        rows = cases(category)
        if data["ids"].tolist() != [row["id"] for row in rows]:
            raise ValueError("Dataset/cache order mismatch")
        trials = []
        for seed in range(3):
            metrics = image_metrics(data["gold"], data["scores"][:, seed])
            metrics.update(pixel_metrics(data["maps"][:, seed], rows))
            trials.append(metrics)
            print(args.device, category, seed, json.dumps(metrics), flush=True)
        report = {"category": category, "cache_sha256": fingerprint, "trials": trials}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
    keys = list(reports[0]["trials"][0])
    macro = {key: np.mean([[r["trials"][seed][key] for r in reports] for seed in range(3)], axis=1).tolist()
             for key in keys}
    result = {"device": args.device, "complete": len(reports) == 15, "test_images": sum(len(cases(c)) for c in args.categories),
              "resolution": [448, 448], "reference_trials": 3, "categories": reports,
              "aggregation": "category macro average within each reference trial; mean and population std over three trials",
              "macro_trials": macro,
              "mean": {key: float(np.mean(values)) for key, values in macro.items()},
              "std": {key: float(np.std(values)) for key, values in macro.items()}}
    (ROOT / f"experiments/exp03_anomalydino/results/448/dino448_{args.device}_metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
