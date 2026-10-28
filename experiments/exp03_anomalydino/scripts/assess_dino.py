"""Summarize complete bottle trials, FP16 drift and MYRIAD-only execution."""

import csv
import json

import numpy as np

from movidius_lab.paths import sha256
from experiments.exp03_anomalydino.model import ROOT, MODEL_DIR, DATA_DIR


def main():
    backends = {key: json.loads((ROOT / f"experiments/exp03_anomalydino/results/224/dino_bottle_{key}.json").read_text())
                for key in ["myriad", "cpu", "mps"]}
    summaries = {}
    for key, result in backends.items():
        assert len(result["runs"]) == 3 and result["cases_per_seed"] == 83
        times = [row["latency_ms"] for run in result["runs"] for row in run["predictions"]]
        summaries[key] = {"image_auroc_mean": result["mean_image_auroc"],
                          "image_auroc_std": float(np.std([x["image_auroc"] for x in result["runs"]])),
                          "pixel_auroc_mean": result["mean_pixel_auroc"],
                          "latency_median_ms": float(np.median(times)), "latency_p95_ms": float(np.percentile(times, 95)),
                          "measured_requests": len(times)}
    parity = {}
    for key in ["myriad", "mps"]:
        score_errors, map_errors = [], []
        for actual, expected in zip(backends[key]["runs"], backends["cpu"]["runs"]):
            assert actual["seed"] == expected["seed"]
            for a, b in zip(actual["predictions"], expected["predictions"]):
                assert a["id"] == b["id"]
                score_errors.append(abs(a["score"] - b["score"]))
            a = np.load(ROOT / f"assets/dino-benchmark/{key}_seed{actual['seed']}.npz")["maps"]
            b = np.load(ROOT / f"assets/dino-benchmark/cpu_seed{actual['seed']}.npz")["maps"]
            map_errors.append(abs(a - b).reshape(-1))
        errors = np.concatenate(map_errors)
        parity[key + "_vs_cpu"] = {"image_score_mae": float(np.mean(score_errors)),
                                    "image_score_max_error": float(np.max(score_errors)),
                                    "patch_distance_mae": float(errors.mean()), "patch_distance_max_error": float(errors.max())}
    with (ROOT / "experiments/exp03_anomalydino/results/224/dino_bottle_myriad.tsv").open() as f:
        profile = list(csv.DictReader(f, delimiter="\t"))
    active = [row for row in profile if row["status"] == "2"]
    cpu_us = sum(int(row["cpu_us"]) for row in profile)
    assert cpu_us == 0 and active
    data = json.loads((DATA_DIR / "manifest.json").read_text())
    used_hashes = {key: value for key, value in data["sha256"].items()
                   if "/test/" in key or "/ground_truth/" in key or key in [f"bottle/train/good/{i:03}.png" for i in range(3)]}
    result = {"scope": "MVTec AD bottle only; 224x224; one shot; no masking or rotations; seeds 0,1,2",
              "evaluated_test_cases": 83, "unique_test_categories": 1, "total_mvtec_categories": 15,
              "metrics": summaries, "parity": parity,
              "device_execution": {"backend": "MYRIAD only", "executed_stages": len(active), "profile_cpu_us": cpu_us,
                                   "head_stages": [r["layer"] for r in active if r["layer"].startswith(("nearest_", "top2/", "image_score"))]},
              "model_provenance": json.loads((MODEL_DIR / "manifest.json").read_text()),
              "data_provenance": {"repo": data["repo"], "revision": data["revision"], "license": data["license"], "sha256": used_hashes},
              "energy": "excluded; no electrical measurement"}
    firmware = ROOT / "third_party/openvino-2020.3/bin/aarch64/Release/lib/usb-ma2450.mvcmd"
    result["firmware_sha256"] = sha256(firmware)
    (ROOT / "experiments/exp03_anomalydino/results/224/dino_assessment.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["data_provenance", "model_provenance", "device_execution"]}, indent=2))


if __name__ == "__main__":
    main()
