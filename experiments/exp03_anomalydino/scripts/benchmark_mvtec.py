"""Cache all 15 categories, three isolated one-shot heads, at 448 square."""

import argparse
import json
import time

import numpy as np
import torch

from experiments.exp03_anomalydino.paper import CATEGORIES, MASKED, MacPaper, MyriadPaper, cases
from experiments.exp03_anomalydino.model import ROOT, DATA_DIR, prepare_image, manifest
from movidius_lab.paths import sha256


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--device", choices=["cpu", "mps", "myriad"], required=True)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cache = ROOT / "assets/mvtec448" / args.device
    cache.mkdir(parents=True, exist_ok=True)
    model = MyriadPaper() if args.device == "myriad" else MacPaper(args.device)
    summaries = []
    for category in args.categories:
        archive, report = cache / (category + ".npz"), cache / (category + ".json")
        if archive.exists() and report.exists():
            print(args.device, category, "cached", flush=True)
            summaries.append(json.loads(report.read_text()))
            continue
        print(args.device, category, "building reference banks", flush=True)
        model.calibrate(category)
        rows = cases(category)
        pixels, _ = prepare_image(rows[0]["path"], 448)
        for _ in range(3):
            model.infer(pixels, category in MASKED)
        maps, scores, times = [], [], []
        cpu_archive = ROOT / "assets/mvtec448/cpu" / (category + ".npz")
        comparison = np.load(cpu_archive) if args.device == "myriad" else None
        differences, mask_differences = [], []
        for i, row in enumerate(rows):
            pixels, _ = prepare_image(row["path"], 448)
            start = time.perf_counter()
            result = model.infer(pixels, category in MASKED)
            elapsed = (time.perf_counter() - start) * 1000
            if result.shape != (3, 1025) or not np.isfinite(result).all():
                raise ValueError("Invalid complete output")
            if comparison is not None:
                expected_score = np.sort(result[:, :1024], axis=1)[:, -10:].mean(axis=1)
                if not np.allclose(expected_score, result[:, -1], atol=.001):
                    raise ValueError("Device image score disagrees with its patch map")
                differences.append(float(np.abs(result[:, -1] - comparison["scores"][i]).max()))
                if category in MASKED:
                    expected_mask = np.any(comparison["maps"][i] > 0, axis=0).ravel()
                    actual_mask = np.any(result[:, :1024] > 0, axis=0)
                    mask_differences.append(int(np.count_nonzero(expected_mask != actual_mask)))
                if i < 3 and differences[-1] > .05:
                    raise ValueError("MYRIAD score differs excessively from the CPU reference")
            maps.append(result[:, :1024].reshape(3, 32, 32))
            scores.append(result[:, -1])
            times.append(elapsed)
            if i % 25 == 0:
                print(args.device, category, f"{i+1}/{len(rows)}", f"{elapsed:.1f} ms", flush=True)
        np.savez_compressed(archive, maps=np.asarray(maps), scores=np.asarray(scores),
                            times=np.asarray(times), ids=np.asarray([row["id"] for row in rows]),
                            gold=np.asarray([row["gold"] for row in rows]))
        summary = {"category": category, "cases": len(rows), "masked": category in MASKED,
                   "joint_three_head_median_ms": float(np.median(times)),
                   "references": [{"id": f"{category}/train/good/{seed:03}.png",
                                   "sha256": sha256(DATA_DIR / category / "train/good" / f"{seed:03}.png")}
                                  for seed in range(3)], "cache_sha256": sha256(archive)}
        if comparison is not None:
            summary.update({"reference_self_scores": model.self_scores,
                            "maximum_score_difference_vs_cpu": max(differences),
                            "mean_score_difference_vs_cpu": float(np.mean(differences)),
                            "mask_pixels_different_vs_cpu": int(sum(mask_differences)),
                            "mask_pixels_compared": len(mask_differences) * 1024})
        report.write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
    result = {"model": manifest(), "device": args.device, "resolution": [448, 448],
              "protocol": "one shot, agnostic: eight rotations, query PCA mask on five categories",
              "pca": ("64 normalized power iterations, FP16 on MYRIAD" if args.device == "myriad" else
                      "original sklearn randomized PCA, fixed random_state=0; CPU preprocessing as in author code"),
              "distance": "1 minus cosine similarity", "reference_banks_are_isolated": True,
              "encoder_shared_across_three_reference_trials": True,
              "timing_boundary": "prepared tensor to three score/maps; includes query mask and synchronization",
              "complete": len(summaries) == 15, "categories": summaries}
    (ROOT / f"experiments/exp03_anomalydino/results/448/dino448_{args.device}_inference.json").write_text(json.dumps(result, indent=2) + "\n")
    if hasattr(model, "close"):
        model.close()


if __name__ == "__main__":
    main()
