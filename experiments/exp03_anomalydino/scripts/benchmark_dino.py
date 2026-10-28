"""Complete bottle test split, three one-shot references, and synchronized timing."""

import argparse
import json
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

from movidius_lab.paths import sha256
from experiments.exp03_anomalydino.model import (ROOT, DATA_DIR, GRAPH_DIR, DinoReference, bottle_cases,
                                manifest, prepare_image, unpack)
from movidius_lab.myriad import Myriad


def pixel_map(patches, shape):
    resized = F.interpolate(torch.from_numpy(patches)[None, None], size=shape,
                            mode="bilinear", align_corners=False)[0, 0].numpy()
    return gaussian_filter(resized, sigma=4)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--device", choices=["cpu", "mps", "myriad"], default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cases = bottle_cases()
    if len(cases) != 83:
        raise ValueError("Expected all 83 bottle test cases")
    train = sorted((DATA_DIR / "bottle/train/good").glob("*.png"))
    if any(seed < 0 or seed >= len(train) for seed in args.seeds):
        parser.error("Invalid one-shot reference index")
    prepared = [prepare_image(row["path"])[0] for row in cases]
    references = [prepare_image(train[seed])[0] for seed in args.seeds]
    originals = [Image.open(row["path"]).size[::-1] for row in cases]
    truth = [np.asarray(Image.open(row["mask"])) > 0 if row["gold"] else np.zeros(shape, dtype=bool)
             for row, shape in zip(cases, originals)]
    truth = np.concatenate([x.ravel() for x in truth])
    path = ROOT / (args.output or f"experiments/exp03_anomalydino/results/224/dino_bottle_{args.device}.json")
    cache = ROOT / "assets/dino-benchmark"
    cache.mkdir(parents=True, exist_ok=True)
    banks, runs = [], []
    if args.device == "myriad":
        with Myriad(GRAPH_DIR / "features") as features:
            for reference in references:
                banks.append(features.infer_tensors({"pixels": reference.astype(np.float16)}).astype(np.float16))
        model = Myriad(GRAPH_DIR / "anomaly")
    else:
        model = DinoReference(args.device)
        banks = [model.features(reference) for reference in references]
    try:
        for seed, bank in zip(args.seeds, banks):
            if args.device == "myriad":
                model.set_input("reference_bank", bank)
                inputs = [x.astype(np.float16) for x in prepared]

                def infer(pixels):
                    model.set_input("pixels", pixels)
                    return model.execute()
            else:
                model.bank = bank
                inputs = prepared
                infer = model.infer
            for _ in range(3):
                infer(inputs[0])
            predictions, maps = [], []
            for i, (row, pixels) in enumerate(zip(cases, inputs)):
                start = time.perf_counter()
                result = infer(pixels)
                elapsed = (time.perf_counter() - start) * 1000
                patches, score = unpack(result)
                assert abs(score - np.sort(patches.ravel())[-2:].mean()) < .002
                maps.append(patches)
                predictions.append({"id": row["id"], "gold": row["gold"], "score": score,
                                    "latency_ms": elapsed})
                if i % 20 == 0:
                    print(f"{args.device} seed={seed} {i+1}/83, {elapsed:.1f} ms", flush=True)
            np.savez_compressed(cache / f"{args.device}_seed{seed}.npz", maps=np.asarray(maps))
            print("Evaluating original-resolution pixel AUROC…", flush=True)
            pixel_scores = np.concatenate([pixel_map(patches, shape).ravel() for patches, shape in zip(maps, originals)])
            times = [x["latency_ms"] for x in predictions]
            runs.append({"seed": seed, "reference": str(train[seed].relative_to(DATA_DIR)),
                         "reference_sha256": sha256(train[seed]),
                         "image_auroc": float(roc_auc_score([x["gold"] for x in predictions], [x["score"] for x in predictions])),
                         "pixel_auroc": float(roc_auc_score(truth, pixel_scores)),
                         "latency": {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))},
                         "predictions": predictions})
            del pixel_scores
            print(json.dumps({k: v for k, v in runs[-1].items() if k != "predictions"}), flush=True)
            if args.device == "myriad":
                model.profile(path.with_suffix(".tsv"))
    finally:
        if args.device == "myriad":
            model.close()
    result = {"model": manifest(), "backend": args.device, "precision": "FP16" if args.device == "myriad" else "FP32",
              "benchmark": "MVTec AD bottle, complete test split; not the 15-category benchmark",
              "cases_per_seed": 83, "normal_cases": 20, "anomalous_cases": 63,
              "resolution": [224, 224], "shots": 1, "reference_selection": "sorted train/good filenames indexed by seed",
              "masking": False, "rotations": False, "nearest_neighbors": 1,
              "distance": "squared L2 of L2-normalized features", "image_score": "mean of top two patch distances",
              "pixel_evaluation": "bilinear resize to original image; Gaussian sigma=4; original ground truth masks",
              "timing": {"batch_size": 1, "host_threads": 4, "warmups_per_seed": 3,
                         "boundary": "preprocessed tensor to host 16x16 patch distances and image score",
                         "excludes": ["model loading", "reference calibration", "image decoding and normalization", "pixel metric postprocessing"]},
              "data_manifest_sha256": sha256(DATA_DIR / "manifest.json"), "runs": runs,
              "mean_image_auroc": float(np.mean([x["image_auroc"] for x in runs])),
              "mean_pixel_auroc": float(np.mean([x["pixel_auroc"] for x in runs]))}
    if args.device == "myriad":
        result["graph"] = model.meta
        result["graph_sha256"] = sha256(model.path.with_suffix(".xml"))
        result["weights_sha256"] = sha256(model.path.with_suffix(".bin"))
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["runs", "model"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
