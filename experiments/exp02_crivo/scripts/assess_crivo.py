"""Compare complete BANKING77 runs and retain probability-level parity evidence."""

import argparse
import json

import numpy as np

from movidius_lab.paths import ROOT, sha256


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cpu", default="experiments/exp02_crivo/results/crivo_banking77_cpu.json")
    parser.add_argument("--stick", default="experiments/exp02_crivo/results/crivo_banking77_myriad.json")
    parser.add_argument("--mps", default="experiments/exp02_crivo/results/crivo_banking77_mps.json")
    parser.add_argument("--output", default="experiments/exp02_crivo/results/crivo_assessment.json")
    args = parser.parse_args()
    cpu = json.loads((ROOT / args.cpu).read_text())
    stick = json.loads((ROOT / args.stick).read_text())
    for field in ["test_sha256", "labels_sha256", "options", "truncation"]:
        if cpu[field] != stick[field]:
            raise ValueError(f"Mismatched comparison field {field}")
    if cpu["model"] != stick["model"]:
        raise ValueError("Model identities differ")
    if not all(x["cases"] == 3080 and x["full_test_set"] for x in [cpu, stick]):
        raise ValueError("Assessment requires the complete test set on both backends")
    a, b = cpu["predictions"], stick["predictions"]
    if [x["id"] for x in a] != [x["id"] for x in b]:
        raise ValueError("Case order or coverage differs")
    if [x["gold"] for x in a] != [x["gold"] for x in b]:
        raise ValueError("Gold labels differ")
    p, q = np.array([x["probabilities"] for x in a]), np.array([x["probabilities"] for x in b])
    sums = q.sum(1)
    q_normalized = np.clip(q / sums[:, None], 1e-15, 1)
    kl = np.sum(p * (np.log(np.maximum(p, 1e-15)) - np.log(q_normalized)), axis=1)
    disagreement = [x["id"] for x, y in zip(a, b) if x["prediction"] != y["prediction"]]
    backends = {"Mac CPU": {k: cpu[k] for k in ["accuracy", "macro_f1", "latency"]},
                "Myriad 2": {k: stick[k] for k in ["accuracy", "macro_f1", "latency"]}}
    mps_path = ROOT / args.mps
    if mps_path.exists():
        mps = json.loads(mps_path.read_text())
        if (any(mps[k] != cpu[k] for k in ["test_sha256", "labels_sha256", "model", "options"])
                or mps["cases"] != 3080 or not mps["full_test_set"]
                or [x["id"] for x in mps["predictions"]] != [x["id"] for x in a]):
            raise ValueError("Metal run does not cover the same full test set")
        backends["Mac Metal GPU"] = {k: mps[k] for k in ["accuracy", "macro_f1", "latency"]}
    profile = (ROOT / args.stick).with_suffix(".tsv")
    stages = [line.split("\t") for line in profile.read_text().splitlines()[1:]]
    active = [s for s in stages if int(s[2]) == 2]
    result = {
        "cases": 3080, "options": 77, "truncation": "none", "backends": backends,
        "top1_agreement": 1 - len(disagreement) / 3080,
        "disagreements": disagreement,
        "macro_f1_delta_stick_minus_cpu": stick["macro_f1"] - cpu["macro_f1"],
        "probability_parity": {"mean_kl_cpu_to_stick": float(kl.mean()),
            "mean_max_absolute_error": float(np.abs(p - q).max(1).mean()),
            "max_absolute_error": float(np.abs(p - q).max()),
            "max_stick_probability_sum_error": float(np.max(np.abs(sums - 1))),
            "kl_note": "Normalize the raw FP16 softmax sums only for this diagnostic; predictions use raw device outputs."},
        "runtime": {"openvino_revision": "2fe9b1523058e282ad374db7dc1b3538c7d2dd27",
            "firmware_sha256": sha256(ROOT / "third_party/openvino-2020.3/bin/aarch64/Release/lib/usb-ma2450.mvcmd"),
            "graph_xml_sha256": sha256(ROOT / "assets/crivo-ir/banking77.xml"),
            "graph_weights_sha256": sha256(ROOT / "assets/crivo-ir/banking77.bin"),
            "graph": json.loads((ROOT / "assets/crivo-ir/banking77.json").read_text()),
            "backend": "MYRIAD exclusively; no CPU plugin or HETERO fallback loaded",
            "executed_profile_stages": len(active),
            "profile_execution_types": sorted({s[1] for s in active}),
            "device_stage_total_us": sum(int(s[3]) for s in active),
            "profile_host_cpu_total_us": sum(int(s[4]) for s in active),
            "profile_note": "Last request only; stage durations can overlap and their sum is not end-to-end latency."},
        "power": "Not measured; requires external USB power measurement",
        "artifacts": {args.cpu: sha256(ROOT / args.cpu), args.stick: sha256(ROOT / args.stick),
                      str(profile.relative_to(ROOT)): sha256(profile)},
    }
    if mps_path.exists():
        result["artifacts"][args.mps] = sha256(mps_path)
    (ROOT / args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
