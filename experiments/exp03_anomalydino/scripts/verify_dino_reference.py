"""Check the explicit reference against Meta and the author's score aggregation."""

import ast
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from experiments.exp03_anomalydino.model import ROOT, DATA_DIR, DinoReference, prepare_image
from experiments.exp03_anomalydino.scripts.fetch_dino import DINO_REV, ANOMALY_REV


def official_model(reference):
    sys.path.insert(0, str(ROOT / "third_party/dinov2"))
    from dinov2.hub.backbones import dinov2_vits14
    model = dinov2_vits14(pretrained=False).eval()
    w, mapped = reference.weights, {}
    for a, b in [("embeddings.cls_token", "cls_token"), ("embeddings.mask_token", "mask_token"),
                 ("embeddings.position_embeddings", "pos_embed"), ("layernorm.weight", "norm.weight"),
                 ("layernorm.bias", "norm.bias"), ("embeddings.patch_embeddings.projection.weight", "patch_embed.proj.weight"),
                 ("embeddings.patch_embeddings.projection.bias", "patch_embed.proj.bias")]:
        mapped[b] = w[a]
    for i in range(12):
        p, q = f"encoder.layer.{i}", f"blocks.{i}"
        for suffix in ["weight", "bias"]:
            mapped[q + ".attn.qkv." + suffix] = torch.cat([w[p + ".attention.attention." + key + "." + suffix]
                                                         for key in ["query", "key", "value"]], dim=0)
            mapped[q + ".attn.proj." + suffix] = w[p + ".attention.output.dense." + suffix]
            for name in ["norm1", "norm2", "mlp.fc1", "mlp.fc2"]:
                mapped[q + "." + name + "." + suffix] = w[p + "." + name + "." + suffix]
        for j in [1, 2]:
            mapped[q + f".ls{j}.gamma"] = w[p + f".layer_scale{j}.lambda1"]
    model.load_state_dict(mapped, strict=True)
    return model


def main():
    torch.set_num_threads(4)
    reference = DinoReference()
    original = official_model(reference)
    source = ast.parse((ROOT / "third_party/anomalydino/src/post_eval.py").read_text())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "mean_top1p")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "author_mean_top1p", "exec"), namespace)
    bank_pixels, _ = prepare_image(DATA_DIR / "bottle/train/good/000.png")
    reference.set_reference(bank_pixels)
    with torch.inference_mode():
        bank = F.normalize(original.get_intermediate_layers(torch.from_numpy(bank_pixels))[0], dim=-1).numpy()[0]
        rows = []
        for name in ["good/000.png", "broken_large/000.png", "contamination/000.png"]:
            pixels, _ = prepare_image(DATA_DIR / "bottle/test" / name)
            official = F.normalize(original.get_intermediate_layers(torch.from_numpy(pixels))[0], dim=-1).numpy()[0]
            explicit = reference.features(pixels).numpy()[0]
            distances = ((official[:, None] - bank[None]) ** 2).sum(-1).min(-1)
            score = namespace["mean_top1p"](distances)
            actual = reference.infer(pixels)[0]
            row = {"case": name, "feature_max_error": float(abs(official - explicit).max()),
                   "distance_max_error": float(abs(distances - actual[:256]).max()),
                   "author_score": float(score), "reference_score": float(actual[-1])}
            assert row["feature_max_error"] < 1e-5 and row["distance_max_error"] < 2e-6
            assert abs(score - actual[-1]) < 2e-6
            rows.append(row)
    result = {"passed": True, "meta_revision": DINO_REV, "anomalydino_revision": ANOMALY_REV,
              "scope": "same pinned weights, original Meta get_intermediate_layers, brute-force normalized squared L2, author's mean_top1p",
              "cases": rows}
    (ROOT / "experiments/exp03_anomalydino/results/224/dino_reference_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
