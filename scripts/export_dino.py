"""Export fixed-shape DINOv2 and complete AnomalyDINO graphs for Myriad 2."""

import argparse
import json

import numpy as np
from safetensors.torch import load_file

from movidius_lab.dino import GRAPH_DIR, MODEL_DIR, position_embeddings
from movidius_lab.legacy_ir import IR


def export(stop="anomaly", layers=12, size=224):
    if size == 448 and stop != "features":
        raise ValueError("Use scripts.export_anomalydino448 for the complete 448-square anomaly graph")
    patches = (size // 14) ** 2
    tokens = patches + 1
    weights = load_file(MODEL_DIR / "model.safetensors")
    w = {k: v.numpy() for k, v in weights.items()}
    g = IR(f"dinov2_small_anomalydino_{size}")
    g.expand_vector_constants = size == 448
    pixels = g.input("pixels", (1, 3, size, size))
    x = g.convolution("patch_projection", pixels, w["embeddings.patch_embeddings.projection.weight"],
                      w["embeddings.patch_embeddings.projection.bias"], 14)
    x = g.permute("patch_order", g.reshape("patch_flatten", x, (1, 384, patches)), (0, 2, 1))
    x = g.node("class_token", "Concat", [g.const("cls", w["embeddings.cls_token"]), x], (1, tokens, 384), {"axis": 1})
    x = g.eltwise("position", x, g.const("position_table", position_embeddings(weights, size).numpy()), "sum")

    def norm(name, x, p):
        return g.norm(name, x, w[p + ".weight"], w[p + ".bias"], eps=1e-6)

    def linear(name, x, p):
        return g.linear(name, x, w[p + ".weight"], w[p + ".bias"])

    for i in range(layers):
        p, n = f"encoder.layer.{i}", f"layer{i}"
        z = norm(n + "/norm1", x, p + ".norm1")
        q, k, v = [g.permute(n + "/" + key + "/heads",
                    g.reshape(n + "/" + key + "/reshape", linear(n + "/" + key, z, p + ".attention.attention." + key),
                              (1, tokens, 6, 64)), (0, 2, 1, 3)) for key in ["query", "key", "value"]]
        scores = g.gemm(n + "/attention_scores", q, k, transpose_b=True, alpha=.125)
        attention = g.node(n + "/attention_softmax", "SoftMax", [scores], scores.shape, {"axis": 3})
        z = g.gemm(n + "/attention_values", attention, v)
        z = g.reshape(n + "/merge", g.permute(n + "/merge_heads", z, (0, 2, 1, 3)), (1, tokens, 384))
        z = linear(n + "/attention_output", z, p + ".attention.output.dense")
        z = g.eltwise(n + "/scale1", z, g.const(n + "/gamma1", w[p + ".layer_scale1.lambda1"][None, None]), "prod")
        x = g.eltwise(n + "/residual1", x, z, "sum")
        z = linear(n + "/fc1", norm(n + "/norm2", x, p + ".norm2"), p + ".mlp.fc1")
        z = linear(n + "/fc2", g.gelu(n + "/gelu", z), p + ".mlp.fc2")
        z = g.eltwise(n + "/scale2", z, g.const(n + "/gamma2", w[p + ".layer_scale2.lambda1"][None, None]), "prod")
        x = g.eltwise(n + "/residual2", x, z, "sum")
    if layers != 12:
        return g, x
    x = g.crop("patch_features", norm("final_norm", x, "layernorm"), 1, 1, patches)
    axis2 = g.const("feature_axis", [2], "I32")
    square = g.power("feature_square", x, power=2)
    total = g.node("feature_sumsq", "ReduceSum", [square, axis2], (1, patches, 1), {"keep_dims": "true"})
    invnorm = g.power("feature_invnorm", total, power=-.5, shift=1e-12)
    x = g.eltwise("normalized_features", x, invnorm, "prod")
    if stop == "features":
        return g, x
    bank = g.input("reference_bank", (1, patches, 384))
    similarity = g.gemm("patch_similarity", x, bank, transpose_b=True)
    distances = g.power("squared_l2", similarity, scale=-2, shift=2)
    distances = g.node("nonnegative_distance", "ReLU", [distances], distances.shape)
    nearest = g.node("nearest_reference", "ReduceMin", [distances, axis2], (1, patches, 1), {"keep_dims": "true"})
    flat = g.reshape("patch_distances", nearest, (1, patches))

    # Tournament losers contain every value except one maximum, including ties.
    winners, losers = flat, []
    axis1 = g.const("patch_axis", [1], "I32")
    for level in range(8):
        count = winners.shape[1] // 2
        pairs = g.reshape(f"top2/{level}/pairs", winners, (1, count, 2))
        wins = g.node(f"top2/{level}/winners", "ReduceMax", [pairs, axis2], (1, count, 1), {"keep_dims": "true"})
        loss = g.node(f"top2/{level}/losers", "ReduceMin", [pairs, axis2], (1, count, 1), {"keep_dims": "true"})
        loss = g.reshape(f"top2/{level}/flatten", loss, (1, count))
        loss = g.node(f"top2/{level}/best_loser", "ReduceMax", [loss, axis1], (1, 1), {"keep_dims": "true"})
        losers.append(loss)
        winners = g.reshape(f"top2/{level}/advance", wins, (1, count))
    losses = g.node("top2/loser_list", "Concat", losers, (1, 8), {"axis": 1})
    second = g.node("top2/second", "ReduceMax", [losses, axis1], (1, 1), {"keep_dims": "true"})
    score = g.power("image_score", g.eltwise("top2/sum", winners, second, "sum"), scale=.5)
    return g, g.node("anomaly_output", "Concat", [flat, score], (1, 257), {"axis": 1})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--size", type=int, choices=[224, 448], default=224)
    parser.add_argument("--stop", choices=["features", "anomaly"])
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 0 <= args.layers <= 12:
        parser.error("Expected 0–12 layers")
    stop = args.stop or ("features" if args.size == 448 else "anomaly")
    if args.size == 448 and stop != "features":
        parser.error("Use python -m scripts.export_anomalydino448 for 448-square anomaly scoring")
    graph, output = export(stop, args.layers, args.size)
    path = GRAPH_DIR / (args.output or (stop if args.size == 224 else f"448/{stop}"))
    graph.save(path, output)
    meta = {"model": "DINOv2-S/14", "resolution": [args.size, args.size], "layers": args.layers,
            "stop": stop if args.layers == 12 else "layers", "output_shape": output.shape,
            "weight_bytes": len(graph.weights), "ir_layers": len(graph.layers), "precision": "FP16",
            "interpolation_offset": .1, "masking": False, "rotation": False,
            "shots": 1, "bank_size": (args.size // 14) ** 2, "vector_constants_expanded": args.size == 448,
            "image_score": f"mean of top floor(N * .01) = {(args.size // 14) ** 2 // 100} patch distances"}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
