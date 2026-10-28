"""Export all Crivo inference operations to a fixed-shape Myriad 2 graph."""

import argparse
import json

import numpy as np
from safetensors.numpy import load_file

from experiments.exp02_crivo.model import MODEL_DIR, ROOT
from movidius_lab.legacy_ir import IR


def export(tokens=960, options=77, stop="probabilities", layers=7):
    w = load_file(MODEL_DIR / "model.safetensors")
    g = IR("crivo17m")
    ids = g.input("input_ids", (1, tokens), "I32")
    mask = g.input("padding_mask", (1, 1, 1, tokens))
    positions = g.input("opt_positions", (options,), "I32")
    emb = g.const("embedding_table", w["encoder.embeddings.tok_embeddings.weight"])
    x = g.node("embedding_lookup", "Gather", [emb, ids], (1, tokens, 256), {"axis": 0})

    def norm(name, x, prefix):
        return g.norm(name, x, w[prefix + ".weight"], w.get(prefix + ".bias"))

    def linear(name, x, prefix):
        return g.linear(name, x, w[prefix + ".weight"], w.get(prefix + ".bias"))

    x = norm("embedding_norm", x, "encoder.embeddings.norm")
    if stop == "embedding":
        return g, x
    freqs = np.outer(np.arange(tokens, dtype=np.float32),
                     1.0 / (160000.0 ** (np.arange(0, 64, 2, dtype=np.float32) / 64.0)))
    angles = np.concatenate([freqs, freqs], -1)[None, None]
    cos, sin = g.const("rope_cos", np.cos(angles)), g.const("rope_sin", np.sin(angles))
    indices = np.arange(tokens)
    local_mask = g.const("local_mask", np.where(np.abs(indices[:, None] - indices[None, :]) <= 64,
                                               0, -10000)[None, None])

    def rotary(name, z):
        first, second = g.crop(name + "/first", z, 3, 0, 32), g.crop(name + "/second", z, 3, 32, 32)
        negative = g.power(name + "/negative", second, scale=-1)
        rotated = g.node(name + "/rotate", "Concat", [negative, first], z.shape, {"axis": 3})
        return g.eltwise(name, g.eltwise(name + "/cos", z, cos, "prod"),
                         g.eltwise(name + "/sin", rotated, sin, "prod"), "sum")

    for i in range(layers):
        p, n = f"encoder.layers.{i}", f"layer{i}"
        z = norm(n + "/attn_norm", x, p + ".attn_norm") if i else x
        qkv = linear(n + "/qkv", z, p + ".attn.Wqkv")
        q, k, v = [g.permute(n + f"/{label}/heads", g.reshape(n + f"/{label}/reshape",
                     g.crop(n + f"/{label}/crop", qkv, 2, j * 256, 256), (1, tokens, 4, 64)),
                     (0, 2, 1, 3)) for j, label in enumerate(["q", "k", "v"])]
        scores = g.gemm(n + "/attention_scores", rotary(n + "/rope_q", q),
                        rotary(n + "/rope_k", k), transpose_b=True, alpha=0.125)
        scores = g.eltwise(n + "/padding", scores, mask, "sum")
        if i % 3:
            scores = g.eltwise(n + "/local", scores, local_mask, "sum")
        attn = g.node(n + "/attention_softmax", "SoftMax", [scores], scores.shape, {"axis": 3})
        z = g.gemm(n + "/attention_values", attn, v)
        z = g.reshape(n + "/merge", g.permute(n + "/merge_heads", z, (0, 2, 1, 3)), (1, tokens, 256))
        x = g.eltwise(n + "/attention_residual", x, linear(n + "/attn_output", z, p + ".attn.Wo"), "sum")
        z = linear(n + "/mlp_input", norm(n + "/mlp_norm", x, p + ".mlp_norm"), p + ".mlp.Wi")
        a, gate = g.crop(n + "/mlp_a", z, 2, 0, 384), g.crop(n + "/mlp_gate", z, 2, 384, 384)
        z = g.eltwise(n + "/gated_gelu", g.gelu(n + "/gelu", a), gate, "prod")
        x = g.eltwise(n + "/mlp_residual", x, linear(n + "/mlp_output", z, p + ".mlp.Wo"), "sum")
    if stop == "layers":
        return g, x
    x = norm("final_norm", x, "encoder.final_norm")
    if stop == "encoder":
        return g, x
    x = g.node("option_lookup", "Gather", [x, positions], (1, options, 256), {"axis": 1})
    x = g.eltwise("question_type_choice", x, g.const("choice_embedding", w["type_embedding.weight"][1][None, None]), "sum")
    x = norm("scorer_norm", x, "scorer.0")
    x = g.gelu("scorer_gelu", linear("scorer_hidden", x, "scorer.1"))
    logits = g.reshape("logits", linear("scorer_output", x, "scorer.3"), (1, options))
    if stop == "logits":
        return g, logits
    return g, g.node("probabilities", "SoftMax", [logits], logits.shape, {"axis": 1})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tokens", type=int, default=960)
    parser.add_argument("--options", type=int, default=77)
    parser.add_argument("--layers", type=int, default=7)
    parser.add_argument("--stop", choices=["embedding", "layers", "encoder", "logits", "probabilities"], default="probabilities")
    parser.add_argument("--output", default="assets/crivo-ir/crivo")
    args = parser.parse_args()
    if args.tokens < 1 or not 2 <= args.options <= 255 or not 0 <= args.layers <= 7:
        parser.error("Expected positive tokens, 2–255 options and 0–7 layers")
    if args.layers != 7 and args.stop not in ("embedding", "layers"):
        parser.error("Partial-layer graphs must stop at an intermediate output")
    graph, output = export(args.tokens, args.options, args.stop, args.layers)
    target = ROOT / args.output
    graph.save(target, output)
    meta = {"tokens": args.tokens, "options": args.options, "layers": args.layers,
            "stop": args.stop, "output_shape": output.shape, "weight_bytes": len(graph.weights),
            "ir_layers": len(graph.layers), "precision": "FP16", "question_type": "choice"}
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
