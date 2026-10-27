"""Check the explicit reference against the author's unmodified implementation."""

import importlib.util
import json
import sys

import numpy as np
import torch

from movidius_lab.crivo import MODEL_DIR, ROOT, CrivoReference, CrivoTokenizer, banking77


def main():
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("crivo_upstream_model", MODEL_DIR / "model.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    model, _, _ = upstream.load(MODEL_DIR)
    reference, tokenizer = CrivoReference(), CrivoTokenizer()
    labels, rows = banking77()
    checks = []
    for row in [rows[i] for i in [0, 40, 1000]]:
        ids, positions = tokenizer.banking(row["text"], labels)
        batch = upstream.collate([(ids.tolist(), positions.tolist(), 1)], tokenizer.tokenizer.pad_token_id)
        with torch.inference_mode():
            expected = model(**batch, head_fp32=True)[0].numpy()
            actual = reference.logits(ids, positions).numpy()
        checks.append({"id": row["id"], "tokens": len(ids),
                       "max_logit_error": float(np.max(np.abs(expected - actual))),
                       "top1_equal": bool(expected.argmax() == actual.argmax())})
    result = {"checks": checks, "passed": all(x["max_logit_error"] < 1e-3 and x["top1_equal"] for x in checks)}
    target = ROOT / "results/crivo_reference_parity.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
