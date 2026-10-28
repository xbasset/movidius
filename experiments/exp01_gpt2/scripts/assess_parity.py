"""Compare fixed-token hybrid logits and likelihoods with host GPT-2.

This small WikiText-2 slice tests implementation parity, not benchmark quality.
"""

import argparse
import hashlib
import json
import math
import os

import torch
import torch.nn.functional as F

from movidius_lab.device import NCS, ROOT
from experiments.exp01_gpt2.model import GPT2, DeviceQKV, DeviceDense


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offload", choices=["qkv", "all-dense"], default="qkv")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--output", default="experiments/exp01_gpt2/results/hybrid_parity.json")
    args = p.parse_args()
    if not 2 <= args.tokens <= 1024:
        p.error("Use 2–1024 tokens")
    torch.set_num_threads(4)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(ROOT / "assets/tiktoken")
    import tiktoken
    text = (ROOT / "assets/wikitext2-test.txt").read_bytes()
    ids = tiktoken.get_encoding("gpt2").encode(text.decode())[:args.tokens+1]
    model = GPT2(ROOT / "assets/gpt2/model.safetensors")
    expected = model.forward(ids[:-1], use_cache=False)
    with NCS() as device:
        hook = DeviceQKV(device, model) if args.offload == "qkv" else DeviceDense(device, model)
        if args.offload == "qkv":
            model.projection_hook = hook
        else:
            model.linear_hook = hook
        try:
            actual = model.forward(ids[:-1], use_cache=False)
        finally:
            hook.close()
    targets = torch.tensor(ids[1:])
    nll_ref = float(F.cross_entropy(expected[0], targets))
    nll_actual = float(F.cross_entropy(actual[0], targets))
    log_ref, log_actual = expected.log_softmax(-1), actual.log_softmax(-1)
    kl = float((log_ref.exp() * (log_ref-log_actual)).sum(-1).mean())
    delta = math.exp(nll_actual-nll_ref)-1
    report = {"purpose": "numerical parity smoke test; not full WikiText-2 benchmark",
              "dataset_url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
              "dataset_sha256": hashlib.sha256(text).hexdigest(), "offload": args.offload,
              "token_ids": ids, "scored_tokens": len(ids)-1, "context": "one causal window, beginning of test file",
              "host_nll": nll_ref, "hybrid_nll": nll_actual,
              "host_slice_perplexity": math.exp(nll_ref), "hybrid_slice_perplexity": math.exp(nll_actual),
              "relative_perplexity_change": delta, "mean_KL_reference_to_hybrid": kl,
              "max_logit_error": float((expected-actual).abs().max()),
              "top1_agreement": float((expected.argmax(-1) == actual.argmax(-1)).float().mean()),
              "device_projection_calls": hook.calls, "finite": bool(torch.isfinite(actual).all()),
              "passes_proposed_1pct_parity_target": bool(math.isfinite(delta) and delta <= 0.01)}
    path = ROOT / args.output
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "token_ids"}, indent=2))
    if not report["finite"]:
        raise AssertionError("Nonfinite device logits")


if __name__ == "__main__":
    main()
