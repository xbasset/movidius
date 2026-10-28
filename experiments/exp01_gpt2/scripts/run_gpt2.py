"""Measure host GPT-2 or a clearly labelled partial Movidius offload."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.exp01_gpt2.model import GPT2, DeviceQKV, DeviceDense
from movidius_lab.device import NCS, ROOT


def run(model, ids, count, temperature=0.0, seed=1337):
    generator = torch.Generator().manual_seed(seed)

    def sample(logits):
        if temperature == 0:
            return int(logits.argmax())
        values, indices = torch.topk(logits.float().cpu(), 40)
        choice = torch.multinomial((values / temperature).softmax(-1), 1, generator=generator)
        return int(indices[choice])

    model.reset()
    begin = time.perf_counter()
    logits = model.forward(ids, last_only=True)
    first = time.perf_counter() - begin
    generated = [sample(logits[0, -1])]
    begin = time.perf_counter()
    for _ in range(count-1):
        logits = model.forward([generated[-1]], last_only=True)
        generated.append(sample(logits[0, -1]))
    elapsed = time.perf_counter() - begin
    return {"generated_ids": generated, "time_to_first_token_seconds": first,
            "decode_seconds": elapsed, "decode_tokens_per_second": (count-1)/elapsed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="In 2017, a small computer could")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--offload", choices=["none", "qkv", "all-dense"], default="none",
                   help="all-dense is an experimental path with a known graph-reload failure")
    p.add_argument("--output", help="Defaults to a separate result file for each offload mode")
    args = p.parse_args()
    if args.hybrid:
        args.offload = "qkv"
    if args.output is None:
        args.output = {"none": "experiments/exp01_gpt2/results/gpt2_host.json", "qkv": "experiments/exp01_gpt2/results/gpt2_hybrid.json",
                       "all-dense": "experiments/exp01_gpt2/results/experimental_all_dense.json"}[args.offload]
    if args.tokens < 2 or args.runs < 1 or args.temperature < 0:
        p.error("Use at least 2 tokens and 1 run")
    torch.set_num_threads(args.threads)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(ROOT / "assets/tiktoken")
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    ids = tokenizer.encode(args.prompt)
    if len(ids) + args.tokens > 1024:
        p.error("Prompt plus generated tokens exceeds the 1024-token context")
    model = GPT2(ROOT / "assets/gpt2/model.safetensors")
    report = {"date": datetime.now(timezone.utc).date().isoformat(), "model": "GPT-2 small pretrained",
              "checkpoint": json.loads((ROOT / "assets/gpt2/manifest.json").read_text()),
              "mode": {"none": "host CPU only", "qkv": "hybrid: block 0 QKV on Movidius; all other transformer math on CPU",
                       "all-dense": "hybrid: all learned dense projections on Movidius; attention, normalization, GELU and residuals on CPU"}[args.offload],
              "all_device_inference": False, "torch": torch.__version__,
              "threads": args.threads, "prompt": args.prompt, "prompt_ids": ids,
              "sampling": "greedy" if args.temperature == 0 else "top-40",
              "temperature": args.temperature, "seed": args.seed,
              "generated_tokens": args.tokens, "runs": []}
    # Compare complete-context evaluation with incremental decoding before timing.
    full = model.forward(ids, use_cache=False)[0, -1]
    model.reset()
    for token in ids:
        cached = model.forward([token])[0, -1]
    error = float((full-cached).abs().max())
    report["kv_cache_max_logit_error"] = error
    if error > 0.001:
        raise AssertionError(f"Cache mismatch: {error}")
    run(model, ids, 4)
    device = hook = None
    try:
        if args.offload != "none":
            device = NCS().__enter__()
            if args.offload == "qkv":
                hook = DeviceQKV(device, model)
                model.projection_hook = hook
            else:
                hook = DeviceDense(device, model)
                model.linear_hook = hook
        for i in range(args.runs):
            result = run(model, ids, args.tokens, args.temperature, args.seed)
            result["text"] = tokenizer.decode(ids + result["generated_ids"])
            report["runs"].append(result)
            print(result["text"], flush=True)
            print(f"decode: {result['decode_tokens_per_second']:.2f} tokens/s", flush=True)
        if hook:
            report["device_projection_calls"] = hook.calls
            report["device_projection_wall_seconds"] = hook.wall_seconds
            if args.offload == "qkv":
                report["device_weight_parameters"] = 768 * 2304 + 2304
            else:
                report["device_operations"] = hook.operations
                report["graph_upload_bytes"] = hook.upload_bytes
                report["weights_resident_on_device"] = "one projection at a time; other weights stored on host"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if hook:
            hook.close()
        if device:
            device.__exit__()
        path = ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
