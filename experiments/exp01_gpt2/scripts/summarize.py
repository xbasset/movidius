"""Render measured JSON artifacts into a short, auditable experiment report."""

import json
import statistics as S

from movidius_lab.device import ROOT


def read(name):
    return json.loads((ROOT / "experiments/exp01_gpt2/results" / f"{name}.json").read_text())


def main():
    cpu, hybrid = read("gpt2_host"), read("gpt2_hybrid")
    qkv, head = read("gpt2_qkv_device"), read("gpt2_lm_head_device")
    parity, reference = read("hybrid_parity"), read("reference_verification")
    image, sample = read("device_probe"), read("gpt2_sample")
    if cpu["mode"] != "host CPU only" or "block 0 QKV" not in hybrid["mode"]:
        raise ValueError("CPU and hybrid result files must contain their respective execution modes")
    if cpu["prompt_ids"] != hybrid["prompt_ids"] or cpu["generated_tokens"] != hybrid["generated_tokens"]:
        raise ValueError("CPU and hybrid benchmark inputs must match")
    dates = sorted({r.get("date", r.get("utc", "").split("T")[0])
                    for r in [cpu, hybrid, image, sample]})
    date_label = " to ".join([dates[0], dates[-1]]) if len(dates) > 1 else dates[0]
    cpu_tps = S.median(r["decode_tokens_per_second"] for r in cpu["runs"])
    hybrid_tps = S.median(r["decode_tokens_per_second"] for r in hybrid["runs"])
    matches = all(r["generated_ids"] == cpu["runs"][0]["generated_ids"] for r in hybrid["runs"])
    qkv_ms = S.median(m["device_ms"] for m in qkv["measurements"])
    qkv_usb_ms = 1000*S.median(m["usb_wall_seconds"] for m in qkv["measurements"])
    head_ms = S.median(m["device_ms"] for m in head["measurements"])
    head_usb_ms = 1000*S.median(m["usb_wall_seconds"] for m in head["measurements"])
    ref_error = max(c["cached_max_logit_error"] for c in reference["cases"])
    sample_quote = "\n".join(">" + (" " + line if line else "") for line in sample["runs"][0]["text"].splitlines())
    report = f"""# GPT-2 meets a 2017 Movidius stick

Measured on {date_label}, Apple M2 Pro, macOS 26.6.2, native ARM64.

## What works

The original Intel Movidius Neural Compute Stick executes the **actual first
attention projection of pretrained GPT-2 small** during live text generation.
The Mac executes the rest of the model. The 768×2304 matrix and its bias contain
1,771,776 parameters, about 1.42% of GPT-2 small's parameter count. This is a
working partial offload, not standalone GPT-2 inference on the stick.

No replacement firmware was necessary for this milestone. A small custom graph
writer feeds dense matrices to the stock NCSDK firmware. The inference process
and USB library are native ARM64. No VM, Docker, Rosetta inference, or global SDK
install was used for the successful path.

## Measured results

| Test | Result | Evidence |
|---|---:|---|
| Real GPT-2 QKV, device kernel | {qkv_ms:.3f} ms | [JSON](results/gpt2_qkv_device.json) |
| Same QKV, including USB/API | {qkv_usb_ms:.3f} ms | Same JSON; ten measured vectors, one warmup |
| GPT-2 vocabulary head, device kernel | {head_ms:.3f} ms | [JSON](results/gpt2_lm_head_device.json) |
| Same vocabulary head, including USB/API | {head_usb_ms:.3f} ms | Same JSON; three measured vectors, one warmup |
| GPT-2 CPU decoding, four threads | {cpu_tps:.2f} tokens/s | [JSON](results/gpt2_host.json) |
| GPT-2 with first QKV offloaded | {hybrid_tps:.2f} tokens/s | [JSON](results/gpt2_hybrid.json) |
| Greedy generated IDs match CPU | {matches}, all three runs | 64 generated tokens per run |
| Independent Transformers reference | Max cached logit error {ref_error:.6f} | [JSON](results/reference_verification.json) |
| Fixed 256-token top-1 agreement | {100*parity['top1_agreement']:.1f}% | [JSON](results/hybrid_parity.json) |
| Fixed-slice perplexity, CPU / hybrid | {parity['host_slice_perplexity']:.5f} / {parity['hybrid_slice_perplexity']:.5f} | Same JSON |
| Relative perplexity change | {100*parity['relative_perplexity_change']:+.4f}% | Same JSON |
| Mean probability-distribution KL | {parity['mean_KL_reference_to_hybrid']:.8f} | Same JSON |
| Power / joules per token | **Not measured** | Requires external measurement |

The vocabulary head is the real **768×50,257** matrix: {head['weight_storage_bytes']/2**20:.2f}
MiB of FP16 weights. Its successful standalone execution tests the largest GPT-2
weight tensor directly. That does not prove that every tensor can remain resident
simultaneously.

For the complete model, the calculated storage budget is approximately **237 MiB
of FP16 weights + 36 MiB of FP16 KV cache at 1,024 tokens = 273 MiB**, before
firmware and scratch buffers. Physical RAM is approximately 512 MiB. This is a
memory estimate, not a demonstrated complete-model allocation. The archived
compiler's 320 MiB graph-file limit is a separate software constraint.

The hybrid is slower than the CPU-only reference. Its tokens/s figure includes
both the Mac and stick; it must never be described as the stick's GPT-2 speed.
Host and hybrid measurements were run separately with identical greedy prompts,
four CPU threads, 64 generated tokens and three runs. Decode timing excludes
model loading, firmware boot, graph upload, initial prompt processing and the
first generated token; time to first token is saved separately. The JSON retains
all run timings. This is an ordinary laptop, not a controlled benchmark lab.

Device timings are stock firmware's reported stage times; USB/API wall time uses
the host monotonic clock. Dense tests use deterministic random inputs and FP16
weights/input, comparing with FP32 accumulation of the rounded values. The QKV
median relative L2 error is {100*S.median(m['relative_l2_error'] for m in qkv['measurements']):.3f}%.
The language parity test compares the hybrid against original FP32 weights.

The parity slice is the first 256 GPT-2 tokens of the tokenized WikiText-2 test
file from PyTorch examples, evaluated as one causal window. It is an
implementation smoke test, **not a full WikiText-2 benchmark**. Raw maximum logit
error is {parity['max_logit_error']:.4f}; likelihood, KL and top-1 agreement are
also reported because raw logits alone do not describe probability agreement.

## A reproducible text sample

Sampling: top-40, temperature 0.8, seed 1337. This is unedited model output,
not a statement of fact. [Full text, token IDs and timing](results/gpt2_sample.json).

{sample_quote}

## What the time-machine claim can currently say

**A 2017 Movidius accelerator can compute genuine GPT-2 matrix operations and
participate in generating text from the 2019 pretrained model.**

It cannot yet say that GPT-2 runs entirely on the stick, that it matches any
larger GPT model, or that it generates at a particular wattage. The host is a
modern laptop. The host bridge was adapted from a 2018 fork; the graph format
was inspected using an archived 1.12 package. This experiment is not yet an
audit proving every piece of software was available at the July 2017 launch.

## What failed and what to try next

Streaming all learned projections through one open device failed when allocating
block 0's MLP expansion graph (NCSDK status -2). The same 768×3072 dimensions
work as a fresh standalone graph. Reserving the largest blob and I/O buffers
first did not resolve it. The root cause is not established; buffer/protocol
lifetime is a hypothesis. [Failure transcript](results/failed_experiments/stream_all_dense.log).

Next: solve graph replacement or build a resident execution graph. Then move
normalization, GELU, attention and KV state onto the device, preserve logits on
larger held-out tests, and measure USB power. Only after that can we claim
whole-model device speed and energy.

The alternative custom-firmware route still needs a usable Myriad 2 SHAVE
toolchain and board initialization code. No replacement executable or unsigned
firmware boot has been demonstrated here. The stock boot transfers a vendor
firmware image into RAM and close resets the device.

## Provenance

- Hardware ID: `03e7:2150` before boot, `03e7:f63b` after boot.
- Firmware SHA256: `{image['firmware_sha256']}`.
- GPT-2 revision: `{cpu['checkpoint']['revision']}`.
- Model SHA256: `{cpu['checkpoint']['files']['model.safetensors']['sha256']}`.
- [Source and dependency notes](../../THIRD_PARTY.md); [reproduction instructions](README.md).
"""
    (ROOT / "experiments/exp01_gpt2/REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
