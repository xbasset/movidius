# GPT-2 meets a 2017 Movidius stick

Measured on 2026-10-26 to 2026-10-27, Apple M2 Pro, macOS 26.6.2, native ARM64.

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
| Real GPT-2 QKV, device kernel | 1.678 ms | [JSON](results/gpt2_qkv_device.json) |
| Same QKV, including USB/API | 5.322 ms | Same JSON; ten measured vectors, one warmup |
| GPT-2 vocabulary head, device kernel | 36.240 ms | [JSON](results/gpt2_lm_head_device.json) |
| Same vocabulary head, including USB/API | 39.737 ms | Same JSON; three measured vectors, one warmup |
| GPT-2 CPU decoding, four threads | 59.39 tokens/s | [JSON](results/gpt2_host.json) |
| GPT-2 with first QKV offloaded | 44.89 tokens/s | [JSON](results/gpt2_hybrid.json) |
| Greedy generated IDs match CPU | True, all three runs | 64 generated tokens per run |
| Independent Transformers reference | Max cached logit error 0.000397 | [JSON](results/reference_verification.json) |
| Fixed 256-token top-1 agreement | 100.0% | [JSON](results/hybrid_parity.json) |
| Fixed-slice perplexity, CPU / hybrid | 41.51421 / 41.50544 | Same JSON |
| Relative perplexity change | -0.0211% | Same JSON |
| Mean probability-distribution KL | 0.00000141 | Same JSON |
| Power / joules per token | **Not measured** | Requires external measurement |

The vocabulary head is the real **768×50,257** matrix: 73.62
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
median relative L2 error is 0.294%.
The language parity test compares the hybrid against original FP32 weights.

The parity slice is the first 256 GPT-2 tokens of the tokenized WikiText-2 test
file from PyTorch examples, evaluated as one causal window. It is an
implementation smoke test, **not a full WikiText-2 benchmark**. Raw maximum logit
error is 1.4522; likelihood, KL and top-1 agreement are
also reported because raw logits alone do not describe probability agreement.

## A reproducible text sample

Sampling: top-40, temperature 0.8, seed 1337. This is unedited model output,
not a statement of fact. [Full text, token IDs and timing](results/gpt2_sample.json).

> In 2017, a scientist built a tiny computer that could predict earthquake events. The project came a month after an earthquake in the Chinese city of Kunming happened. It was reported on by local media on Wednesday, the day after the earthquake was reported.
>
> The team used a computer to create a simulation that made predictions on a set of earthquakes. The map is shown in the accompanying graphic.
>
> The team believes it is possible to develop a computer model that can predict earthquakes, especially in a seismically unstable area where there

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
- Firmware SHA256: `bd0ef7bed794379d4c77f4d2434dfa0da905acdd2abbf8df64a23d69cc5a49c0`.
- GPT-2 revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Model SHA256: `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.
- [Source and dependency notes](THIRD_PARTY.md); [reproduction instructions](README.md).
