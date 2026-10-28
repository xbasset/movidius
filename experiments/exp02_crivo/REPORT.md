# Crivo-17M on a 2017 Movidius stick

The complete Crivo-17M neural network runs on the original Intel Neural Compute
Stick (Myriad 2), including token embedding lookup, all seven encoder layers,
the decision head and the final softmax. The Mac tokenizes text, pads inputs,
transfers buffers over USB and displays the returned probabilities. There is no
host attention, host embedding lookup or CPU fallback in the stick execution path.
All weights remain resident in one graph. Only token IDs, masks and option
positions go in and the final probabilities come back; intermediate activations
are never transferred to the host between layers.

This is the second experiment in the time-machine project. The earlier GPT-2
experiment in [REPORT.md](../exp01_gpt2/REPORT.md) offloaded one projection; this experiment
uses a complete, much smaller, modern decision model.

Measured on 2026-10-27, on the original Myriad 2 stick and an Apple M2 Pro Mac
running macOS 26.6.2. All inference processes and the device runtime are native
ARM64.

**Result:** all 3,080 BANKING77 cases completed on the stick, with all 77 options
and no truncation or inference errors. Classification quality is retained.

| Backend | Macro-F1 | Accuracy | Median latency | p95 latency |
|---|---:|---:|---:|---:|
| Mac CPU, 4 threads, FP32 | 79.59% | 79.61% | 34.11 ms | 35.64 ms |
| Mac Metal GPU, FP32 | 79.59% | 79.61% | 9.13 ms | 9.73 ms |
| **Myriad 2 stick, FP16** | **79.64%** | **79.68%** | **661.32 ms** | **662.04 ms** |

The stick agrees with the CPU on **99.55%** of decisions: 14 of 3,080 differ.
It gets two more cases correct overall; this tiny difference is not evidence
of a meaningful quality improvement. The CPU is about 19 times faster and the
Metal GPU about 72 times faster in these implementations. This is a complete
execution demonstration on old hardware, not a speedup for this Mac.

Evidence: [comparison](results/crivo_assessment.json),
[CPU predictions](results/crivo_banking77_cpu.json),
[GPU predictions](results/crivo_banking77_mps.json),
[stick predictions](results/crivo_banking77_myriad.json),
[device stage profile](results/crivo_banking77_myriad.tsv).

The separate three-option decision demo takes about **52 ms** on the stick.
The full 77-intent demo takes about **662 ms** and identifies the supplied
missing-card message as `card_arrival`. Both were verified offline with
`--no-reference`: no model inference runs on the Mac in that mode.
[Short demo](results/crivo_demo.json), [BANKING77 demo](results/crivo_banking77_demo.json).

## Reproduce

On the tested Apple Silicon Mac, with Xcode command-line tools, CMake, Git, uv,
autoconf, automake, GNU libtool, make and wget available:

```sh
bash experiments/exp02_crivo/setup.sh
.venv/bin/python -m experiments.exp02_crivo.scripts.verify_crivo
.venv/bin/python -m experiments.exp02_crivo.scripts.verify_crivo_lifecycle

# No model weights are loaded or evaluated on the Mac in this mode.
.venv/bin/python -m experiments.exp02_crivo.scripts.run_crivo --no-reference

# Classify a short English message among the full set of 77 intents.
.venv/bin/python -m experiments.exp02_crivo.scripts.run_crivo --banking --no-reference \
  --text 'Where is my card? It has not arrived yet.' \
  --output experiments/exp02_crivo/results/crivo_banking77_demo.json
```

The plain demo chooses among refund, cancellation and information. Omitting
`--no-reference` also evaluates the independent Mac reference and compares
probabilities and latency. Input text is English, as in the model's training.

Plug in the original Myriad 2 stick. Run one device program at a time. Device
commands need ordinary host USB access; in a sandbox, use the host's USB
permissions. No sudo or permanent firmware flashing is involved. The stock
OpenVINO firmware is uploaded to RAM and the device resets on close.

Run the comparisons sequentially, without builds competing for CPU resources:

```sh
.venv/bin/python -m experiments.exp02_crivo.scripts.benchmark_crivo --device cpu
.venv/bin/python -m experiments.exp02_crivo.scripts.benchmark_crivo --device mps
.venv/bin/python -m experiments.exp02_crivo.scripts.benchmark_crivo --device myriad
.venv/bin/python -m experiments.exp02_crivo.scripts.assess_crivo
```

The stick's complete test takes roughly 35 minutes on this setup. A quick smoke
test can use `--limit 3 --output output/crivo_smoke.json`; it is not a benchmark
score. Raw per-case checkpoints are saved under ignored `results/raw/`.

## Model and benchmark

- Model: [tardellirs/crivo-17m](https://huggingface.co/tardellirs/crivo-17m),
  revision `8e42a0dbc309ca3360bfa94ebfd30061d6994ba7`.
- Architecture: Ettin/ModernBERT encoder, 7 layers, hidden size 256, 4 attention
  heads, head dimension 64, gated GELU MLP with intermediate size 384.
- The port implements `choice` decisions. The question-type embedding is fixed
  to `choice`; it does not claim support for the complete Jev API.
- The original weights are converted from FP32 to FP16. No retraining, pruning,
  option shortlist or substitution with another model is used.
- BANKING77: all 3,080 held-out messages, all 77 labels, original category order.
  The adapter matches the [Decision Index direct adapter](https://github.com/apolinario/decision-index/blob/main/decision_index/suite/build/normalize_direct.py).
  Its checked source hashes are in `scripts/fetch_crivo.py`.
- Input construction is `[CLS] instructions [SEP] state [SEP] [OPT] option ...`.
  BANKING77 uses the kit's empty JSON state and puts the customer request in the
  instructions. Option keys are `option_0` through `option_76`.
- Actual token lengths range from 878 to 955. The device graph pads to 960;
  padding keys are masked in every attention layer. Nothing is truncated.

The model card/board reports about 79.61% BANKING77 macro-F1. Our Mac run
achieves 79.59%. The public Decision Index
score of 22.15 is an aggregate chance-corrected skill score, not an accuracy.
This BANKING77 experiment is not a complete Decision Index submission.

The model was trained on related task families, including public benchmark
source training splits. This is a hardware implementation and quality-retention
test, not evidence of unseen-task general intelligence.

## Why a different runtime

The first experiment's NCSDK graph writer only exposed static dense operations.
OpenVINO 2020.3 is the last supported generation for Myriad 2 and provides
device operations for Gather, dynamic GEMM, reductions and Erf. These are enough
to express Crivo without replacing the firmware or moving attention onto the Mac.

The project builds the archived runtime natively for ARM64 and writes IR v7
directly. Neither an ONNX converter nor the old Python model optimizer is needed.
The CPU, GPU and GNA inference plugins are disabled in this build. The transport
bridge loads the network with the explicit `MYRIAD` backend.

Two host compatibility changes are recorded in
`../../scripts/patches/openvino-macos-arm64.patch`:

1. Skip the Linux-only cross-compilation protoc download when the ONNX importer
   is disabled. The installed CMake itself runs under Rosetta, so the supplied
   toolchain file explicitly targets native ARM64.
2. Preserve referenced XLink event semaphores until the waiting API call releases
   them. The old cleanup could destroy a semaphore before its waiter consumed
   the reset notification, triggering a second close of a freed USB handle on
   macOS. The patch serializes cleanup and release with the existing mutex and
   moves destruction of the dispatcher-closed notification to its consumer.

The complete benchmark uses the same neural graph and firmware as the delivered
demo. The final host patch also fixes the last shutdown-notification race; this
change is outside inference timing. Three consecutive open/infer/close cycles
and the delivered 77-intent demo were checked after that final fix.
[Lifecycle evidence](results/crivo_lifecycle_verification.json).

Native libusb 1.0.29 and the macOS frameworks come from the first experiment's
local build. Downloaded sources, binaries, weights and generated graphs remain
under ignored workspace directories.

## What executes where

| Work | Location |
|---|---|
| Text/JSON formatting and tokenization | Mac |
| Padding and deterministic attention-mask preparation | Mac |
| Token embedding table lookup | Stick, Gather |
| Layer normalization, including mean and variance | Stick, reductions and arithmetic |
| QKV and MLP projections | Stick, GEMM |
| Rotary positions | Stick arithmetic with precomputed graph constants |
| Global and local attention, QK and attention-times-V | Stick, dynamic GEMM and softmax |
| GELU and gating | Stick, Erf and arithmetic |
| Option-position lookup and decision head | Stick |
| Final distribution over the supplied options | Stick, softmax |
| Label decoding, ranking for display and benchmark scoring | Mac |

The graph has fixed sequence and option counts. The three-option demo uses
64 tokens; BANKING77 uses 960 tokens and 77 options. Longer inputs raise an error.
This is not an unrestricted long-context server or a text generator.

The port favors a direct implementation: local attention is evaluated as dense
attention with a window mask, and GELU uses Erf. Separate reductions and arithmetic
implement layer normalization. These choices preserve the model's computation
but leave substantial room for kernel fusion and sliding-window optimization.

The FP16 graph contains about 32.2 MiB of learned weights, plus rotary and mask
constants. Successful complete graph allocation demonstrates that this workload
fits; the earlier GPT-2 partial result did not establish that.

## Measurement boundaries

Latency is measured one request at a time, batch size one, after one warmup.
It includes input-buffer preparation and USB/runtime calls through receipt of
host-readable probabilities. It excludes model loading, graph compilation,
firmware boot and tokenization. The Mac reference uses variable input lengths;
the stick computes its fixed padded length. Raw per-case measurements and the
device's final-request stage profile are retained. Device profiling is enabled.

CPU results use four PyTorch threads and FP32 weights/operations with FP64 output
softmax. Metal GPU results use FP32, including output softmax. Device results use
FP16. The Mac Neural Engine is not measured. This is an ordinary laptop benchmark,
not a claim that this reference is the fastest possible Mac implementation.

**Energy measurement is outside this experiment's scope.** No instrumented
measurement of stick or whole-system electrical consumption is available. The
exploratory Mac component estimates have been removed from the interface and
results; this report compares prediction quality and latency only.

## Numerical checks

The explicit Mac reference was checked against the author's unmodified model
implementation on three complete BANKING77 inputs. Maximum logit error was
below 0.00002 and all three decisions matched.
[Reference check](results/crivo_reference_parity.json).
The macro-F1 implementation also exactly matches the official Decision Index
scorer on the complete CPU prediction set.
[Metric check](results/crivo_metric_verification.json).

FP16 outputs are not bit-identical to the FP32 reference. Across BANKING77, the
mean per-request maximum probability difference is **0.40 percentage points**;
the largest individual probability difference is **9.35 points**. Mean
KL(CPU || stick) is 0.01025 after normalizing FP16 softmax sums for that diagnostic
only. The largest raw softmax-sum error is 0.00167. Classification uses the raw
device outputs, without host softmax or probability correction.

The final-request profile contains **354 executed device stages**, with zero
host CPU time reported for those stages. This supports the execution audit;
the stronger guarantee is the complete graph loaded on the explicit MYRIAD-only
backend. The profile is a single-request snapshot, not the latency distribution.

## Code map

| File | Responsibility |
|---|---|
| `scripts/fetch_crivo.py` | Pinned, hash-checked public weights, tokenizer and BANKING77 inputs |
| `model.py` | Exact input layout, explicit Mac reference, benchmark metric |
| `scripts/verify_crivo.py` | Independent check against the author's original implementation |
| `scripts/verify_crivo_lifecycle.py` | Repeated physical-device open/infer/close regression check |
| `../../movidius_lab/legacy_ir.py` | Small static graph serializer |
| `scripts/export_crivo.py` | Complete Crivo graph, including attention and decision head |
| `../../movidius_lab/myriad_bridge.cpp` | OpenVINO lifecycle, USB buffers and profiling; no model math |
| `../../movidius_lab/myriad.py` | Python transport wrapper and shape checks |
| `scripts/run_crivo.py` | Demonstration and optional Mac comparison |
| `scripts/benchmark_crivo.py` | Full test-set quality and single-request latency |
| `scripts/assess_crivo.py` | Cross-backend quality, decision and probability comparison |

## Historical claim and licensing

The claim concerns running a 2026 pretrained model on 2017 hardware. It does not
establish that its training data, discovery or training compute were available
in 2017. The stick launched after the Transformer paper appeared and before GPT.
This experiment uses a 2020 stock firmware/runtime, not a July 2017 software stack.

Crivo's declared license is CC BY-NC 4.0. Its model card and upstream data terms
remain applicable. OpenVINO's sources retain their Apache 2.0 notices; vendor
firmware and libusb retain their respective licenses. Setup fetches dependencies
from their upstream locations instead of adding them to this repository.
