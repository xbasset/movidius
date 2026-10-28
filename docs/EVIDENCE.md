# Evidence index

Technical sequence and source material for the three experiments. Each claim
below links to the implementation report and its measured artifacts. The blog
article is a separate project.

## 01 — Real GPT-2 computation, partial offload

**Established:** stock Myriad 2 firmware executes actual GPT-2 matrices. During
generation, block 0's QKV projection runs on the stick; the Mac executes the
remaining operations. The complete vocabulary projection also runs as a
standalone device test.

| Evidence | What it establishes |
|---|---|
| [Report](../experiments/exp01_gpt2/REPORT.md), [commands](../experiments/exp01_gpt2/) | Execution and timing boundaries |
| [QKV](../experiments/exp01_gpt2/results/gpt2_qkv_device.json), [vocabulary projection](../experiments/exp01_gpt2/results/gpt2_lm_head_device.json) | Real weights, numerical error, kernel and USB/API timings |
| [CPU generation](../experiments/exp01_gpt2/results/gpt2_host.json), [hybrid generation](../experiments/exp01_gpt2/results/gpt2_hybrid.json) | Same prompt and greedy token IDs; CPU 59.39 versus hybrid 44.89 tokens/s |
| [Reference verification](../experiments/exp01_gpt2/results/reference_verification.json), [fixed-token parity](../experiments/exp01_gpt2/results/hybrid_parity.json) | Independent reference, logits, likelihood and probability drift |
| [Seeded sample](../experiments/exp01_gpt2/results/gpt2_sample.json) | Unedited generated text, token IDs and timing |
| [Failed wider offload](../experiments/exp01_gpt2/results/failed_experiments/stream_all_dense.log) | The attempted all-dense streaming path did not complete |

**Limit:** no complete GPT-2 device inference or full language-model benchmark.
The hybrid tokens/s figure includes the Mac. The next milestone uses a smaller
complete decision model and a different stock runtime.

## 02 — Complete Crivo-17M decisions

**Established:** embeddings, all seven encoder layers, decision head and
softmax execute in one resident MYRIAD-only graph. The Mac supplies token IDs,
masks and option positions; it receives probabilities.

| Evidence | What it establishes |
|---|---|
| [Report and protocol](../experiments/exp02_crivo/REPORT.md), [local demo](../experiments/exp02_crivo/DEMO.md) | Full execution boundary, benchmark and interactive interface |
| [Assessment](../experiments/exp02_crivo/results/crivo_assessment.json) | All 3,080 BANKING77 cases, 79.64% stick versus 79.59% Mac macro-F1 |
| [Stick predictions](../experiments/exp02_crivo/results/crivo_banking77_myriad.json), [CPU](../experiments/exp02_crivo/results/crivo_banking77_cpu.json), [Metal](../experiments/exp02_crivo/results/crivo_banking77_mps.json) | Per-case predictions and latency, with all 77 intents |
| [Stage profile](../experiments/exp02_crivo/results/crivo_banking77_myriad.tsv) | 354 executed device stages, zero reported CPU kernel time |
| [Reference](../experiments/exp02_crivo/results/crivo_reference_parity.json), [metric](../experiments/exp02_crivo/results/crivo_metric_verification.json), [lifecycle](../experiments/exp02_crivo/results/crivo_lifecycle_verification.json), [live interface](../experiments/exp02_crivo/results/crivo_live_verification.json) | Independent model/scorer checks and working device/interface lifecycle |

**Limit:** a closed-set English banking-intent task, not the complete Jev API or
Decision Index leaderboard. The Mac is faster. Exploratory energy estimates
were removed before the visual experiment; no energy result is retained here.

## 03 — Complete visual anomalies, then the full paper benchmark

### 03a — 224 × 224 bottle pilot and camera

**Established:** all twelve DINOv2-S blocks plus reference matching and image
scoring run on the stick. Three independent one-shot references cover all 83
bottle test images. The local camera demo calibrates on the stick and can run
stick-only live inference.

| Evidence | What it establishes |
|---|---|
| [224 report, commands and camera guide](../experiments/exp03_anomalydino/REPORT_224.md) | Pilot protocol and host/device boundary |
| [Assessment](../experiments/exp03_anomalydino/results/224/dino_assessment.json), [stick predictions](../experiments/exp03_anomalydino/results/224/dino_bottle_myriad.json) | 99.15% image AUROC, 546 ms median; CPU/GPU comparison and hashes |
| [Live integration](../experiments/exp03_anomalydino/results/224/dino_live_verification.json), [camera verification](../experiments/exp03_anomalydino/results/224/dino_camera_verification.json) | Reference replacement, real camera capture and device inference; recovery after Mac locking |

**Limit:** one category, no foreground mask or reference rotations. Camera
verification demonstrates functionality, not camera anomaly accuracy.

### 03b — 448 × 448, all 15 MVTec AD categories

**Established:** all 1,725 test images, three independent one-shot trials,
eight rotations per reference, device PCA masking and all six paper metrics.

| Evidence | What it establishes |
|---|---|
| [448 report](../experiments/exp03_anomalydino/REPORT_448.md), [protocol](../experiments/exp03_anomalydino/PROTOCOL_448.md) | Exact comparison to AnomalyDINO-S (448), one-shot, agnostic |
| [Assessment](../experiments/exp03_anomalydino/results/448/dino448_assessment.json) | 96.54% image AUROC stick, 96.52% Mac, 96.5% paper; six metrics and numerical drift |
| [All predictions](../experiments/exp03_anomalydino/results/448/dino448_predictions.json), [dataset hashes](../experiments/exp03_anomalydino/results/448/dino448_dataset.json) | Per-image scores for each backend/reference and identity of every selected input |
| [Stick metrics](../experiments/exp03_anomalydino/results/448/dino448_myriad_metrics.json), [CPU](../experiments/exp03_anomalydino/results/448/dino448_cpu_metrics.json), [Metal](../experiments/exp03_anomalydino/results/448/dino448_mps_metrics.json) | Per-category, per-reference scores; macro averaging before trial statistics |
| [Single-head latency: stick](../experiments/exp03_anomalydino/results/448/dino448_myriad_latency.json), [CPU](../experiments/exp03_anomalydino/results/448/dino448_cpu_latency.json), [Metal](../experiments/exp03_anomalydino/results/448/dino448_mps_latency.json) | Same 150 requests: 2,743 / 112 / 33 ms median; separate from joint evaluation throughput |
| [Device verification](../experiments/exp03_anomalydino/results/448/dino448_verification.json), [source verification](../experiments/exp03_anomalydino/results/448/dino448_reference_verification.json), [broadcast reproducer](../experiments/exp03_anomalydino/results/448/dino448_broadcast_verification.json), [stage profile](../experiments/exp03_anomalydino/results/448/dino448_myriad.tsv) | PCA checks, legacy broadcast correction, 965 executed stages and zero reported CPU kernel time |

**Limit:** device PCA uses 64 FP16 power iterations instead of the author's
randomized SVD; the difference is documented and assessed. The camera demo
remains at 224. Pilot/full scores use different populations and protocols and
cannot establish a resolution-only improvement or regression.

## Original checkpoints, in order

These commits preserve the code and documentation as they existed at each
milestone, before the repository was reorganized. All recorded commits are from
2026-10-27; the order, rather than the date, distinguishes the steps.

| Order | Checkpoint | Commit |
|---|---|---|
| 01 | GPT-2 partial offload | [8ed64d4](https://github.com/xbasset/movidius/commit/8ed64d4f2080521755b56abdaedcfb901597cc70) |
| 02 | Crivo complete model and BANKING77 | [f686c61](https://github.com/xbasset/movidius/commit/f686c61cd0857dcd78cfd3751461bc19b8b0af36) |
| 02 | Local decision interface | [1d8ed87](https://github.com/xbasset/movidius/commit/1d8ed873884d1d9152d288e8e4efc5e430ac9117) |
| 02 | Energy estimates removed from the experiment | [f2088e8](https://github.com/xbasset/movidius/commit/f2088e8ff9e67b0028608a77f6cd912a0f8f07de) |
| 03a | DINO 224 pilot and camera demo | [583b8f0](https://github.com/xbasset/movidius/commit/583b8f0e597409193074f71e733a1b53c86c6330) |
| 03a | Camera recovery rechecked | [f3b780b](https://github.com/xbasset/movidius/commit/f3b780bf749be8d41fa1469bc92711eaef9ba9a2) |
| 03b | DINO 448, complete 15-category benchmark | [76d8fbb](https://github.com/xbasset/movidius/commit/76d8fbbd71187de18e8c99b7ecfb350711b0ea51) |

## Artifact integrity after reorganization

[evidence.json](evidence.json) enumerates **all 48** previously published JSON,
TSV and failure-log files. For each, it records its old path, current path, byte
count, SHA256 and the last commit that changed it. The baseline is `76d8fbb`.
Every file was moved without changing its contents. Documentation links and
runtime output paths were updated separately.

Some artifact fields still contain historical paths such as `results/...` or
`scripts/...`. Interpret them in their recorded commit; use the manifest to
resolve moved evidence files in the current checkout. Hashes cover file bytes,
not timestamps or new directory names. The manifest does not independently
certify that the original measurements were correct.

```sh
# No model, network or USB access required; standard library only.
python3 -m scripts.check_repository
# With Git history present, also compare each artifact to its recorded commit.
python3 -m scripts.check_repository --history
```

The checker also validates local Markdown links, documented Python/shell entry
points and separation between experiment imports. It fails if a published
measurement snapshot changes; a new experiment run should retain its own
evidence rather than silently replacing the historical record.

Across all three experiments, the host handles input preparation, USB transport,
metrics and display. Device-only neural inference is established for Crivo and
AnomalyDINO, with the explicit boundaries in their reports. No energy advantage,
training on the stick, full GPT-2 device execution or original 2017 software
availability is claimed. Sources and their terms remain in
[THIRD_PARTY.md](../THIRD_PARTY.md).

## Reorganization validation — 2026-10-28

The reorganization was checked on the same Mac and physical stick:

- All 48 measurement files match both their SHA256 entries and their recorded Git commits.
- Shared/experiment modules import, documented entry points resolve, and local Markdown links and shell syntax validate.
- GPT-2 generates text again through the moved hybrid entry point.
- Both local interfaces serve their moved HTML/JS/CSS and pass the existing real stick/CPU/Metal API checks, including request bounds, serialized jobs and reference replacement/restoration for DINO.
- One masked 448 test image, `capsule/test/crack/000.png`, reproduces its first-reference cached patch map and score exactly on each of CPU, Metal and stick (maximum absolute error 0.0).
- The independent pixel AUROC, F1-max and AUPRO checks still pass, including ties across chunk boundaries. Both generated reports read the relocated evidence successfully.

Fresh smoke outputs were written under ignored `output/`; historical results
were not overwritten. These are regression checks, not a new complete benchmark
run. No physical camera was activated during these checks; the previous camera
verification remains the evidence for that milestone.
