# Full MVTec AD: DINOv2-S + AnomalyDINO on the 2017 stick

**At 448 × 448, all 15 categories and all 1,725 test images run on Myriad 2.**
Three independent one-shot reference choices give **96.54% mean image AUROC**,
compared with **96.52% on the Mac** and **96.5% in the paper**.
The largest absolute mean difference from CPU across all six metrics is
**0.038 percentage points**.
The complete encoder, PCA mask, nearest-neighbor search and image score execute
on the stick. No test images were used for training or reference selection.

This is the full benchmark extension of the [224-square pilot](REPORT_224.md).
[Protocol and reproduction commands](PROTOCOL_448.md) describe the pinned
sources, inference boundaries, solver approximation and metric definitions.

## Quality against the paper

Target: [Table 2, AnomalyDINO-S (448), one-shot, agnostic](https://arxiv.org/html/2405.14529v3#S4.T2).
All entries are percentages: mean ± population standard deviation of three
reference trials, after equal-weight category averaging within each trial.

| Metric | Paper | Stick | Mac CPU | Mac GPU |
|---|---:|---:|---:|---:|
| Image AUROC | 96.5 ± 0.4 | 96.54 ± 0.39 | 96.52 ± 0.40 | 96.52 ± 0.40 |
| Image F1-max | 96.0 ± 0.2 | 95.99 ± 0.27 | 95.98 ± 0.21 | 95.98 ± 0.21 |
| Image AP | 98.1 ± 0.3 | 98.13 ± 0.22 | 98.15 ± 0.25 | 98.15 ± 0.25 |
| Pixel AUROC | 96.3 ± 0.1 | 96.28 ± 0.15 | 96.26 ± 0.14 | 96.26 ± 0.14 |
| Pixel F1-max | 57.9 ± 0.8 | 57.98 ± 0.79 | 57.94 ± 0.78 | 57.94 ± 0.78 |
| AUPRO ≤ 0.3 | 91.7 ± 0.1 | 91.76 ± 0.05 | 91.72 ± 0.06 | 91.72 ± 0.06 |

The Mac reference reproduces all six published mean values at the paper's
one-decimal precision. The device uses FP16 and replaces randomized SVD with
64 power iterations; its results are an implementation comparison rather than
a claim of bitwise equivalence. AUROC measures ranking quality. F1-max is an
oracle evaluation statistic over thresholds, not a selected deployment threshold.

## Single-reference latency

Measured on one Intel NCS1 / Myriad 2 and one Apple M2 Pro with 16 GiB RAM.
Each backend processes the same **150 batch-one requests**: ten evenly spaced
test IDs per category, the first reference and eight rotations, three warmups
per category. Quality inference and pixel evaluation have finished before timing.

| Backend | Median | p95 |
|---|---:|---:|
| Myriad 2 stick, FP16 | 2743.38 ms | 2746.78 ms |
| Mac CPU, FP32 | 112.40 ms | 119.93 ms |
| Mac GPU / Metal, FP32 | 33.27 ms | 35.31 ms |

Prepared host tensor → returned 32 × 32 map and image score; includes conversion,
transfers, query masking and synchronization. Graph load, reference calibration,
image decoding/resizing and pixel-metric postprocessing are excluded. CPU uses
four PyTorch threads. The Metal baseline performs the model and matching on GPU
and the author's sklearn PCA on CPU. The stick calculates PCA in its fixed graph
even for categories where the resulting mask is disabled. Core ML and the
Apple Neural Engine are not benchmarked.

The Mac is 24.4× faster on CPU and
82.5× faster on Metal for these implementations.
Energy is excluded.

For quality evaluation, one encoder pass feeds three isolated one-shot heads:
**5,175 image/reference evaluations per backend**. Those joint timings in the
prediction archive are diagnostic throughput, not the single-head timings above.
On the matched latency requests, the maximum score difference between the
single-head graph and the first joint head is
**0.000000** across the three backends.

## Per-category image AUROC

Mean of the three reference trials, in %. Small categories have the same weight
as larger categories in the final macro average.

| Category | Test images | Stick | CPU | GPU |
|---|---:|---:|---:|---:|
| bottle | 83 | 99.44 | 99.47 | 99.47 |
| cable | 150 | 91.37 | 91.60 | 91.60 |
| capsule | 132 | 90.39 | 90.03 | 90.03 |
| carpet | 117 | 100.00 | 100.00 | 100.00 |
| grid | 78 | 99.14 | 99.19 | 99.19 |
| hazelnut | 110 | 97.95 | 98.01 | 98.01 |
| leather | 124 | 100.00 | 100.00 | 100.00 |
| metal_nut | 115 | 99.92 | 99.89 | 99.89 |
| pill | 167 | 92.52 | 92.33 | 92.33 |
| screw | 160 | 89.17 | 89.08 | 89.08 |
| tile | 117 | 100.00 | 99.98 | 99.98 |
| toothbrush | 42 | 99.63 | 99.63 | 99.63 |
| transistor | 100 | 90.90 | 90.92 | 90.92 |
| wood | 79 | 98.63 | 98.83 | 98.83 |
| zipper | 151 | 98.98 | 98.85 | 98.85 |

The six metrics for every category/reference are in
[stick](results/448/dino448_myriad_metrics.json),
[CPU](results/448/dino448_cpu_metrics.json) and
[GPU](results/448/dino448_mps_metrics.json).

## Numerical drift and execution evidence

Against CPU, the stick's score mean absolute error is **0.004721**,
maximum **0.029911**. Patch-distance mean absolute error is
**0.002595**, maximum **0.494629**.
On the five masked categories, nonzero map support differs at
**1,088 / 625,664 positions (0.174%)**.
This combines solver differences, FP16 feature drift and zero rounding. The
separate same-input PCA checks isolate the solver in
[verification](results/448/dino448_verification.json).
Reference self-distances range from 0.004250 to 0.005615;
FP16 can yield a positive distance even for the reference image itself.

The [device profile](results/448/dino448_myriad.tsv) records
**965 executed stages**, including
**419 PCA stages**, and **0 CPU kernel microseconds**.
The host bridge explicitly loads `MYRIAD` only. It has no fallback model.
The CPU still handles image preparation, tensor transport and metric evaluation.

At 1,025 tokens, our legacy runtime corrupted tail rows of constant-vector
broadcasts. Explicitly expanding the constants fixes this without changing the
model values; it grows the IR weight file to about 187 MB while retaining the
same 22.1M learned parameters. The
[broadcast reproducer](results/448/dino448_broadcast_verification.json) documents the
failure and correction. Firmware remains the stock RAM-loaded image.

[Independent source verification](results/448/dino448_reference_verification.json)
checks the explicit reference against Meta and the author's mask code.
The device verification also checks complete feature outputs and confirms that
the previous 224-square XML and weights remain unchanged. The streaming pixel
metric checks cover score ties and chunk boundaries against sklearn and direct
PRO thresholds. All original-resolution pixels are evaluated.

## Published evidence and scope

- [Assessment and graph/firmware hashes](results/448/dino448_assessment.json).
- [Per-image labels, three scores and joint timing for every backend](results/448/dino448_predictions.json).
- [Dataset revision and hashes of all selected files](results/448/dino448_dataset.json).
- Single-head latency requests: [stick](results/448/dino448_myriad_latency.json),
  [CPU](results/448/dino448_cpu_latency.json), [GPU](results/448/dino448_mps_latency.json).

No raw camera images, USB identifiers, private host metadata, pretrained weights
or generated graph binaries are published. Dataset/model terms are recorded in
[THIRD_PARTY.md](../../THIRD_PARTY.md). Reference choice, one physical device and this
specific runtime constrain the comparison; the reported standard deviations
cover reference variation only. The camera demo remains the 224-square pilot.

The historical result concerns **2017 hardware executing a later visual
Transformer and a 2024–2025 method**. It does not establish that their training
data, recipe or weights were available in 2017. The host runtime dates from 2020.
