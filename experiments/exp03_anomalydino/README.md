# 03 — DINOv2-S + AnomalyDINO: visual anomalies

After [02 — complete text decisions](../exp02_crivo/), the third experiment
runs a complete visual Transformer and anomaly-scoring pipeline on the stick.
There are **two successive milestones within this experiment**, with separate
protocols and evidence.

## 03a — 224 × 224 pilot and camera demo

First, the complete **83-image MVTec AD bottle test split**, three separate
one-shot references, no rotations or foreground masking:

| Backend | Image AUROC | Median latency |
|---|---:|---:|
| Stick | 99.15% | 546 ms |
| Mac CPU | 99.23% | 36 ms |
| Mac GPU / Metal | 99.23% | 15 ms |

This established complete device encoding, patch matching and scoring, then
reference calibration and live inference from the Mac camera. It is a one-category
pilot, not the full paper benchmark. See the [224 report and camera guide](REPORT_224.md)
and [224 measurements](results/224/).

```sh
# From the repository root, after the shared prerequisites.
bash experiments/exp03_anomalydino/setup_224.sh
.venv/bin/python -m experiments.exp03_anomalydino.scripts.serve_dino
```

Open <http://127.0.0.1:8766/> in Chrome. Start with a public bottle example;
optionally activate the camera, capture a normal reference and select stick-only
live inference. The server does not save camera images. Stop it before a benchmark.

## 03b — 448 × 448, all 15 categories, paper comparison

Then, **all 1,725 test images**, three independent one-shot trials, eight
rotations per reference and PCA foreground masking for the five specified
categories. The complete encoder, PCA approximation, mask morphology,
nearest-neighbor matching and score aggregation execute on the stick.

| Backend | Image AUROC | Median single-reference latency |
|---|---:|---:|
| Stick | 96.54% | 2,743 ms |
| Mac CPU | 96.52% | 112 ms |
| Mac GPU / Metal | 96.52% | 33 ms |
| Paper, matching 448 one-shot agnostic protocol | 96.5% | — |

The [448 report](REPORT_448.md) compares all six paper metrics and the three
backends. The [448 protocol](PROTOCOL_448.md) gives the complete commands and
documents the device's FP16 power-iteration PCA approximation and legacy-runtime
broadcast correction. [448 results](results/448/) include all image scores,
per-category metrics, dataset hashes, latency requests and device profile.

```sh
bash experiments/exp03_anomalydino/setup_448.sh
# Continue with quality, metric and latency commands in PROTOCOL_448.md.
```

The camera demo remains at **224 × 224**. The full benchmark compares quality
and latency at 448; no 448 camera performance or energy measurement is claimed.
The 224 and 448 AUROCs cover different test populations and preprocessing and
should not be interpreted as a resolution-only comparison.

## Code and evidence

| Path | Role |
|---|---|
| [model.py](model.py) | DINOv2 reference, 224/448 preprocessing and feature extraction |
| [paper.py](paper.py) | Complete 448 paper pipeline for Mac and stick |
| [pca_ir.py](pca_ir.py) | Device PCA mask and morphology graph |
| [metrics.py](metrics.py) | Image/pixel metrics and bounded-memory exact threshold evaluation |
| [live.py](live.py), [web/](web/) | 224 reference lifecycle and camera/image interface |
| [scripts/](scripts/) | Pinned fetch/export, 224/448 benchmarks, parity, metric and runtime checks |
| [results/224/](results/224/), [results/448/](results/448/) | Separate published snapshots from the two milestones |

The same 22.1M-parameter checkpoint is used for both resolutions; no MVTec
fine-tuning is performed. The shared runtime is reused from Crivo, while the
experiment has no code dependency on the Crivo model or interface.

Original milestones: [224 + camera](https://github.com/xbasset/movidius/commit/583b8f0e597409193074f71e733a1b53c86c6330),
[camera recovery check](https://github.com/xbasset/movidius/commit/f3b780bf749be8d41fa1469bc92711eaef9ba9a2),
[448 / all 15 categories](https://github.com/xbasset/movidius/commit/76d8fbbd71187de18e8c99b7ecfb350711b0ea51).
