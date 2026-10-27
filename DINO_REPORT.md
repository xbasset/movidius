# DINOv2 Small + AnomalyDINO on Myriad 2

**A complete visual anomaly detector runs on the original 2017 Movidius stick.**
At 224 × 224, all twelve DINOv2-S/14 layers, feature normalization, nearest-neighbor
distances and image-score aggregation execute in a MYRIAD-only graph. On the
complete MVTec AD **bottle** test split, three one-shot reference choices give
**99.15% mean image AUROC, 98.64% mean pixel AUROC and 546 ms median inference**.
The camera capture, reference calibration and live inference were also tested
with the Mac's physical FaceTime camera in Chrome.

This is a one-category pilot, not a result on all fifteen MVTec AD categories.
It evaluates detection/localization quality and latency. Energy is excluded.

## Results

Measured on Intel Movidius NCS1 / Myriad 2 and Apple M2 Pro, macOS 26.6.2.
Each backend processes the same 83 test images for each reference: **249 measured
requests per backend**, batch size one. The three trials use respectively
`train/good/000.png`, `001.png`, and `002.png`, with no test-image selection.

| Backend | Precision | Image AUROC, mean | Pixel AUROC, mean | Median latency | p95 latency |
|---|---|---:|---:|---:|---:|
| Myriad 2 stick | FP16 | **99.153%** | **98.636%** | **546.06 ms** | 546.90 ms |
| Mac CPU, 4 threads | FP32 | 99.233% | 98.638% | 36.29 ms | 40.38 ms |
| Mac GPU, Metal | FP32 | 99.233% | 98.638% | 15.20 ms | 22.42 ms |

AUROC measures ranking quality, not the percentage of correctly classified
images at a chosen threshold. No threshold was tuned on the test set. Scores
are distances, not probabilities. The Mac is faster: approximately 15× on CPU
and 36× on Metal for these reference implementations. Neither Core ML nor the
Mac Neural Engine was benchmarked; these are ordinary PyTorch baselines.

| One-shot reference index | Stick image AUROC | CPU / GPU image AUROC |
|---|---:|---:|
| 0 | 99.762% | 99.762% |
| 1 | 99.762% | 99.921% |
| 2 | 97.937% | 98.016% |

Reference choice matters more here than the mean FP16 ranking difference.
The stick's image AUROC is 0.079 percentage points below the Mac's mean.
Its image-score mean absolute difference from CPU is 0.00642, maximum 0.02407.
Patch-distance mean absolute difference is 0.00379, maximum **0.17679**: similar
aggregate localization quality does not mean every patch matches. Even a
self-reference can produce a small positive score on FP16 (0.01074 in the live
verification), rather than exact zero. CPU and Metal image scores differ by at
most 0.0000093 in these trials.

Raw per-image IDs, labels, scores and timings are retained in
[stick](results/dino_bottle_myriad.json), [CPU](results/dino_bottle_cpu.json) and
[Metal](results/dino_bottle_mps.json) result files. The
[assessment](results/dino_assessment.json) includes numerical drift, provenance,
file hashes and device execution evidence. Detailed patch maps are generated
under ignored `assets/dino-benchmark/` for the assessment script.

## What runs where

| Phase | Stick mode |
|---|---|
| Image decode, square crop, bicubic resize, RGB normalization | Mac preprocessing |
| Patch projection and positional embeddings | Stick |
| All 12 Transformer blocks: attention, LayerNorm, GELU, MLP, LayerScale | Stick |
| Final LayerNorm and L2 feature normalization | Stick |
| All query/reference patch similarities and nearest distances | Stick |
| Mean of the two largest patch distances, the image score | Stick |
| USB transport, metrics, heatmap enlargement and display | Mac |

There is no CPU/GPU model fallback inside stick inference. The host receives
only **256 patch distances plus one score** per inference. A 256 × 384 reference
bank is supplied as a graph input; its upload and image upload are included in
timing. No intermediate Transformer activations are processed on the host.

Calibration uses a separate, complete encoder graph **on the stick** to obtain
the reference bank. The host copies its output into the anomaly graph's reference
input. It neither computes nor normalizes those features. Only one device graph
is open at a time; calibration swaps encoder/anomaly graphs, then inference keeps
the anomaly graph resident. CPU and GPU comparisons calibrate their own banks
with the same image and their own precision.

The anomaly IR contains 864 layers and 43,257,668 bytes of FP16 weights/constants.
Its final-request [device profile](results/dino_bottle_myriad.tsv) contains
**567 executed stages** and reports zero profile `cpu_us`, including distance
and score-reduction stages. This is graph-execution evidence; it does not imply
that capture, USB transport or display use no CPU.

The native ARM64 OpenVINO 2020.3 runtime and stock Myriad 2 RAM firmware are
reused from Crivo. No firmware was rewritten or flashed. The Myriad library
still has historical `crivo_*` transport symbol names, but accepts general
tensor graphs through `Myriad.infer_tensors()`.

## Protocol and comparisons

- Model: pinned [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small),
  22,056,576 checkpoint parameters; the non-register DINOv2-S/14 student, introduced
  in 2023. Training and the larger distillation teacher are not run on the stick.
- Method: the patch-based, one-shot path of
  [AnomalyDINO](https://github.com/dammsi/AnomalyDINO), preprint 2024 / WACV 2025.
  We use no foreground masking and no reference rotations. For bottle, these
  choices also match the author's `informed` preprocessing configuration.
- Dataset: [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad),
  CVPR 2019; all 83 bottle test images, comprising 20 normal and 63 anomalous.
  References come only from the normal training split. The three reference
  indices follow the author's deterministic sorted-file selection.
- Inputs: square 224 × 224, 16 × 16 patches, plus one class token in the encoder.
  Original non-register position interpolation uses bicubic scale `16.1 / 37`,
  `align_corners=False`, no antialiasing. Positional resizing is precomputed at
  export, with no per-image model calculation on the host.
- Distance: squared L2 between L2-normalized patch features, one nearest neighbor.
  The image score is the mean of `floor(256 × .01) = 2` largest distances. A
  reduction tournament computes this exactly, including ties, on the stick.
- Pixel evaluation: raw 16 × 16 distances are bilinearly enlarged to each
  original image's dimensions, smoothed with SciPy Gaussian `sigma=4`, then
  evaluated against original pixel masks. This common **host metric
  postprocessing** uses PyTorch bilinear interpolation rather than OpenCV.
  Camera display enlargement is visualization, not an extra model or classifier.
- Timing: preprocessed tensor to completed host map/score, three warmups per
  reference, one request per test image. Loading, calibration, image decode,
  normalization and pixel metric postprocessing are excluded. Requests and
  backends run sequentially. Other laptop activity is not controlled.

The author's default resolution is **448**, and published headline scores average
multiple categories/settings. Our 224-square bottle result must not be compared
directly with those headline numbers. Meaningful comparisons need the same
resolution, references, categories, preprocessing and scoring. The CPU/Metal
results above provide that matched baseline for this port.

The explicit FP32 reference was checked on three public test images against the
**original Meta model code**, loaded strictly with the same mapped weights:
normalized patch features match exactly. Brute-force squared distances and the
author's original `mean_top1p` function agree with the reference within 0.000002.
See [reference verification](results/dino_reference_verification.json).

## Reproduce

From this repository on the prepared Mac, with the stick connected:

```sh
bash scripts/setup_dino.sh
.venv/bin/python -m scripts.benchmark_dino --device myriad
.venv/bin/python -m scripts.benchmark_dino --device cpu
.venv/bin/python -m scripts.benchmark_dino --device mps
.venv/bin/python -m scripts.assess_dino
```

The complete setup script succeeded on this Mac. Re-exported anomaly XML/BIN
hashes exactly match the benchmark artifacts. A complete 77-option Crivo request
also still returned `card_arrival` on the physical stick at 661 ms after the
shared transport changes.

The setup builds/reuses the common native runtime, fetches pinned assets,
exports the encoder and anomaly graphs, and checks the reference. Model weights
are verified against fixed SHA256 hashes. Original-format PNGs are obtained
from the pinned public `foersben/mvtec-ad` mirror because historical direct MVTec
download links were unavailable. The mirror revision and all used image/mask
SHA256 hashes are retained in the public assessment. This records the evaluated
files; it is not independent verification against an official archive hash.

Only one program should own the stick. Stop Crivo or the camera server with
Ctrl+C before benchmarking, and stop benchmarking before starting a demo.

## Camera and image demo

```sh
.venv/bin/python -m scripts.serve_dino
```

Open <http://127.0.0.1:8766/> in **Chrome**. The local interface also supports
uploaded JPEG/PNG images and three public bottle examples.

1. Select **Cassée**, then **Comparer les trois moteurs** for a repeatable demo.
2. Select **Activer la caméra** and allow camera access for this local site.
   No microphone is requested. The camera starts only after that action.
3. Put a fixed object on a stable background. Select **Utiliser l’image actuelle
   comme référence normale**. This recalibrates each backend, including the
   stick's complete encoder; it does not train any weights.
4. Change the object's appearance, then compare all three backends or select
   **Live sur le stick**. The live mode processes one frame at a time, without a
   frame queue. CPU/GPU model inference does not run in stick-only live mode.
5. Stop live mode, turn the camera off, and optionally restore the MVTec reference.

The physical FaceTime camera was tested in Chrome: capture, reference encoding,
multiple completed stick-only live frames and camera shutdown all worked.
Observed device inference was about 545 ms/frame. The reciprocal inference
ceiling is about 1.8 frames/s; this is not a measured end-to-end browser frame
rate. See [camera verification](results/dino_camera_verification.json).
Camera access in the in-app browser was not validated; use Chrome for this demo.

After the Mac was locked and unlocked, an inference returned `Failed to queue inference:
NC_ERROR`. Selecting **Reprendre la référence MVTec** rebuilt the device graphs
and restored inference without unplugging the stick. Camera reopening and live
inference were then verified again at about 545 ms/image. For a camera-specific
reference, recapture the normal image after this reset.

The camera demo is a functional demonstration, **not a camera accuracy
benchmark**. Pose, illumination, shadows and background changes can raise the
score. There is no calibrated normal/defect threshold. Heatmaps use the same
fixed display scale across backends, rather than stretching each image's
minimum/maximum into an apparently strong anomaly.

The server binds only to loopback, rejects foreign Host/Origin headers, bounds
image requests, and serializes model work on one worker. It keeps the current
reference and job in memory; it does not log bodies, save camera images or use
browser storage. The JSON export includes maps, scores and latencies, without
source images. Reference and result images disappear when replaced or the demo
closes. MVTec example files are the downloaded dataset, not recorded camera data.

With the server running:

```sh
.venv/bin/python -m scripts.verify_dino_live
```

This tests real predictions, reference replacement/restoration without USB
replugging, the PNG transport used by the camera, score aggregation, input bounds
and local-origin restrictions. It uses public data; it does not activate a camera.
See [live verification](results/dino_live_verification.json).

## Code map

| File | Role |
|---|---|
| `movidius_lab/dino.py` | Preprocessing, explicit FP32 reference and dataset inputs |
| `scripts/export_dino.py` | Complete fixed-shape FP16 encoder and anomaly graph |
| `movidius_lab/legacy_ir.py` | Shared IR writer, with image patch convolution |
| `movidius_lab/myriad.py` | MYRIAD-only graph transport, reused by Crivo and DINO |
| `scripts/benchmark_dino.py` | Complete bottle trials and image/pixel metrics |
| `scripts/assess_dino.py` | Aggregate metrics, drift, profile and provenance |
| `movidius_lab/dino_live.py` | Reference lifecycle and resident comparisons |
| `scripts/serve_dino.py`, `web/dino/` | Local camera/image interface |
| `scripts/verify_dino_reference.py`, `scripts/verify_dino_live.py` | Reference and hardware integration checks |

## Historical interpretation

The demonstrable result is that 2017 inference hardware can execute a complete
2023 self-supervised visual Transformer and a 2024–2025 anomaly method at useful
interactive speed, with near-identical bottle ranking quality to this Mac
reference. It does not show that the weights, training data or training recipe
were available in 2017, or that the pipeline outperforms current Mac hardware.
The actual host runtime is from 2020. We make no energy-efficiency claim.

Next useful steps are more MVTec categories, more reference choices, comparison
with the author's 448-resolution setting, and reducing device/runtime overhead.
They are separate experiments, not implied by this pilot result.
