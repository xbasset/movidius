# DINOv2-S / AnomalyDINO at 448 × 448

This extends the [224-square bottle pilot](REPORT_224.md) to every MVTec AD
category. The measured results are in the [448 report](REPORT_448.md).
The model, training and pretrained weights are unchanged. No fine-tuning uses
MVTec AD test images.

## Comparison target

Use the **AnomalyDINO-S (448), one-shot, agnostic** row of
[Table 2 in Damm et al.](https://arxiv.org/html/2405.14529v3#S4.T2), WACV 2025:

| Image AUROC | Image F1-max | Image AP | Pixel AUROC | Pixel F1-max | AUPRO, FPR ≤ 0.3 |
|---:|---:|---:|---:|---:|---:|
| 96.5 ± 0.4% | 96.0 ± 0.2% | 98.1 ± 0.3% | 96.3 ± 0.1% | 57.9 ± 0.8% | 91.7 ± 0.1% |

The abstract's 96.6% result uses 672 resolution. It is a different target.
Reference choices follow the paper's three deterministic trials, rather than
three randomly selected training images.

## Inputs and scoring

- All **1,725 test images**, normal and anomalous, from all 15 categories.
- For each category, three separate one-shot trials use the first, second and
  third sorted normal training image (`000.png`, `001.png`, `002.png`).
- Each reference is augmented by eight rotations: 0, 45, …, 315 degrees. Match the
  author's OpenCV linear warp with `BORDER_DEFAULT`, then the bicubic PIL resize
  and ImageNet normalization. Each bank has 8,192 patch vectors.
- All dataset images are square. The shared camera preprocessor's square crop
  therefore leaves their aspect ratio unchanged. The model sees 448 × 448 pixels,
  32 × 32 patch tokens and one CLS token.
- Use the non-register DINOv2-S/14 backbone and the original Meta positional
  interpolation offset of 0.1. Extract the last normalized patch features.
- Query foreground masking applies to **capsule, hazelnut, pill, screw and
  toothbrush**. References remain unmasked. PCA centers the raw final-LayerNorm
  features; threshold the oriented first component at 10 and adapt its sign
  using the central `[6:25, 6:25]` region and 35% criterion.
- Preserve the pinned author's morphology on the **flattened N × 1 mask**:
  dilation then closing with a 3 × 3 kernel. Reshaping before morphology would
  implement a different algorithm.
- L2-normalize patch features. Each query patch's distance is **1 − cosine** to
  its nearest reference patch. Fill masked-out patch distances with zero.
- The image score is the mean of the ten largest distances, including the zero
  background entries. Scores are anomaly distances, not probabilities.
- For pixel evaluation only, bilinear-upsample the 32 × 32 map to the original
  image dimensions, then Gaussian-smooth with sigma 4. Use original GT masks.

The source implementation is pinned at
[`b9d1c26`](https://github.com/dammsi/AnomalyDINO/tree/b9d1c2648e3a5247437d4d953d907a8f3d994457).
See its `backbones.py`, `detection.py`, `utils.py` and `post_eval.py`.

## Execution boundaries and differences

On the stick, patch projection, all 12 Transformer blocks, normalization, query
PCA, mask morphology, nearest-neighbor matching and top-ten aggregation execute
inside one **MYRIAD-only graph**. There is no host model fallback. The host
decodes/rotates/resizes images, normalizes RGB inputs, copies tensors and displays
or evaluates returned maps. Reference feature extraction also runs on the stick;
the host only concatenates the eight feature outputs into each bank.

For quality evaluation, one encoder pass feeds **three isolated heads**. Each
head sees one reference and its eight rotations. Banks are never united across
trials, and this does not turn the experiment into three-shot detection.
The three-head timing is throughput for this evaluation arrangement, not
single-reference latency. A separate single-head graph supplies that comparison.

The stock VPU runtime cannot execute sklearn's randomized SVD. The stick uses
**64 normalized power iterations in FP16** for the same first principal
component, with deterministic initialization, largest-component sign convention,
and an intermediate scale of 1/128 to avoid overflow. This is an approximation
to the original solver. Isolated checks measure mask disagreements. The full
benchmark compares nonzero map support, which combines solver differences,
FP16 backbone drift and zero rounding, as well as scores and metrics. It does
not claim bit-for-bit algorithm identity.
The CPU and Metal quality baselines use the original sklearn solver with
`random_state=0`. As in the author's implementation, their PCA is on the CPU;
the Metal model and nearest-neighbor search execute on the GPU.

The FP32 references use explicit PyTorch cosine matching in place of Faiss's
normalized squared-L2 search divided by two. These are mathematically equivalent
for unit vectors. Floating-point arithmetic, library versions, fixed PCA seed,
and float32 pixel-map postprocessing can produce small differences.

At 1,025 tokens, our legacy OpenVINO 2020.3 / MA2450 path corrupts tail rows of
constant-vector broadcasts. A standalone LayerNorm reproducer isolates this from the
reductions and from USB transfer. Expanding constant gamma/bias/LayerScale
vectors to their full tensor shape fixes it without changing their values.
This increases the IR weights from about 43 MB to **187 MB**. The learned model
still has 22.1 million parameters. Dynamic PCA mean broadcasts are materialized
with a matrix product. No firmware is changed or flashed.

## Metrics and latency

Compute each of the six metrics independently for each category/reference.
Macro-average the 15 categories within each trial, then report the mean and
population standard deviation of those three macro scores. Do not pool pixels
or images across categories to produce a replacement leaderboard metric.
F1-max is an evaluation statistic chosen over all thresholds, not a deployable
threshold selected without labels.

AUPRO weights every connected GT region equally, uses 8-connectivity, and
integrates to FPR 0.3. The evaluator merges equal score thresholds before
integration. It sorts packed score/index pairs in place and streams curve
chunks to bound memory. Synthetic checks compare pixel AUROC and F1 with
sklearn, and AUPRO with direct region-overlap thresholds, including ties crossing
a chunk boundary. It uses all original-resolution pixels, with no histogram
approximation or mask downsampling.

The controlled latency run uses ten evenly spaced sorted test IDs from each
category: **150 batch-one requests per backend**, with the first reference and
three warmups per category. Time prepared host tensor → returned raw patch map
and score. Include transfers, FP16 input conversion, synchronization and query
masking; exclude image decode/resize, graph load, reference calibration and pixel
metric postprocessing. Run the three devices sequentially after quality/metric
jobs finish. CPU uses four PyTorch threads. Metal uses FP32; Core ML and the
Apple Neural Engine are not tested. The fixed stick graph calculates PCA even
when its mask is disabled, which is included in its latency.

## Reproduce

From the existing native macOS ARM64 project setup:

```sh
bash experiments/exp03_anomalydino/setup_448.sh

# Produce quality predictions. Run one physical-stick job at a time.
caffeinate -i .venv/bin/python -u -m experiments.exp03_anomalydino.scripts.benchmark_mvtec --device cpu
caffeinate -i .venv/bin/python -u -m experiments.exp03_anomalydino.scripts.benchmark_mvtec --device mps
caffeinate -i .venv/bin/python -u -m experiments.exp03_anomalydino.scripts.benchmark_mvtec --device myriad

# Evaluate every original-resolution pixel, sequentially to bound memory.
caffeinate -i .venv/bin/python -u -m experiments.exp03_anomalydino.scripts.evaluate_mvtec --device cpu
caffeinate -i .venv/bin/python -u -m experiments.exp03_anomalydino.scripts.evaluate_mvtec --device mps
caffeinate -i .venv/bin/python -u -m experiments.exp03_anomalydino.scripts.evaluate_mvtec --device myriad

# Stop other model/metric jobs before timing.
caffeinate -i .venv/bin/python -u -m experiments.exp03_anomalydino.scripts.latency_mvtec --device cpu
caffeinate -i .venv/bin/python -u -m experiments.exp03_anomalydino.scripts.latency_mvtec --device mps
caffeinate -i .venv/bin/python -u -m experiments.exp03_anomalydino.scripts.latency_mvtec --device myriad

.venv/bin/python -m experiments.exp03_anomalydino.scripts.verify_dino448
.venv/bin/python -m experiments.exp03_anomalydino.scripts.verify_vector_broadcast
.venv/bin/python -m experiments.exp03_anomalydino.scripts.assess_mvtec
```

Allow about two hours for a complete stick quality run plus calibration and
latency checks; pixel metrics also take substantial time. `caffeinate -i`
prevents idle sleep while its command runs. It does not change screen locking.
The selected dataset download is about 1.6 GiB: test images, GT masks and only
three normal references per category. Setup also needs space for the model,
runtime, generated graphs and reference caches. No images or generated graph
binaries are committed.

Predictions checkpoint per completed category under `assets/mvtec448/` and the
benchmark resumes those categories. The dataset fetcher verifies every selected
file against the pinned mirror. Reference caches additionally check encoder and
source-image hashes; metric caches check the prediction archive hash. **Delete
the relevant prediction/metric cache after changing code, model or protocol**
to request a fresh run. Partial categories rerun after interruption.
The evaluator can follow an ongoing producer with `--wait-for-cache`; avoid
running two full pixel evaluators together on a 16 GiB Mac.

Results are local measurements on one stick and one Apple M2 Pro. Energy is
excluded. The 2017 hardware runs a model and method developed later; this does
not establish that their training recipe, data or weights existed in 2017.
The host runtime is from 2020. The camera demo remains the separately documented
224-square pilot.
