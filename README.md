# Movidius time machine

Three experiments, in the order they were built, on an original **Intel Neural
Compute Stick / Myriad 2 (2017)** and an **Apple M2 Pro Mac**. This repository
contains the implementations, reproduction commands and measured evidence.

| Step | Experiment | Demonstrated result | Start here |
|---|---|---|---|
| 01 | GPT-2: partial language-model offload | Real GPT-2 QKV computation on the stick during text generation; the Mac runs the rest. A separate test executes the full vocabulary projection. | [Experiment 01](experiments/exp01_gpt2/) |
| 02 | Crivo-17M: complete text decisions | All neural inference on the stick; **79.64% macro-F1** on all 3,080 BANKING77 cases, versus 79.59% on the Mac. Local text interface. | [Experiment 02](experiments/exp02_crivo/) |
| 03 | DINOv2-S + AnomalyDINO: visual anomalies | Complete device inference, first a **224 × 224 bottle pilot and camera demo**, then **448 × 448 / all 15 MVTec AD categories**: **96.54% image AUROC**, versus 96.52% on the Mac and 96.5% in the paper. | [Experiment 03](experiments/exp03_anomalydino/) |

The Mac is faster in these implementations. The experiments establish execution
and prediction quality on older hardware. They do not establish an energy
advantage, or that the later models' weights, data and training recipes were
available in 2017. No replacement firmware was written or flashed.

## Read and verify the evidence

Follow **01 → 02 → 03**, then **224 → 448** inside the third experiment. Each
folder contains its code, commands, reports and original numerical results.

- [Evidence index](docs/EVIDENCE.md): claims, limitations, source commits and direct measurement links.
- [Artifact manifest](docs/evidence.json): original/new paths, SHA256 hashes and source commits for all 48 published measurement files. Their contents are unchanged by the reorganization.
- [Setup and shared runtime](docs/SETUP.md): clean checkout, prerequisites and hardware boundaries.
- [Sources and licenses](THIRD_PARTY.md): model, dataset, runtime and firmware provenance.

Check the published evidence and repository links without downloading models or
connecting a stick:

```sh
python3 -m scripts.check_repository
```

## Run a demo

Run commands **from the repository root**. First follow the setup in the relevant
experiment. Only one process can own the USB stick; stop a demo before starting
another demo or benchmark.

```sh
# 01 — text generation, with the first QKV projection on the stick.
.venv/bin/python -m experiments.exp01_gpt2.scripts.run_gpt2 --hybrid \
  --tokens 32 --prompt 'In 2017, a scientist built a tiny computer that' \
  --output output/gpt2_demo.json

# 02 — decisions and probabilities, compared with CPU and Metal.
.venv/bin/python -m experiments.exp02_crivo.scripts.serve_crivo
# Open http://127.0.0.1:8765/ ; stop with Ctrl+C before step 03.

# 03 — image/camera anomaly maps at 224 × 224.
.venv/bin/python -m experiments.exp03_anomalydino.scripts.serve_dino
# Open http://127.0.0.1:8766/ in Chrome.
```

The camera demo remains at 224 × 224. The 448 × 448 experiment is the separately
documented full benchmark, including device PCA masking and all six paper metrics.

## Repository map

```text
experiments/
  exp01_gpt2/          model, scripts, setup, report, results
  exp02_crivo/         model, scripts, setup, reports, web demo, results
  exp03_anomalydino/   model, scripts, 224/448 setup and reports, web demo
    results/224/      initial bottle benchmark and camera verification
    results/448/      15-category benchmark and paper comparison
movidius_lab/         shared USB bindings, graph writers and local HTTP worker
scripts/              shared runtime builds and evidence verification
docs/                 setup and evidence index
```

Weights, datasets, firmware, build outputs and caches remain in the shared,
ignored `assets/`, `third_party/`, `build/`, `.local/` and `.cache/` directories.
Private captures belong in ignored `results/private/` or `results/raw/`;
throwaway reruns can use ignored `output/`. Published results are snapshots:
benchmark defaults regenerate them, so use a separate checkout or an explicit
`--output` where available when preserving the recorded runs.
