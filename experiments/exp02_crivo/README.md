# 02 — Crivo-17M: complete text decisions

After [01 — partial GPT-2 offload](../exp01_gpt2/), this experiment moves a
complete modern decision model onto the stick: embeddings, seven encoder
layers, decision head and softmax. The Mac formats/tokenizes inputs and displays
the probabilities. The graph has no host model fallback.

Read the [report and reproduction protocol](REPORT.md),
[local interface guide](DEMO.md) and [published results](results/).
Continue with [03 — visual anomaly detection](../exp03_anomalydino/).

## Recorded outcome

All **3,080 BANKING77 test messages**, all 77 intents, no truncation:

| Backend | Macro-F1 | Median latency |
|---|---:|---:|
| Stick, FP16 | 79.64% | 661.32 ms |
| Mac CPU, FP32 | 79.59% | 34.11 ms |
| Mac GPU / Metal, FP32 | 79.59% | 9.13 ms |

The stick agrees with CPU on 99.55% of decisions. The slightly higher F1 is
numerical variation, not evidence of a meaningful quality gain. This evaluates
one task, not the complete Jev Decision Index. Energy is outside the scope.

## Run

From the repository root, with [shared prerequisites](../../docs/SETUP.md):

```sh
bash experiments/exp02_crivo/setup.sh

# Complete neural inference on the stick, without evaluating a host model.
.venv/bin/python -m experiments.exp02_crivo.scripts.run_crivo --banking --no-reference \
  --text 'Where is my card? It has not arrived yet.' --output output/crivo_demo.json

# Stop the command above before starting the resident local interface.
.venv/bin/python -m experiments.exp02_crivo.scripts.serve_crivo
```

Open <http://127.0.0.1:8765/> to enter a message, inspect all 77 probabilities
and compare stick/CPU/Metal latency. Stop with Ctrl+C before another stick job.
The interface performs three synchronous requests per backend; the benchmark
uses its separately documented one-request protocol.

The [report](REPORT.md#reproduce) lists independent reference checks, lifecycle
checks and complete sequential CPU/GPU/stick benchmark commands. A full stick
benchmark takes roughly 35 minutes on the tested setup. The [demo guide](DEMO.md)
documents local input handling and live verification.

## Code and evidence

| Path | Role |
|---|---|
| [model.py](model.py) | Exact input construction, explicit Mac reference and macro-F1 |
| [live.py](live.py) | Resident models and sequential latency comparisons |
| [scripts/](scripts/) | Fetch/export, demo, benchmark, reference and lifecycle checks |
| [web/](web/) | Buildless local text interface |
| [results/](results/) | Predictions, profiles, parity and interface verification |

Complete inference required the shared OpenVINO 2020.3 runtime and direct IR
writer, replacing the first experiment's dense-only graph path. Stock firmware
is still loaded to RAM. [Original milestone](https://github.com/xbasset/movidius/commit/f686c61cd0857dcd78cfd3751461bc19b8b0af36).
