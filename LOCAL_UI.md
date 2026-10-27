# Local Crivo workbench

Start the resident, localhost-only interface after the existing Crivo setup:

```sh
bash scripts/setup_crivo.sh  # once, if not already prepared
.venv/bin/python -m scripts.serve_crivo
```

Open <http://127.0.0.1:8765>. The server binds only to `127.0.0.1`. Use `--port`
to select another local port. Stop with Ctrl+C before running any other stick
program: the interface keeps the Myriad graph open and its weights resident.
No administrator helper, web framework, CDN, external font or cloud inference
service is used. The existing Python dependencies are sufficient.

## Decisions and latency

Enter a short English banking message, select an example, or press Cmd/Ctrl+Enter.
The same Crivo-17M checkpoint is evaluated sequentially on the stick, Mac CPU
and Metal GPU. All 77 BANKING77 options are retained. This is a closed set of
banking intents, including for messages outside that domain.

- Each latency is the median of three completed, synchronous requests with
  resident models. Loading and tokenization are excluded; tokenization time is
  displayed separately.
- CPU uses four threads and FP32; Metal uses FP32; Myriad uses the complete FP16
  graph. There is no host-model fallback for the stick.
- The Mac evaluates the actual input length. The stick pads to its fixed 960
  tokens, with padding masked. Overlong inputs fail without truncation.
- The probability table can show the ten leading intents or all 77. Raw FP16
  probabilities are displayed without host renormalization.
- Editing the message clears its previous results. Only one measurement can
  run at a time. Unavailable backends are identified instead of substituted.

Messages are not written to logs, files or browser storage. The server holds
only the current job result in memory. The JSON export downloads the current
probabilities, timings and energy metrics; it does not include the message text.

## Mac energy measurement

“Mesurer l’énergie” takes roughly 15 seconds. It reads the system library
IOReport's **Energy Model** counters without sudo. These are model-based CPU,
GPU and ANE energy estimates, not readings from an electrical power meter.
The API is private and may change with macOS; missing counters are reported
as unavailable. Tested here on Apple M2 Pro / macOS 26.6.2.

The channels expose different units on this Mac:

| Channel | Native unit | Conversion to joules |
|---|---|---|
| CPU Energy | mJ | × 0.001 |
| GPU Energy | nJ | × 0.000000001 |
| ANE0 | mJ | × 0.001 |

For each backend, the app first allows a short settling interval, then samples
one second at idle and a serial inference loop lasting at least three seconds.
The input shape is warmed before measurement. Each loop remains batch size one.

`average watts = counter delta in joules / sampling seconds`

`gross joules per decision = counter delta in joules / completed decisions`

`incremental joules per decision = (joules − idle watts × sampling seconds) / completed decisions`

The large figures are **gross energy, idle included**. The smaller detail shows
the estimated increment above the immediately preceding idle window. Negative
increments are retained in exported data and shown as “below background noise”,
not clamped into a fake zero-cost result. Other applications, browser rendering,
DVFS and thermal state affect these machine-wide component counters. Short
windows are useful for exploration, not calibrated cross-device efficiency claims.

The UI separates these three modes:

1. Mac components while the CPU runs Crivo.
2. Mac components while the GPU runs Crivo.
3. Mac components **while accompanying stick inference**. This excludes the
   stick's USB electrical consumption and must never be called the stick's energy.

The sum is not the Mac's total electrical draw, wall power or battery energy.
The UI deliberately leaves stick electrical energy unavailable: this stick has
no voltage/current telemetry in its inference API. An external inline USB meter
is needed to measure it. A USB descriptor's negotiated current is not a reading.

Numerical examples from one repeated public BANKING77 test input are saved in
[results/crivo_mac_energy.json](results/crivo_mac_energy.json), including sampling
durations, completed counts, idle windows, native channel metadata, joules and
watts. They are a short software-estimate session, not a full-dataset energy
benchmark or a measured CPU-versus-stick electrical-efficiency comparison.

The read-only IOReport binding follows the public API usage documented in
[macmon's source](https://github.com/vladkens/macmon/blob/main/src/sources.rs)
(MIT). No macmon binary, Rust runtime or privileged service is installed.

## Validation and code

With the server running and the stick connected:

```sh
.venv/bin/python -m scripts.verify_crivo_live
```

This exercises real stick/CPU/GPU predictions, the complete 77-option outputs,
energy-counter arithmetic and per-backend idle windows, input bounds, serialized
jobs, and rejection of foreign Host/Origin headers. It saves numerical evidence
without message text. Browser QA also covers example selection, probability
display, energy results and clearing stale results when the input changes.

| File | Responsibility |
|---|---|
| `web/` | Buildless French interface, probability comparison and local JSON export |
| `scripts/serve_crivo.py` | Loopback HTTP API, bounded jobs and one model-owning worker |
| `movidius_lab/live.py` | Resident models, sequential latency and energy trials |
| `movidius_lab/energy.py` | Unit-aware, read-only IOReport snapshots and counter deltas |
| `scripts/verify_crivo_live.py` | Integration checks against the running hardware-backed server |

The server rejects foreign origins, unexpected Host headers, non-JSON POSTs and
oversized requests. It does not enable CORS or log request bodies. Models and
IOReport resources close on the owning worker when the server stops.
