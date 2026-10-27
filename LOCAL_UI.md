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
probabilities and timings; it does not include the message text.

## Experiment scope

The experiment evaluates prediction quality and latency. Energy measurement is
excluded: software estimates of Mac components do not cover USB stick power,
and no external electrical instrumentation is available. The exploratory energy
UI, counter binding and numerical artifact have been removed.

## Validation and code

With the server running and the stick connected:

```sh
.venv/bin/python -m scripts.verify_crivo_live
```

This exercises real stick/CPU/GPU predictions, the complete 77-option outputs,
input bounds, serialized jobs, and rejection of foreign Host/Origin headers.
It saves numerical evidence without message text. Browser QA also covers example
selection, probability display and clearing stale results when the input changes.

| File | Responsibility |
|---|---|
| `web/` | Buildless French interface, probability comparison and local JSON export |
| `scripts/serve_crivo.py` | Loopback HTTP API, bounded jobs and one model-owning worker |
| `movidius_lab/live.py` | Resident models and sequential latency trials |
| `scripts/verify_crivo_live.py` | Integration checks against the running hardware-backed server |

The server rejects foreign origins, unexpected Host headers, non-JSON POSTs and
oversized requests. It does not enable CORS or log request bodies. Models close
on the owning worker when the server stops.
