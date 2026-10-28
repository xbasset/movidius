# 01 — GPT-2: partial offload

The first milestone establishes that the 2017 stick can execute real GPT-2
weights and participate in text generation. Only block 0's QKV projection is
offloaded during generation; the Mac executes the remaining model. A separate
test runs the complete vocabulary projection on the device.

Read the [measured report](REPORT.md) and [original results](results/).
Continue with [02 — complete Crivo inference](../exp02_crivo/).

## Recorded outcome

- GPT-2 QKV: 1.678 ms device kernel, 5.322 ms including USB/API.
- Vocabulary projection: 768 × 50,257, 73.62 MiB FP16 weights; 36.240 ms kernel.
- Generation: 44.89 tokens/s hybrid versus 59.39 tokens/s CPU; all greedy IDs match.
- The 256-token WikiText-2 slice is a parity smoke test, not a full benchmark.
- Streaming all projections failed; the [failure transcript](results/failed_experiments/stream_all_dense.log) is retained. Complete GPT-2 inference on the stick was not demonstrated.

## Reproduce

See [shared prerequisites](../../docs/SETUP.md). Run from the repository root:

```sh
bash experiments/exp01_gpt2/setup.sh

.venv/bin/python -m experiments.exp01_gpt2.scripts.verify_reference
.venv/bin/python -m experiments.exp01_gpt2.scripts.probe_device --inception
.venv/bin/python -m experiments.exp01_gpt2.scripts.test_dense_device --gpt2 --runs 10 \
  --output experiments/exp01_gpt2/results/gpt2_qkv_device.json
.venv/bin/python -m experiments.exp01_gpt2.scripts.test_dense_device --gpt2 --matrix lm-head --runs 3 \
  --output experiments/exp01_gpt2/results/gpt2_lm_head_device.json

.venv/bin/python -m experiments.exp01_gpt2.scripts.run_gpt2 --tokens 64 --runs 3 \
  --prompt 'The scientist opened the box and found' \
  --output experiments/exp01_gpt2/results/gpt2_host.json
.venv/bin/python -m experiments.exp01_gpt2.scripts.run_gpt2 --hybrid --tokens 64 --runs 3 \
  --prompt 'The scientist opened the box and found' \
  --output experiments/exp01_gpt2/results/gpt2_hybrid.json
.venv/bin/python -m experiments.exp01_gpt2.scripts.assess_parity --tokens 256

# Reproduce the recorded seeded sample before rendering its report.
.venv/bin/python -m experiments.exp01_gpt2.scripts.run_gpt2 --hybrid --tokens 96 --runs 1 \
  --temperature 0.8 --prompt 'In 2017, a scientist built a tiny computer that' \
  --output experiments/exp01_gpt2/results/gpt2_sample.json
.venv/bin/python -m experiments.exp01_gpt2.scripts.summarize
```

These commands regenerate published snapshots. For a quick fresh demo, use
`--output output/gpt2_demo.json` and fewer tokens. Generation defaults to greedy
decoding; positive temperature uses seeded top-40 sampling. The 1,024-token
context limit is enforced. Generated text can repeat and invent facts.

The retained failure path is `run_gpt2 --offload all-dense`; it fails on the
third graph upload in the recorded experiment. Do not present that path as
complete inference. [Original milestone](https://github.com/xbasset/movidius/commit/8ed64d4f2080521755b56abdaedcfb901597cc70).

## Code and evidence

| Path | Role |
|---|---|
| [model.py](model.py) | Minimal GPT-2 decoder, KV cache and device hooks |
| [scripts/](scripts/) | Weight fetch, generation, device arithmetic checks, parity and report rendering |
| [results/](results/) | Published timings, token IDs, reference checks and failed experiment |
| [device.py](../../movidius_lab/device.py), [graph.py](../../movidius_lab/graph.py) | Shared NCSDK transport and dense graph writer |

The report distinguishes device-kernel time, USB/API wall time and hybrid
generation. Its hybrid rate includes the Mac's work. Energy was not measured.
