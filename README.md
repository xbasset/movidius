# Movidius time machine

Can a $79 USB accelerator launched in July 2017 execute a language model
published in 2019?

**First result:** pretrained GPT-2 small generates text with its first attention
projection computed on an original Intel Movidius Neural Compute Stick. The Mac
runs the remaining transformer operations. The stick also independently executes
GPT-2's full 768×50,257 vocabulary projection, using about 74 MiB of FP16 weights.

Read [REPORT.md](REPORT.md) for measured timings, numerical checks, the failed
broader offload experiment, and the boundaries of the historical claim. Raw
measurements are under [results/](results/).

## Run the working demo

On a fresh clone, first prepare the dependencies, weights and native runtime
using the [setup instructions](#set-up-from-a-clean-checkout) below. Plug in the
original Myriad 2 stick, then run from the repository directory:

```sh
.venv/bin/python -m scripts.run_gpt2 --hybrid --tokens 96 --runs 1 \
  --temperature 0.8 --prompt 'In 2017, a scientist built a tiny computer that' \
  --output results/gpt2_sample.json
```

The command boots stock RAM firmware, loads the real GPT-2 QKV weights, generates
text, records token IDs/timings, and resets the stick on close. It needs direct USB
access. In a Codex sandbox, execute the device command with the host's normal USB
permissions; a sandboxed process may not see the device. No `sudo` is needed on
the tested laptop. Run only one device experiment at a time.

For CPU-only inference, omit `--hybrid`. Generation uses four CPU threads. It is
greedy by default; positive temperature enables seeded top-40 sampling. GPT-2's
1,024-token context is enforced rather than silently sliding cached positions.
Generated text can repeat and invent facts; it is a base language model.

## Reproduce the assessment

Run sequentially so model jobs do not compete for CPU resources:

```sh
# Independent reference: full and cached logits against Transformers.
.venv/bin/python -m scripts.verify_reference

# Stock Inception-v1 smoke test on the physical stick.
.venv/bin/python -m scripts.probe_device --inception

# Actual GPT-2 weights on device, compared with FP32 accumulation of FP16 values.
.venv/bin/python -m scripts.test_dense_device --gpt2 --runs 10 \
  --output results/gpt2_qkv_device.json
.venv/bin/python -m scripts.test_dense_device --gpt2 --matrix lm-head --runs 3 \
  --output results/gpt2_lm_head_device.json

# Same prompt, greedy decoding, 64 generated tokens, three separate timed runs.
.venv/bin/python -m scripts.run_gpt2 --tokens 64 --runs 3 \
  --prompt 'The scientist opened the box and found' --output results/gpt2_host.json
.venv/bin/python -m scripts.run_gpt2 --hybrid --tokens 64 --runs 3 \
  --prompt 'The scientist opened the box and found' --output results/gpt2_hybrid.json

# Fixed-token likelihood/logit parity, using a 256-token WikiText-2 slice.
.venv/bin/python -m scripts.assess_parity --tokens 256

# Run the sampled demo above, then rebuild the report from the JSON evidence.
.venv/bin/python -m scripts.summarize
```

Device matrix timings exclude graph upload and boot. The hybrid generation
benchmark includes CPU work, USB transfer and sampling; its decode rate excludes
initialization and prefill. The report distinguishes these timing boundaries.
The parity slice is a smoke test, not a full WikiText benchmark or leaderboard
submission. Absolute logit error, distribution KL, top-1 agreement and likelihood
change are all retained. Energy has not been measured.

## Set up from a clean checkout

Tested: Apple M2 Pro, macOS 26.6.2, Python 3.12, Xcode command-line tools.
The bootstrap also expects `uv`, Git, curl, autoconf, automake, GNU libtool and make.
These tools were already present on this laptop. All project dependencies and
build outputs stay in this directory; no global Python/SDK install is performed.

```sh
git clone https://github.com/xbasset/movidius.git
cd movidius
bash scripts/setup.sh
```

This installs the recorded Python dependency versions, checks out pinned NCSDK and
libusb sources, builds a native ARM64 library, and downloads pinned public GPT-2
weights plus evaluation text. Allow roughly 2 GiB of disk space. The model's
SHA256 is checked, and the WikiText file's SHA256 is recorded with the parity
results. The stock firmware and sample graph come from the pinned macOS fork.

The old graph compiler package is not needed to run the custom graph writer.
Its archive location and hash are documented in [THIRD_PARTY.md](THIRD_PARTY.md).
It was inspected to understand the binary format, without installing TensorFlow
1.x, Caffe or a Linux VM.

## Code map

| File | Responsibility |
|---|---|
| [movidius_lab/device.py](movidius_lab/device.py) | Typed bindings, device lifetime, graph execution |
| [movidius_lab/graph.py](movidius_lab/graph.py) | Small dense-matrix graph serializer |
| [movidius_lab/gpt2.py](movidius_lab/gpt2.py) | Minimal GPT-2 decoder, KV cache and device hooks |
| [scripts/run_gpt2.py](scripts/run_gpt2.py) | Generation and timing |
| [scripts/verify_reference.py](scripts/verify_reference.py) | Independent implementation check |
| [scripts/assess_parity.py](scripts/assess_parity.py) | Fixed-token host/device comparison |
| [scripts/summarize.py](scripts/summarize.py) | Report generation from measured artifacts |

`third_party/`, `.venv/`, `.cache/`, `.local/`, `build/` and downloaded `assets/`
are ignored. Keep the small source files, documentation, dependency locks and
`results/` when sharing the experiment. Dependencies retain their own licenses.

## Next iteration

1. Resolve graph replacement while a device is open. The current single-QKV path
   keeps one graph resident and works reliably in these tests.
2. Offload a complete decoder block, then the full model, with device-resident
   weights and KV state. Preserve reference probabilities at each step.
3. Measure longer contexts and full held-out benchmarks. Add a USB power meter
   for device watts/joules per token; record the host's contribution separately.

An experimental `--offload all-dense` path is retained to reproduce the failure:

```sh
.venv/bin/python -m scripts.run_gpt2 --offload all-dense --tokens 8 --runs 1 \
  --prompt 'Once upon a time' --output results/experimental_all_dense.json
```

It currently fails on the third projection upload. Pre-reserving larger buffers
did not fix it. [Saved transcript](results/failed_experiments/stream_all_dense.log).
This path is not a completed whole-model demonstration.

## Historical reference points

- [Intel's July 20, 2017 launch entry](https://newsroom.intel.com/new-technologies/intel-newsroom-archive-2017)
- [Original stick architecture and RAM boot behavior](https://movidius.github.io/ncsdk/ncs.html)
- [GPT-2's February 2019 announcement](https://openai.com/index/better-language-models/)
- [nanoGPT architecture/training reference](https://github.com/karpathy/nanoGPT)
- [llama2.c tiny-model comparison project](https://github.com/karpathy/llama2.c)

The defensible historical claim currently concerns hardware execution capability.
It does not establish the cost of discovering or training the model in 2017.
