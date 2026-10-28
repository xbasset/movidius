# Setup and shared runtime

The three experiments use the original **NCS1 / Myriad 2 (MA2450)**, not NCS2 /
Myriad X. The tested host is an Apple M2 Pro, 16 GiB RAM, macOS 26.6.2,
native ARM64 Python 3.12.

## Prepare a checkout

Prerequisites: Xcode command-line tools, `uv`, Git, curl, CMake, autoconf,
automake, GNU libtool and make. These were already installed on the tested Mac.
Crivo's original setup notes also list wget. Dependencies and generated assets
stay local to the checkout; the scripts do not install a global Python SDK.

```sh
git clone https://github.com/xbasset/movidius.git
cd movidius

# Choose an experiment. Each setup includes its shared runtime prerequisites.
bash experiments/exp01_gpt2/setup.sh
bash experiments/exp02_crivo/setup.sh
bash experiments/exp03_anomalydino/setup_224.sh
# After the pilot, prepare the full benchmark:
bash experiments/exp03_anomalydino/setup_448.sh
```

Run all Python commands as modules from the repository root. The experiment
folders are Python packages; no editable install or `PYTHONPATH` change is needed.
Recorded Python versions are in [requirements-lock.txt](../requirements-lock.txt).
The complete 448 dataset selection alone is about 1.6 GiB; allow additional
space for weights, source trees, runtime builds, generated graphs and caches.

## Shared code

| Component | Used by | Role |
|---|---|---|
| [setup.sh](../scripts/setup.sh), [build_runtime.sh](../scripts/build_runtime.sh) | All experiments | Python environment, pinned NCSDK/libusb sources, native USB runtime |
| [device.py](../movidius_lab/device.py), [graph.py](../movidius_lab/graph.py) | GPT-2 | NCSDK v1 device lifetime and dense graph format |
| [setup_myriad.sh](../scripts/setup_myriad.sh), [build_myriad_bridge.sh](../scripts/build_myriad_bridge.sh) | Crivo and AnomalyDINO | Pinned OpenVINO 2020.3 runtime and native bridge |
| [legacy_ir.py](../movidius_lab/legacy_ir.py) | Crivo and AnomalyDINO | Explicit legacy IR graph serialization |
| [myriad.py](../movidius_lab/myriad.py), [myriad_bridge.cpp](../movidius_lab/myriad_bridge.cpp) | Crivo and AnomalyDINO | MYRIAD-only tensor transport and profiling |
| [workbench.py](../movidius_lab/workbench.py) | Both local interfaces | Loopback HTTP handling and serialized worker lifecycle |
| [paths.py](../movidius_lab/paths.py) | Shared utilities | Repository root and artifact hashing without model imports |

`scripts/setup.sh` now prepares only the common dependencies and NCSDK runtime.
GPT-2 downloads belong to experiment 01's setup. Preparing Crivo or DINO no
longer downloads GPT-2. Existing ignored weights, graphs and caches retain their
paths and can be reused after the reorganization.

The bridge retains its historical `libcrivo_myriad.dylib` filename and C symbols
for compatibility. It contains transport and lifecycle code; model computation
is expressed by each experiment's graph writer.

## Physical device use

Plug in the original stick and run one device process at a time. USB commands
need normal host USB access; a sandbox may hide the device. No `sudo`, Linux VM,
Docker or Rosetta inference is required on the tested Mac. Vendor firmware is
loaded into RAM and the device resets on close. The two runtime generations use
different stock firmware images, whose hashes are in the experiment evidence.

The camera demo is local to `127.0.0.1:8766`, requests no microphone and records
no camera images. See the [224 report](../experiments/exp03_anomalydino/REPORT_224.md)
for camera permissions, calibration and recovery after Mac locking.

Keep model jobs, pixel metric jobs and builds separate when comparing latency.
The original protocols define warmups, thread counts, precision and timing
boundaries. The reorganization does not introduce new benchmark measurements.
