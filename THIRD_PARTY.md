# Sources and provenance

Downloaded dependencies live outside the small project sources and are ignored
by `.gitignore`. Preserve their notices; this project does not relicense Intel's
SDK, vendor firmware, pretrained weights or other dependencies.

| Dependency | Pinned identity | Usage |
|---|---|---|
| [milosgajdos/ncsdk](https://github.com/milosgajdos/ncsdk/tree/macos-V1) | `38d0dfb9659fb0c4a1c5166471a9364d784262d1`, August 17, 2018 | NCSDK v1 C host API, stock RAM firmware, supplied Inception graph/test image |
| [libusb](https://github.com/libusb/libusb/releases/tag/v1.0.29) | `15a7ebb4d426c5ce196684347d2b7cafad862626` | Native USB transport, statically linked into local host library |
| [GPT-2 small](https://huggingface.co/openai-community/gpt2) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | Public pretrained FP32 weights, converted to FP16 for device graphs |
| [PyTorch examples WikiText-2 test](https://github.com/pytorch/examples/blob/main/word_language_model/data/wikitext-2/test.txt) | SHA256 `d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0` | Small numerical parity fixture; original corpus is tokenized WikiText-2 |
| [Crivo-17M](https://huggingface.co/tardellirs/crivo-17m) | `8e42a0dbc309ca3360bfa94ebfd30061d6994ba7` | Complete decision model; declared CC BY-NC 4.0; weights/source/tokenizer hashes checked by `scripts/fetch_crivo.py` |
| [OpenVINO 2020.3.0](https://github.com/openvinotoolkit/openvino/tree/2020.3.0) | `2fe9b1523058e282ad374db7dc1b3538c7d2dd27` | Native ARM64 Myriad 2 runtime and compiler; source modifications recorded in `scripts/patches/openvino-macos-arm64.patch` |
| OpenVINO's pinned nGraph / ADE submodules | `1797d7fb712d2ffa6048ce5f5b3b560b84ab5ae4` / `cbe2db61a659c2cc304c3837406f95c39dfa938e` | Runtime build dependencies, fetched by Git submodule |
| [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data) | Revision `57ec275d8078af65b7731c2a98be812d844a6d6b`; test SHA256 `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`; labels SHA256 `53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63` | All 3,080 public test cases, 77 labels; hashes match the Decision Index's pinned sources |
| [Decision Index](https://github.com/apolinario/decision-index) | Inspected kit revision `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` | Direct BANKING77 input layout and independent metric verification; MIT license; not a runtime dependency |

The exact model SHA256 is checked in `scripts/fetch_gpt2.py` and saved in
`assets/gpt2/manifest.json`, copied into generation result JSON. Python package
versions are captured in `requirements-lock.txt`; direct dependencies are listed
in `requirements.txt`.

Published results omit raw USB registry captures, device serial/session/location
identifiers, automatic host metadata and exact execution timestamps. Dates,
generic hardware/software specifications and model/file hashes are retained.
The failure transcript replaces the original absolute directory with
`<workspace>`. Redaction does not alter numerical measurements.

## Archived compiler-format reference

Intel's historical download URL returned 404 during this experiment. A public
community repository retained the package:

- URL: https://raw.githubusercontent.com/yockgen/movidius/master/ncsdk/NCSDK-1.12.01.01.tar.gz
- SHA256: `df5261ab79697f3c564279010aebbc8fd171f57290a84f636201aee82368a3b2`
- Inspected files: `tk/Models/NetworkStage.py`, `Blob.py`, `MyriadParam.py`,
  `EnumDeclarations.py`, and `tk/Controllers/FileIO.py`.

The new graph writer implements only the dense operation's binary data layout:
a 264-byte header, one 227-byte stage descriptor, 8-byte-aligned payload start,
and 64-byte-aligned FP16 weights/bias. It does not load the old Python compiler,
copy executable kernels, or provide a SHAVE C compiler. NCSDK's graph version
number 2 here is separate from the NCAPI host version, which is v1.

The downloaded 2018 fork includes an Intel Software Tools license and restrictive
source headers. It is not treated as a blanket Apache-licensed dependency.
Consult its LICENSE before redistributing SDK sources or firmware. It remains
under ignored `third_party/` locally, fetched by the setup script.

## Local build adaptations

- Native ARM64 static libusb 1.0.29, rather than the laptop's x86_64 Homebrew
  libusb 1.0.22.
- `ac_cv_func_pipe2=no`: the configure test accepted a weak SDK symbol that was
  unavailable at runtime. The first builds crashed in `usbi_create_event`.
- `clang -include string.h`: supplies a header missing in the old C source.
- Local dynamic library and sibling `mvnc/MvNCAPI.mvcmd`; no `sudo make install`.

## Crivo runtime firmware

The second experiment uses OpenVINO's stock Myriad 2 RAM firmware, separate from
the NCSDK firmware above:

- Upstream archive: <https://download.01.org/opencv/2020/openvinotoolkit/2020.3/inference_engine/firmware_usb-ma2450_1119.zip>.
- Firmware file: `usb-ma2450.mvcmd`.
- SHA256: `6cf079e614d0fa68381ec29d0b30f44766cf3cf62355620a91912128f0bfdaac`.

The upstream CMake build also downloads Myriad X/PCIe firmware dependencies;
neither is used for this Myriad 2 experiment. The setup script checks the actual
Myriad 2 firmware hash. No firmware binary or generated model graph is committed.

The graph writer converts the pinned Crivo parameters to FP16 and adds constant
rotary positions and attention masks. Their generated-file hashes are retained
in the final assessment. Crivo is an independent Jev-style model; this project
does not redistribute Jev weights or use its hosted inference API.

No vendor firmware was rewritten or flashed. The stock image is uploaded to RAM
through the ordinary USB boot path. Its embedded strings contain a March 2017
component build date; this is not proof of the complete image's release date.

## DINOv2 / AnomalyDINO experiment

| Dependency | Pinned identity | Usage |
|---|---|---|
| [DINOv2 Small](https://huggingface.co/facebook/dinov2-small) | `ed25f3a31f01632728cabb09d1542f84ab7b0056`; weights SHA256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` | Apache 2.0 checkpoint, converted to FP16; all 12 student layers |
| [Meta DINOv2 code](https://github.com/facebookresearch/dinov2) | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | Apache 2.0; independent original-model verification and position interpolation behavior |
| [AnomalyDINO](https://github.com/dammsi/AnomalyDINO) | `b9d1c2648e3a5247437d4d953d907a8f3d994457` | Apache 2.0; method, reference selection, aggregation and evaluation conventions |
| [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | Original-format PNGs from [foersben/mvtec-ad](https://huggingface.co/datasets/foersben/mvtec-ad), `c75b39616f84db43677bcc8228caaafaf5096d7f` | CC BY-NC-SA 4.0; 224 bottle pilot and 448 all-15-category experiment, all 1,725 test images/masks and first three normal references per category |

The model and sources remain in ignored `assets/` and `third_party/`; dataset
images, camera captures, generated IRs and firmware are not redistributed in
this repository. Downloaded dataset license/readme files are preserved locally.
Used dataset image hashes and weight hashes are recorded in the public
assessment. Source/model license terms do not replace the dataset's separate
non-commercial terms. See [DINO_REPORT.md](DINO_REPORT.md) for the tested subset,
reference verification, pixel postprocessing and exact runtime boundaries.
The [448 protocol](DINO448_PROTOCOL.md) follows the same pinned AnomalyDINO
source for rotations, query masks and all six evaluation metrics. Its device
PCA uses a documented power-iteration approximation. OpenCV is pinned in the
Python dependency files. The full selected-file hashes are published in
`results/dino448_dataset.json`; generated maps and reference banks remain local.

Scientific references: Oquab et al., *DINOv2: Learning Robust Visual Features
without Supervision* (2023); Damm et al., *AnomalyDINO: Boosting Patch-Based
Few-Shot Anomaly Detection with DINOv2* (WACV 2025); Bergmann et al., *MVTec AD —
A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection* (CVPR 2019).
