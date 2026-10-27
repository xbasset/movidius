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

No vendor firmware was rewritten or flashed. The stock image is uploaded to RAM
through the ordinary USB boot path. Its embedded strings contain a March 2017
component build date; this is not proof of the complete image's release date.
