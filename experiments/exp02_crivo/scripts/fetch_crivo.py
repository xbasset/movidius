"""Fetch pinned Crivo sources/weights without executing downloaded Python code."""

import hashlib
import json

import requests

from movidius_lab.paths import ROOT
REPO = "tardellirs/crivo-17m"
REVISION = "8e42a0dbc309ca3360bfa94ebfd30061d6994ba7"
BANKING_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
FILES = ["README.md", "config.json", "model.py", "crivo.py", "decision_index_engine.py",
         "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
BANKING_FILES = {
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
}


def git_blob_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main():
    target = ROOT / "assets/crivo-17m"
    target.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    metadata = session.get(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}",
                           params={"blobs": "true"}, timeout=60)
    metadata.raise_for_status()
    upstream = {f["rfilename"]: f for f in metadata.json()["siblings"]}
    hashes = {}
    for name in FILES:
        path = target / name
        expected = upstream[name].get("lfs", {}).get("sha256")
        expected_blob = upstream[name].get("blobId")
        valid = path.exists() and (hashlib.sha256(path.read_bytes()).hexdigest() == expected if expected
                                   else git_blob_sha(path.read_bytes()) == expected_blob)
        if not valid:
            response = session.get(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", timeout=180)
            response.raise_for_status()
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(response.content)
            if expected and hashlib.sha256(response.content).hexdigest() != expected:
                raise ValueError(f"SHA256 mismatch: {name}")
            if not expected and git_blob_sha(response.content) != expected_blob:
                raise ValueError(f"Git blob mismatch: {name}")
            temporary.replace(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name}: {path.stat().st_size:,} bytes", flush=True)
    manifest = {"repo": REPO, "revision": REVISION, "license": "CC-BY-NC-4.0", "sha256": hashes}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    banking = ROOT / "assets/banking77"
    banking.mkdir(parents=True, exist_ok=True)
    for name, expected in BANKING_FILES.items():
        path = banking / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            response = session.get("https://raw.githubusercontent.com/PolyAI-LDN/"
                                   f"task-specific-datasets/{BANKING_REVISION}/banking_data/{name}", timeout=60)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != expected:
                raise ValueError(f"BANKING77 SHA256 mismatch: {name}")
            path.write_bytes(response.content)
        print(f"BANKING77/{name}: verified", flush=True)


if __name__ == "__main__":
    main()
