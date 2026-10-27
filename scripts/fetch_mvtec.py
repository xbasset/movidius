"""Fetch all MVTec AD test images/masks and the first three train references."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import threading

from huggingface_hub import HfApi
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from movidius_lab.dino import DATA_DIR
from scripts.fetch_dino import DATA_REV

CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
              "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
local = threading.local()


def fetch(file):
    path = DATA_DIR / file.path
    if ".." in Path(file.path).parts or Path(file.path).is_absolute():
        raise ValueError("Invalid dataset path")
    def valid():
        if not path.exists():
            return False
        data = path.read_bytes()
        if file.lfs:
            return hashlib.sha256(data).hexdigest() == file.lfs.sha256
        return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == file.blob_id
    if not valid():
        if not hasattr(local, "session"):
            local.session = requests.Session()
            retry = Retry(total=6, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
            local.session.mount("https://", HTTPAdapter(max_retries=retry))
        response = local.session.get(f"https://huggingface.co/datasets/foersben/mvtec-ad/resolve/{DATA_REV}/{file.path}", timeout=120)
        response.raise_for_status()
        expected = file.lfs.sha256 if file.lfs else file.blob_id
        digest = hashlib.sha256(response.content).hexdigest() if file.lfs else hashlib.sha1(
            f"blob {len(response.content)}\0".encode() + response.content).hexdigest()
        if digest != expected:
            raise ValueError(f"Dataset hash mismatch: {file.path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return file.path, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = [x for x in HfApi().list_repo_tree("foersben/mvtec-ad", repo_type="dataset",
             revision=DATA_REV, recursive=True) if hasattr(x, "size")]
    selected = []
    counts = {}
    for category in CATEGORIES:
        refs = sorted([x for x in files if x.path.startswith(category + "/train/good/")
                       and x.path.endswith(".png")], key=lambda x: x.path)[:3]
        test = [x for x in files if x.path.startswith(category + "/test/") and x.path.endswith(".png")]
        masks = [x for x in files if x.path.startswith(category + "/ground_truth/") and x.path.endswith(".png")]
        counts[category] = len(test)
        selected.extend(refs + test + masks)
    if sum(counts.values()) != 1725 or any(n == 0 for n in counts.values()):
        raise ValueError("Incomplete fifteen-category MVTec AD test split")
    needed = sum(x.size for x in selected if not (DATA_DIR / x.path).exists())
    if shutil.disk_usage(DATA_DIR).free < needed + 512 * 1024**2:
        raise RuntimeError("Insufficient disk space for selected dataset files")
    print(f"Verifying/downloading {len(selected)} files, {needed / 1024**3:.2f} GiB missing", flush=True)
    hashes = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch, file) for file in selected]):
            name, digest = future.result()
            hashes[name] = digest
            if len(hashes) % 100 == 0:
                print(f"{len(hashes)}/{len(selected)} files ready", flush=True)
    manifest = {"repo": "foersben/mvtec-ad", "revision": DATA_REV, "license": "CC-BY-NC-SA-4.0",
                "categories": CATEGORIES, "test_counts": counts, "test_images": 1725,
                "references_per_category": 3, "sha256": dict(sorted(hashes.items()))}
    (DATA_DIR / "manifest_15cat.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("All 15 test categories and references verified.", flush=True)


if __name__ == "__main__":
    main()
