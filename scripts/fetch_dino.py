"""Fetch pinned model, original-format bottle images and reference sources."""

import hashlib
import json
from pathlib import Path
import subprocess

from huggingface_hub import snapshot_download
import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL_REV = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
DATA_REV = "c75b39616f84db43677bcc8228caaafaf5096d7f"
DINO_REV = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
ANOMALY_REV = "b9d1c2648e3a5247437d4d953d907a8f3d994457"
MODEL_HASHES = {
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    "README.md": "4c20dca454a8e5c670e8de5c7e6040f512aeca5438516f7623eedc4e3b00599c",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    session = requests.Session()
    model = ROOT / "assets/dinov2-small"
    model.mkdir(parents=True, exist_ok=True)
    for name, expected in MODEL_HASHES.items():
        path = model / name
        if not path.exists() or sha256(path) != expected:
            response = session.get(f"https://huggingface.co/facebook/dinov2-small/resolve/{MODEL_REV}/{name}", timeout=180)
            response.raise_for_status()
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(response.content)
            if sha256(temporary) != expected:
                raise ValueError(f"Model SHA256 mismatch: {name}")
            temporary.replace(path)
        print(f"Verified {name}", flush=True)
    provenance = {"repo": "facebook/dinov2-small", "revision": MODEL_REV, "license": "Apache-2.0",
                  "sha256": MODEL_HASHES, "meta_code_revision": DINO_REV,
                  "anomalydino_code_revision": ANOMALY_REV}
    (model / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    data = ROOT / "assets/mvtec-ad"
    snapshot_download("foersben/mvtec-ad", repo_type="dataset", revision=DATA_REV,
                      allow_patterns=["bottle/**"], local_dir=data, max_workers=4)
    files = sorted((data / "bottle").rglob("*.png"))
    manifest = {"repo": "foersben/mvtec-ad", "revision": DATA_REV,
                "source": "MVTec AD dataset, public mirror; original PNG layout",
                "license": "CC-BY-NC-SA-4.0",
                "sha256": {str(p.relative_to(data)): sha256(p) for p in files}}
    if len(files) != 355 or len(list((data / "bottle/test").glob("*/*.png"))) != 83:
        raise ValueError("Incomplete MVTec AD bottle split")
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    source = ROOT / "third_party/dinov2"
    if not source.exists():
        source.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "remote", "add", "origin", "https://github.com/facebookresearch/dinov2.git"], check=True)
        subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", DINO_REV], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != DINO_REV:
        raise ValueError("Unexpected Meta reference revision")
    source = ROOT / "third_party/anomalydino"
    for name in ["src/backbones.py", "src/detection.py", "src/post_eval.py", "src/utils.py", "LICENSE", "README.md"]:
        path = source / name
        response = session.get(f"https://raw.githubusercontent.com/dammsi/AnomalyDINO/{ANOMALY_REV}/{name}", timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    (source / "revision.txt").write_text(ANOMALY_REV + "\n")
    print("Pinned DINOv2, AnomalyDINO and all bottle images ready.", flush=True)


if __name__ == "__main__":
    main()
