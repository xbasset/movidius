"""Download public GPT-2 weights, record the revision and verify file identity."""

import hashlib
import json
import os
from pathlib import Path

import requests

from movidius_lab.paths import ROOT
DEST = ROOT / "assets/gpt2"
DEST.mkdir(parents=True, exist_ok=True)
repo = "openai-community/gpt2"
revision = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
expected = {
    "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
    "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
}
files = {}
for filename in ["config.json", "model.safetensors"]:
    target = DEST / filename
    if not target.exists():
        with requests.get(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", stream=True, timeout=120) as r:
            r.raise_for_status()
            temp = target.with_suffix(target.suffix + ".partial")
            with temp.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
            temp.replace(target)
    with target.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != expected[filename]:
        raise ValueError(f"{target}: SHA256 mismatch; remove it and retry")
    files[filename] = {"bytes": target.stat().st_size, "sha256": digest}
    print(filename, files[filename], flush=True)
os.environ["TIKTOKEN_CACHE_DIR"] = str(ROOT / "assets/tiktoken")
import tiktoken
tiktoken.get_encoding("gpt2")
(DEST / "manifest.json").write_text(json.dumps({"repo": repo, "revision": revision, "files": files}, indent=2) + "\n")
