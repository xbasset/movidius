"""Exercise physical inference, reference replacement and the camera image API."""

import argparse
import base64
from io import BytesIO
import json
import time

import numpy as np
import requests

from movidius_lab.dino import ROOT, DATA_DIR, prepare_image


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    session = requests.Session()

    def wait(identifier):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            result = session.get(base + "/api/jobs/" + identifier, timeout=5).json()
            if result["state"] == "error":
                raise RuntimeError(result["error"])
            if result["state"] == "complete":
                return result["result"]
            time.sleep(.15)
        raise TimeoutError("Demo request did not finish")

    def run(**payload):
        response = session.post(base + "/api/run", json=payload, timeout=5)
        assert response.status_code == 202, response.text
        return wait(response.json()["id"])

    status = session.get(base + "/api/status", timeout=5).json()
    assert status["ready"] and all(x["ready"] for x in status["backends"].values()), status
    assert session.get(base + "/api/status", headers={"Host": "foreign.example"}, timeout=5).status_code == 403
    assert session.post(base + "/api/run", json={}, headers={"Origin": "https://foreign.example"}, timeout=5).status_code == 403
    assert session.post(base + "/api/run", data='x' * 1_500_001, headers={"Content-Type": "application/json"}, timeout=5).status_code == 400
    assert session.post(base + "/api/run", json={"kind": "energy"}, timeout=5).status_code == 400
    normal = run(example="normal", runs=1)
    job = session.post(base + "/api/run", json={"example": "broken"}, timeout=5).json()
    assert session.post(base + "/api/run", json={"example": "normal"}, timeout=5).status_code == 400
    broken = wait(job["id"])
    for key in ["stick", "cpu", "mps"]:
        row = broken["results"][key]
        values = np.asarray(row["patch_distances"])
        assert values.shape == (16, 16) and np.isfinite(values).all()
        assert abs(np.sort(values.ravel())[-2:].mean() - row["score"]) < .002
        assert row["score"] > normal["results"][key]["score"] and row["median_ms"] > 0
    # Public data exercises the same PNG transport used by the browser camera.
    _, image = prepare_image(DATA_DIR / "bottle/train/good/001.png")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    calibrated = run(kind="reference", image=encoded)
    assert not calibrated["errors"]
    same = run(image=encoded, runs=1)
    for key in ["stick", "cpu", "mps"]:
        assert same["results"][key]["score"] < .02
    reset = run(kind="reference", example="reference")
    assert not reset["errors"]
    summary = {"passed": True, "backends": ["stick", "cpu", "mps"],
               "checks": ["foreign Host/Origin rejected", "bounded requests", "one active job",
                          "energy excluded", "physical stick inference", "16x16 maps and device top-two score",
                          "broken score exceeds normal", "PNG camera transport using public data",
                          "reference replacement and restoration without device replug"],
               "physical_camera_tested_by_this_script": False,
               "sample_scores": {key: {"normal": normal["results"][key]["score"],
                                       "broken": broken["results"][key]["score"],
                                       "self_reference": same["results"][key]["score"]}
                                 for key in ["stick", "cpu", "mps"]}}
    (ROOT / "results/dino_live_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
