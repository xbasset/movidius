"""Exercise the running local UI API, real inference, and energy accounting."""

import argparse
import json
import time

import requests

from movidius_lab.crivo import ROOT


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    session = requests.Session()

    def wait(identifier):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            result = session.get(base + "/api/jobs/" + identifier, timeout=5).json()
            if result["state"] != "running":
                return result
            time.sleep(.2)
        raise TimeoutError("Live measurement did not complete")

    for _ in range(100):
        status = session.get(base + "/api/status", timeout=5).json()
        if status["ready"]:
            break
        time.sleep(.2)
    assert status["ready"] and all(x["ready"] for x in status["backends"].values())
    assert session.get(base + "/api/status", headers={"Host": "external.example"}, timeout=5).status_code == 403
    assert session.post(base + "/api/run", json={"message": "test"},
                        headers={"Origin": "https://external.example"}, timeout=5).status_code == 403
    assert session.post(base + "/api/run", data="x" * 20000,
                        headers={"Content-Type": "application/json"}, timeout=5).status_code == 400
    invalid = session.post(base + "/api/run", json={"message": "long " * 700}, timeout=5).json()
    assert wait(invalid["id"])["state"] == "error"

    message = "I still have not received my new card, I ordered over a week ago."
    job = session.post(base + "/api/run", json={"message": message}, timeout=5).json()
    assert session.post(base + "/api/run", json={"message": message}, timeout=5).status_code == 400
    comparison = wait(job["id"])
    assert comparison["state"] == "complete", comparison
    for value in comparison["result"]["results"].values():
        assert value["decision"] == "card_arrival" and value["median_ms"] > 0
        assert len(value["probabilities"]) == 77 and len(value["samples_ms"]) == 3

    job = session.post(base + "/api/run", json={"kind": "energy", "message": message}, timeout=5).json()
    energy = wait(job["id"])
    assert energy["state"] == "complete", energy
    data = energy["result"]
    for value in data["results"].values():
        assert value["seconds"] >= 3 and value["inferences"] >= 1
        assert value["sum_j"] > 0
        assert abs(value["sum_j"] - sum(value["components_j"].values())) < 1e-9
        assert abs(value["sum_w"] * value["seconds"] - value["sum_j"]) < 1e-9
        assert abs(value["gross_j_per_inference"] * value["inferences"] - value["sum_j"]) < 1e-9
        assert value["stick_electrical_j"] is None
    assert data["stick_energy"]["available"] is False
    artifact = {**data, "fixture": "BANKING77 public test case 1, repeated; one short sampling session",
                "software_only": True, "channel_metadata": status["energy"]["channels"]}
    (ROOT / "results/crivo_mac_energy.json").write_text(json.dumps(artifact, indent=2) + "\n")
    summary = {"passed": True, "backends": list(comparison["result"]["results"]),
               "checks": ["host/origin restrictions", "request bounds", "no truncation", "one active job",
                          "physical stick + CPU + Metal decisions", "77 probabilities", "energy unit accounting",
                          "per-backend idle windows", "stick energy explicitly unmeasured"]}
    (ROOT / "results/crivo_live_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
