"""Boot the supplied firmware and optionally execute a known reference network."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from movidius_lab.device import NCS, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inception", action="store_true")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--output", default="results/device_probe.json")
    args = p.parse_args()
    report = {"date": datetime.now(timezone.utc).date().isoformat(), "device_inference": False}
    fw = ROOT / "build/mvnc/MvNCAPI.mvcmd"
    report["firmware_sha256"] = hashlib.sha256(fw.read_bytes()).hexdigest()
    try:
        start = time.perf_counter()
        with NCS(verbose=True) as device:
            report.update(open_seconds=time.perf_counter() - start, boot_succeeded=True)
            print("Firmware booted; device opened", device.name, flush=True)
            if args.inception:
                from PIL import Image
                base = ROOT / "third_party/ncsdk/examples"
                graph_path = base / "tensorflow/inception_v1/graph"
                image_path = base / "data/images/nps_electric_guitar.png"
                image = Image.open(image_path).convert("RGB")
                w, h = image.size
                side = min(w, h)
                image = image.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2)).resize((224, 224))
                tensor = (np.asarray(image, dtype=np.float32) - 128) / 128
                data = graph_path.read_bytes()
                report["graph_sha256"] = hashlib.sha256(data).hexdigest()
                graph = device.allocate(data)
                try:
                    times, device_times = [], []
                    for i in range(args.runs + 1):
                        start = time.perf_counter()
                        output = device.run(graph, tensor)
                        elapsed = time.perf_counter() - start
                        timings = device.timings(graph).tolist()
                        if i:
                            times.append(elapsed)
                            device_times.append(sum(timings))
                    labels = (base / "tensorflow/inception_v1/categories.txt").read_text().splitlines()
                    labels = [s for s in labels if s != "classes"]
                    top = np.argsort(output)[-5:][::-1]
                    report.update(device_inference=True, model="inception_v1", runs=args.runs,
                                  end_to_end_seconds=times, device_milliseconds=device_times,
                                  top5=[{"index": int(i), "label": labels[i], "probability": float(output[i])} for i in top])
                    print(json.dumps(report["top5"], indent=2), flush=True)
                finally:
                    device.release(graph)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        path = ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
