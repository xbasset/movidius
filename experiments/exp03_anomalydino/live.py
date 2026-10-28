"""Resident anomaly comparison; camera images and references stay in memory."""

import base64
from io import BytesIO
import time

import numpy as np
from PIL import Image
import torch

from experiments.exp03_anomalydino.model import DATA_DIR, GRAPH_DIR, DinoReference, prepare_image, unpack
from movidius_lab.myriad import Myriad

EXAMPLES = {"reference": "train/good/000.png", "normal": "test/good/000.png",
            "broken": "test/broken_large/000.png", "contamination": "test/contamination/000.png"}


class DinoStick:
    def __init__(self):
        self.model = None

    def set_reference(self, pixels):
        self.close()
        with Myriad(GRAPH_DIR / "features") as encoder:
            bank = encoder.infer_tensors({"pixels": pixels.astype(np.float16)})
        self.model = Myriad(GRAPH_DIR / "anomaly")
        self.model.set_input("reference_bank", bank.astype(np.float16))

    def infer(self, pixels):
        if self.model is None:
            raise RuntimeError("Référence stick indisponible ; refaire la calibration")
        self.model.set_input("pixels", pixels.astype(np.float16, copy=False))
        return self.model.execute()

    def close(self):
        if self.model:
            self.model.close()
            self.model = None


class LiveDino:
    def __init__(self, progress=lambda value: None):
        torch.set_num_threads(4)
        self.models, self.errors = {}, {}
        self.reference_png, self.reference_label = None, None
        for key in ["cpu", "mps", "stick"]:
            progress(f"Chargement de {key}…")
            try:
                self.models[key] = DinoStick() if key == "stick" else DinoReference(key)
            except Exception as error:
                self.errors[key] = str(error)
        self.calibrate({"example": "reference"}, progress)

    @staticmethod
    def image(payload):
        if "example" in payload:
            key = payload["example"]
            if key not in EXAMPLES:
                raise ValueError("Exemple inconnu")
            return Image.open(DATA_DIR / "bottle" / EXAMPLES[key]), "MVTec bottle / " + EXAMPLES[key]
        encoded = payload.get("image", "")
        if not isinstance(encoded, str) or not encoded.startswith(("data:image/jpeg;base64,", "data:image/png;base64,")):
            raise ValueError("Image JPEG ou PNG requise")
        raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
        if len(raw) > 1024 * 1024:
            raise ValueError("Image trop grande")
        image = Image.open(BytesIO(raw))
        if image.width * image.height > 4_194_304:
            raise ValueError("Image limitée à quatre millions de pixels")
        return image, "Caméra / image locale — en mémoire uniquement"

    def calibrate(self, payload, progress=lambda value: None):
        image, label = self.image(payload)
        pixels, preview = prepare_image(image)
        errors = {}
        for key, model in self.models.items():
            progress(f"Référence normale : {key}…")
            try:
                model.set_reference(pixels)
                model.infer(pixels)
                self.errors.pop(key, None)
            except Exception as error:
                errors[key] = str(error)
                self.errors[key] = str(error)
        if len(errors) == len(self.models):
            raise RuntimeError("Aucun moteur n’a pu encoder cette référence")
        buffer = BytesIO()
        preview.save(buffer, format="PNG")
        self.reference_png, self.reference_label = buffer.getvalue(), label
        return {"reference": label, "errors": errors}

    def status(self):
        return {"backends": {key: {"ready": key in self.models and key not in self.errors,
                                    "error": self.errors.get(key)} for key in ["stick", "cpu", "mps"]},
                "resolution": [224, 224], "reference": self.reference_label,
                "scope": "Encodeur, normalisation, distances et score sur le moteur choisi"}

    def compare(self, payload, progress=lambda value: None):
        image, _ = self.image(payload)
        pixels, _ = prepare_image(image)
        backend, runs = payload.get("backend", "all"), payload.get("runs", 3)
        if backend not in ["all", "stick", "cpu", "mps"] or runs not in [1, 3]:
            raise ValueError("Moteur ou nombre de passages invalide")
        results = {}
        for key in (["stick", "cpu", "mps"] if backend == "all" else [backend]):
            if key not in self.models or key in self.errors:
                results[key] = {"error": self.errors.get(key, "Moteur indisponible")}
                continue
            progress(f"Inférence : {key}…")
            if key == "cpu":
                torch.set_num_threads(4)
            prepared = pixels.astype(np.float16) if key == "stick" else pixels
            times = []
            for _ in range(runs):
                start = time.perf_counter()
                output = self.models[key].infer(prepared)
                times.append((time.perf_counter() - start) * 1000)
            patches, score = unpack(output)
            results[key] = {"score": score, "patch_distances": patches.tolist(),
                            "median_ms": float(np.median(times)), "samples_ms": times,
                            "precision": "FP16" if key == "stick" else "FP32"}
        return {"results": results, "runs": runs, "reference": self.reference_label,
                "boundary": "Image préparée → carte 16×16 et score reçus ; modèles résidents"}

    def close(self):
        if "stick" in self.models:
            self.models["stick"].close()
