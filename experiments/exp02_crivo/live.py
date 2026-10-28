"""Resident backends and sequential live comparisons, owned by one worker thread."""

import time

import numpy as np
import torch

from experiments.exp02_crivo.model import ROOT, CrivoReference, CrivoTokenizer, banking77
from movidius_lab.myriad import Myriad


class LiveCrivo:
    def __init__(self, progress=lambda value: None):
        torch.set_num_threads(4)
        self.labels, _ = banking77()
        self.tokenizer = CrivoTokenizer()
        self.models, self.errors = {}, {}
        ids, positions = self.tokenizer.banking("I still have not received my new card.", self.labels)
        for backend in ["cpu", "mps", "stick"]:
            progress(f"Chargement de {backend}…")
            try:
                if backend == "stick":
                    model = Myriad(ROOT / "assets/crivo-ir/banking77")
                else:
                    if backend == "mps" and not torch.backends.mps.is_available():
                        raise RuntimeError("GPU Metal indisponible")
                    model = CrivoReference(backend)
                self.models[backend] = model
                self.infer(backend, ids, positions)
            except Exception as error:
                self.errors[backend] = str(error)
                failed = self.models.pop(backend, None)
                if backend == "stick" and failed:
                    failed.close()

    def status(self):
        return {"backends": {key: {"ready": key in self.models, "error": self.errors.get(key)}
                             for key in ["stick", "cpu", "mps"]},
                "options": len(self.labels), "max_tokens": 960}

    def reconnect(self):
        if "stick" not in self.models:
            self.models["stick"] = Myriad(ROOT / "assets/crivo-ir/banking77")
            self.errors.pop("stick", None)
        return self.status()

    def infer(self, backend, ids, positions):
        if backend == "stick":
            probabilities = self.models[backend].infer(ids, positions).flatten()
        else:
            if backend == "cpu":
                # Restore this worker's setting after other native runtimes initialize.
                torch.set_num_threads(4)
            probabilities = self.models[backend].probabilities(ids, positions)
        if not np.isfinite(probabilities).all() or abs(float(probabilities.sum()) - 1) > 0.01:
            raise RuntimeError("Distribution de probabilités invalide")
        return probabilities

    def encoded(self, message):
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Saisis un message en anglais.")
        if len(message) > 4000:
            raise ValueError("Message trop long.")
        ids, positions = self.tokenizer.banking(message.strip(), self.labels)
        if len(ids) > 960:
            raise ValueError(f"Ce message demande {len(ids)} tokens avec les 77 intentions ; maximum 960. Rien n’a été tronqué.")
        return ids, positions

    def compare(self, message, progress=lambda value: None):
        start = time.perf_counter()
        ids, positions = self.encoded(message)
        tokenization_ms = (time.perf_counter() - start) * 1000
        results = {}
        for key in ["stick", "cpu", "mps"]:
            if key not in self.models:
                results[key] = {"error": self.errors.get(key, "Backend indisponible")}
                continue
            progress(f"Inférence : {key} · 3 passages…")
            durations = []
            try:
                for _ in range(3):
                    start = time.perf_counter()
                    probabilities = self.infer(key, ids, positions)
                    durations.append((time.perf_counter() - start) * 1000)
            except Exception as error:
                results[key] = {"error": str(error)}
                continue
            best = int(probabilities.argmax())
            results[key] = {"decision": self.labels[best], "probability": float(probabilities[best]),
                            "probabilities": probabilities.tolist(), "median_ms": float(np.median(durations)),
                            "samples_ms": durations, "precision": "FP16" if key == "stick" else "FP32"}
        return {"tokens": len(ids), "tokenization_ms": tokenization_ms, "labels": self.labels,
                "results": results, "runs": 3, "boundary": "Entrées tokenisées → probabilités reçues ; modèles résidents"}

    def close(self):
        if "stick" in self.models:
            self.models["stick"].close()
