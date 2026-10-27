"""Resident backends and sequential live comparisons, owned by one worker thread."""

import time

import numpy as np
import torch

from movidius_lab.crivo import ROOT, CrivoReference, CrivoTokenizer, banking77
from movidius_lab.energy import MacEnergy
from movidius_lab.myriad import Myriad


class LiveCrivo:
    def __init__(self, progress=lambda value: None):
        torch.set_num_threads(4)
        self.labels, _ = banking77()
        self.tokenizer = CrivoTokenizer()
        self.models, self.errors = {}, {}
        self.energy, self.energy_error = None, None
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
        try:
            self.energy = MacEnergy()
        except Exception as error:
            self.energy_error = str(error)

    def status(self):
        return {"backends": {key: {"ready": key in self.models, "error": self.errors.get(key)}
                             for key in ["stick", "cpu", "mps"]},
                "energy": {"available": self.energy is not None, "error": self.energy_error,
                           "channels": self.energy.metadata if self.energy else [],
                           "scope": "Estimation des composants CPU + GPU + ANE du Mac ; consommation du stick exclue"},
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

    def window(self, operation):
        before = self.energy.snapshot()
        after = None
        try:
            value = operation()
            after = self.energy.snapshot()
            return value, self.energy.delta(before, after)
        finally:
            self.energy.release(before)
            if after:
                self.energy.release(after)

    def measure_energy(self, message, seconds=3.0, progress=lambda value: None):
        if self.energy is None:
            raise RuntimeError(self.energy_error or "Compteurs énergétiques indisponibles")
        ids, positions = self.encoded(message)
        # Warm each input shape before the counter windows, including Metal compilation.
        for key in self.models:
            self.infer(key, ids, positions)
        results = {}
        for key in ["cpu", "mps", "stick"]:
            if key not in self.models:
                results[key] = {"error": self.errors.get(key, "Backend indisponible")}
                continue
            progress(f"Énergie : repos avant {key}…")
            time.sleep(.5)
            _, idle = self.window(lambda: time.sleep(1))
            progress(f"Énergie : {key} · lot d’au moins {seconds:g} secondes…")

            def workload():
                count, started = 0, time.perf_counter()
                while time.perf_counter() - started < seconds:
                    self.infer(key, ids, positions)
                    count += 1
                return count

            count, sample = self.window(workload)
            if sample["sum_j"] <= 0:
                raise RuntimeError("Compteurs énergétiques inactifs ; aucune estimation fiable disponible")
            incremental = sample["sum_j"] - idle["sum_w"] * sample["seconds"]
            results[key] = {**sample, "inferences": count, "idle": idle,
                            "gross_j_per_inference": sample["sum_j"] / count,
                            "incremental_j_per_inference": incremental / count,
                            "stick_electrical_j": None,
                            "interpretation": "Mac accompagnant le stick ; stick exclu" if key == "stick"
                                              else "Composants du Mac pendant l’inférence"}
        return {"results": results, "target_seconds": seconds, "tokens": len(ids),
                "host_torch_threads": 4,
                "stick_energy": {"available": False, "reason": "Wattmètre USB absent"},
                "caveat": "Estimation de composants à l’échelle du Mac, sensible aux autres applications ; ni consommation électrique totale ni mesure du stick."}

    def close(self):
        if "stick" in self.models:
            self.models["stick"].close()
        if self.energy:
            self.energy.close()
