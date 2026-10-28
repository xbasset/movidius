"""Regression check for XLink shutdown: repeat a real decision across device resets."""

import json

from experiments.exp02_crivo.model import ROOT, CrivoTokenizer
from movidius_lab.myriad import Myriad


def main():
    tokenizer = CrivoTokenizer()
    ids, positions = tokenizer.encode(
        {"ticket": "My card was charged twice for the same order."},
        "What does the customer need?",
        {"refund": "Money back for a duplicate charge", "cancel": "Cancel the order", "info": "Ask for information"})
    checks = []
    for i in range(3):
        with Myriad(ROOT / "assets/crivo-ir/demo") as model:
            probabilities = model.infer(ids, positions).flatten()
            assert int(probabilities.argmax()) == 0
        checks.append({"cycle": i + 1, "prediction": "refund",
                       "probabilities": probabilities.tolist(), "closed": True})
        print(f"Open/infer/close cycle {i + 1} passed", flush=True)
    (ROOT / "experiments/exp02_crivo/results/crivo_lifecycle_verification.json").write_text(
        json.dumps({"checks": checks, "passed": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
