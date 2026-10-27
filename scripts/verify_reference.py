"""Independently check the minimal model against Hugging Face GPT-2."""

import json

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from movidius_lab.device import ROOT
from movidius_lab.gpt2 import GPT2


def main():
    torch.set_num_threads(4)
    model = GPT2(ROOT / "assets/gpt2/model.safetensors")
    config = GPT2Config.from_json_file(str(ROOT / "assets/gpt2/config.json"))
    reference = GPT2LMHeadModel(config).eval()
    state = {"transformer." + k: v for k, v in model.w.items() if not k.endswith(".attn.bias")}
    state["lm_head.weight"] = model.w["wte.weight"]
    reference.load_state_dict(state, strict=True)
    report = {"reference": "transformers.GPT2LMHeadModel", "torch": torch.__version__, "cases": []}
    with torch.inference_mode():
        for ids in [[15496, 995], [464, 3139, 286, 4881, 318], [40, 1101, 257, 2746, 326, 460, 467, 284, 511, 2456]]:
            actual = model.forward(ids, use_cache=False)
            expected = reference(torch.tensor([ids])).logits
            max_error = float((actual - expected).abs().max())
            torch.testing.assert_close(actual, expected, atol=0.0002, rtol=0.0002)
            model.reset()
            cached = torch.cat([model.forward([i]) for i in ids], dim=1)
            cache_error = float((cached - expected).abs().max())
            torch.testing.assert_close(cached, expected, atol=0.0002, rtol=0.0002)
            report["cases"].append({"tokens": ids, "full_max_logit_error": max_error,
                                    "cached_max_logit_error": cache_error})
    (ROOT / "results/reference_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
