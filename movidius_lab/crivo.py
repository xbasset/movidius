"""Crivo tokenization and an explicit PyTorch reference for the device port."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "assets/crivo-17m"
BANKING_DIR = ROOT / "assets/banking77"


def banking77():
    """Match Decision Index's direct BANKING77 adapter, including option order."""
    labels = json.loads((BANKING_DIR / "categories.json").read_text())
    with (BANKING_DIR / "test.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    return labels, [{"id": f"BANKING77:test:{i}", "text": row["text"],
                     "gold": labels.index(row["category"])} for i, row in enumerate(rows)]


class CrivoTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        self.opt_id = self.tokenizer.convert_tokens_to_ids("[OPT]")

    def encode(self, state, instructions, options):
        def render(value):
            if value is None:
                return ""
            return value.strip() if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

        def ids(value):
            return self.tokenizer(render(value), add_special_tokens=False)["input_ids"]

        tok = self.tokenizer
        tokens = [tok.cls_token_id, *ids(instructions), tok.sep_token_id, *ids(state), tok.sep_token_id]
        positions = []
        for key, description in options.items():
            positions.append(len(tokens))
            description = render(description)
            text = f"{key}: {description}" if description and description != key else key
            tokens.extend([self.opt_id, *ids(text)])
        return np.asarray(tokens, dtype=np.int32), np.asarray(positions, dtype=np.int32)

    def banking(self, text, labels):
        return self.encode({}, "Classify the banking intent of this user request:\n" + text,
                           {f"option_{i}": label for i, label in enumerate(labels)})


class CrivoReference:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.weights = {key: value.to(self.device) for key, value in
                        load_file(MODEL_DIR / "model.safetensors").items()}
        self.config = json.loads((MODEL_DIR / "config.json").read_text())

    def linear(self, x, prefix):
        return F.linear(x, self.weights[prefix + ".weight"], self.weights.get(prefix + ".bias"))

    def norm(self, x, prefix, eps=1e-5):
        return F.layer_norm(x, (x.shape[-1],), self.weights[prefix + ".weight"],
                            self.weights.get(prefix + ".bias"), eps)

    @torch.inference_mode()
    def logits(self, ids, opt_positions, question_type=1):
        ids = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=self.device)
        positions = torch.as_tensor(np.asarray(opt_positions), dtype=torch.long, device=self.device)
        n = ids.shape[0]
        x = self.norm(F.embedding(ids, self.weights["encoder.embeddings.tok_embeddings.weight"]),
                      "encoder.embeddings.norm")
        freqs = torch.outer(torch.arange(n, device=self.device, dtype=torch.float32),
                            1.0 / (160000.0 ** (torch.arange(0, 64, 2, device=self.device) / 64.0)))
        cos = torch.cat((freqs, freqs), -1).cos()[None]
        sin = torch.cat((freqs, freqs), -1).sin()[None]

        def rotary(a):
            return a * cos + torch.cat((-a[..., 32:], a[..., :32]), -1) * sin

        for i in range(7):
            p = f"encoder.layers.{i}"
            z = self.norm(x, p + ".attn_norm") if i else x
            qkv = self.linear(z, p + ".attn.Wqkv").reshape(n, 3, 4, 64)
            q, k, v = (qkv[:, j].transpose(0, 1) for j in range(3))
            mask = None
            if self.config["layer_types"][i] == "sliding_attention":
                loc = torch.arange(n, device=self.device)
                mask = (loc[:, None] - loc[None, :]).abs() <= 64
            a = F.scaled_dot_product_attention(rotary(q), rotary(k), v, attn_mask=mask)
            x = x + self.linear(a.transpose(0, 1).reshape(n, 256), p + ".attn.Wo")
            u, gate = self.linear(self.norm(x, p + ".mlp_norm"), p + ".mlp.Wi").chunk(2, -1)
            x = x + self.linear(F.gelu(u) * gate, p + ".mlp.Wo")
        x = self.norm(x, "encoder.final_norm")[positions] + self.weights["type_embedding.weight"][question_type]
        x = self.linear(F.gelu(self.linear(self.norm(x, "scorer.0"), "scorer.1")), "scorer.3")
        return x.flatten()

    def probabilities(self, ids, opt_positions):
        logits = self.logits(ids, opt_positions)
        # Metal supports FP32, while the author's CPU scoring uses FP64 softmax.
        if self.device.type == "cpu":
            logits = logits.double()
        return logits.softmax(-1).cpu().numpy()


def macro_f1(gold, prediction, classes=77):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (gold, prediction), 1)
    tp = np.diag(matrix)
    denom = matrix.sum(0) + matrix.sum(1)
    return float(np.divide(2 * tp, denom, out=np.zeros(classes, dtype=np.float64), where=denom != 0).mean())


def model_manifest():
    return json.loads((MODEL_DIR / "manifest.json").read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
