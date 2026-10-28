"""DINOv2-S/14 and the unmasked, unaugmented AnomalyDINO scoring path."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from movidius_lab.paths import ROOT
MODEL_DIR = ROOT / "assets/dinov2-small"
DATA_DIR = ROOT / "assets/mvtec-ad"
GRAPH_DIR = ROOT / "assets/dino-ir"
SIZE, PATCHES, WIDTH = 224, 256, 384


def prepare_image(image, size=224):
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = image.convert("RGB")
    # Bottle is square; camera inputs use a documented center-square crop.
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
    array = (array - np.array([.485, .456, .406], dtype=np.float32)[:, None, None]) / np.array(
        [.229, .224, .225], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(array[None]), image


def position_embeddings(weights, size=224):
    grid = size // 14
    table = weights["embeddings.position_embeddings"]
    patches = table[:, 1:].reshape(1, 37, 37, WIDTH).permute(0, 3, 1, 2)
    # Match the original Meta / AnomalyDINO non-register backbone's offset.
    patches = F.interpolate(patches, scale_factor=((grid + .1) / 37, (grid + .1) / 37),
                            mode="bicubic", align_corners=False, antialias=False)
    return torch.cat([table[:, :1], patches.permute(0, 2, 3, 1).reshape(1, grid * grid, WIDTH)], dim=1)


class DinoReference:
    def __init__(self, device="cpu", size=224):
        self.device = torch.device(device)
        self.size, self.patches = size, (size // 14) ** 2
        weights = load_file(MODEL_DIR / "model.safetensors")
        self.position = position_embeddings(weights, size).to(self.device)
        self.weights = {k: v.to(self.device) for k, v in weights.items()}
        self.bank = None

    def linear(self, x, prefix):
        return F.linear(x, self.weights[prefix + ".weight"], self.weights.get(prefix + ".bias"))

    def norm(self, x, prefix):
        return F.layer_norm(x, (WIDTH,), self.weights[prefix + ".weight"],
                            self.weights[prefix + ".bias"], 1e-6)

    @torch.inference_mode()
    def features(self, pixels, layers=12, normalize=True):
        x = torch.as_tensor(pixels, device=self.device)
        x = F.conv2d(x, self.weights["embeddings.patch_embeddings.projection.weight"],
                     self.weights["embeddings.patch_embeddings.projection.bias"], stride=14)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([self.weights["embeddings.cls_token"], x], 1) + self.position
        for i in range(layers):
            p = f"encoder.layer.{i}"
            z = self.norm(x, p + ".norm1")
            q, k, v = [self.linear(z, p + ".attention.attention." + key).reshape(1, self.patches + 1, 6, 64).transpose(1, 2)
                       for key in ["query", "key", "value"]]
            z = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, self.patches + 1, WIDTH)
            x = x + self.linear(z, p + ".attention.output.dense") * self.weights[p + ".layer_scale1.lambda1"]
            z = self.linear(F.gelu(self.linear(self.norm(x, p + ".norm2"), p + ".mlp.fc1")), p + ".mlp.fc2")
            x = x + z * self.weights[p + ".layer_scale2.lambda1"]
        if layers != 12:
            result = x
        else:
            result = self.norm(x, "layernorm")[:, 1:]
            if normalize:
                result = F.normalize(result, dim=-1)
        return result

    def set_reference(self, pixels):
        self.bank = self.features(pixels)

    @torch.inference_mode()
    def infer(self, pixels):
        if self.bank is None:
            raise ValueError("Reference image required")
        query = self.features(pixels)
        distances = (2 - 2 * (query @ self.bank.transpose(-1, -2))).clamp_min(0).min(-1).values
        score = distances.topk(2, dim=-1).values.mean(-1, keepdim=True)
        result = torch.cat([distances, score], -1).cpu().numpy()
        return result


def unpack(result):
    values = np.asarray(result).reshape(-1)
    if values.shape != (257,) or not np.isfinite(values).all():
        raise ValueError("Invalid anomaly output")
    return values[:256].reshape(16, 16), float(values[-1])


def bottle_cases():
    root = DATA_DIR / "bottle"
    return [{"id": str(p.relative_to(DATA_DIR)), "path": p,
             "gold": int(p.parent.name != "good"),
             "mask": root / "ground_truth" / p.parent.name / (p.stem + "_mask.png")}
            for p in sorted((root / "test").glob("*/*.png"))]


def manifest():
    return json.loads((MODEL_DIR / "manifest.json").read_text())
