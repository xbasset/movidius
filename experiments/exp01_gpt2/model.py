"""GPT-2 small inference with explicit KV caching and an optional device hook."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


class GPT2:
    def __init__(self, path, device="cpu"):
        self.w = load_file(str(path), device=device)
        self.device = device
        self.projection_hook = None
        self.linear_hook = None
        self.cache = None

    def reset(self):
        self.cache = None

    def norm(self, x, prefix):
        return F.layer_norm(x, (768,), self.w[prefix + ".weight"], self.w[prefix + ".bias"], 1e-5)

    def linear(self, x, prefix):
        if self.linear_hook is not None:
            return self.linear_hook(x, prefix, self.w[prefix + ".weight"], self.w[prefix + ".bias"])
        return x @ self.w[prefix + ".weight"] + self.w[prefix + ".bias"]

    @torch.inference_mode()
    def forward(self, tokens, use_cache=True, last_only=False):
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.device).reshape(1, -1)
        past = self.cache[0][0].shape[2] if use_cache and self.cache else 0
        length = tokens.shape[1]
        if past + length > 1024:
            raise ValueError("GPT-2's learned positional embeddings stop at 1024 tokens; reset the cache")
        positions = torch.arange(past, past + length, device=self.device)
        x = self.w["wte.weight"][tokens] + self.w["wpe.weight"][positions]
        new_cache = []
        for layer in range(12):
            prefix = f"h.{layer}"
            normalized = self.norm(x, prefix + ".ln_1")
            if self.projection_hook is not None and layer == 0:
                qkv = self.projection_hook(normalized)
            else:
                qkv = self.linear(normalized, prefix + ".attn.c_attn")
            q, k, v = [t.reshape(1, length, 12, 64).transpose(1, 2) for t in qkv.split(768, dim=-1)]
            if past:
                k = torch.cat((self.cache[layer][0], k), dim=2)
                v = torch.cat((self.cache[layer][1], v), dim=2)
            new_cache.append((k, v))
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(64)
            visible = torch.arange(k.shape[2], device=self.device)[None, :] <= positions[:, None]
            scores = scores.masked_fill(~visible, -torch.inf)
            attention = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(1, length, 768)
            x = x + self.linear(attention, prefix + ".attn.c_proj")
            hidden = self.linear(self.norm(x, prefix + ".ln_2"), prefix + ".mlp.c_fc")
            hidden = F.gelu(hidden, approximate="tanh")
            x = x + self.linear(hidden, prefix + ".mlp.c_proj")
        if use_cache:
            self.cache = new_cache
        x = self.norm(x[:, -1:] if last_only else x, "ln_f")
        if self.linear_hook is not None:
            return self.linear_hook(x, "lm_head", self.w["wte.weight"].T, None)
        return x @ self.w["wte.weight"].T


class DeviceQKV:
    """Offload only block 0's QKV projection; all other operations stay on host."""

    def __init__(self, device, model):
        import numpy as np
        from movidius_lab.graph import dense_graph
        self.device = device
        self.np = np
        self.graph = device.allocate(dense_graph(
            model.w["h.0.attn.c_attn.weight"].cpu().numpy(),
            model.w["h.0.attn.c_attn.bias"].cpu().numpy()))
        self.calls = 0
        self.wall_seconds = 0.0

    def __call__(self, x):
        import time
        results = []
        for row in x.reshape(-1, 768).cpu().numpy():
            start = time.perf_counter()
            results.append(self.device.run(self.graph, row))
            self.wall_seconds += time.perf_counter() - start
            self.calls += 1
        return torch.from_numpy(self.np.stack(results).astype(self.np.float32)).to(x.device).reshape(*x.shape[:-1], 2304)

    def close(self):
        self.device.release(self.graph)


class DeviceDense:
    """Experimental graph streaming; currently fails during graph replacement."""

    def __init__(self, device, model):
        self.device = device
        self.blobs = {}
        self.calls = 0
        self.upload_bytes = 0
        self.wall_seconds = 0.0
        self.operations = {}
        # Test the buffer-capacity hypothesis by reserving the largest shapes.
        # This did not fix the observed third-graph failure; retained to reproduce it.
        import numpy as np
        from movidius_lab.graph import dense_graph
        blob = dense_graph(model.w["wte.weight"].T.cpu().numpy())
        graph = device.allocate(blob)
        try:
            device.run(graph, np.zeros(3072, dtype=np.float16))
        finally:
            device.release(graph)
        self.blobs["lm_head"] = blob

    def __call__(self, x, name, weight, bias):
        import time
        import numpy as np
        from movidius_lab.graph import dense_graph
        if name not in self.blobs:
            self.blobs[name] = dense_graph(weight.cpu().numpy(), None if bias is None else bias.cpu().numpy())
        blob = self.blobs[name]
        start = time.perf_counter()
        graph = self.device.allocate(blob)
        results = []
        try:
            for row in x.reshape(-1, weight.shape[0]).cpu().numpy():
                padded = np.pad(row, (0, 3072-len(row)))
                results.append(self.device.run(graph, padded))
                self.calls += 1
                self.operations[name] = self.operations.get(name, 0) + 1
        finally:
            self.device.release(graph)
        self.wall_seconds += time.perf_counter() - start
        self.upload_bytes += len(blob)
        return torch.from_numpy(np.stack(results).astype(np.float32)).to(x.device).reshape(*x.shape[:-1], weight.shape[1])

    def close(self):
        pass
