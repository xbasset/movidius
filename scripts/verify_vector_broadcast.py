"""Reproduce the large-vector broadcast error and its constant-expansion fix."""

import json

import numpy as np
import torch
import torch.nn.functional as F

from movidius_lab.dino import ROOT, GRAPH_DIR, DinoReference
from movidius_lab.legacy_ir import IR
from movidius_lab.myriad import Myriad


def main():
    torch.set_num_threads(4)
    weights = DinoReference().weights
    weight, bias = [weights['encoder.layer.0.norm1.' + suffix] for suffix in ['weight', 'bias']]
    rows = []
    for count in [257, 1025]:
        values = np.random.default_rng(0).standard_normal((1, count, 384)).astype(np.float16)
        target = F.layer_norm(torch.from_numpy(values.astype(np.float32)), (384,), weight, bias, 1e-6).numpy()
        for expanded in [False, True]:
            graph = IR('broadcast_check')
            graph.expand_vector_constants = expanded
            x = graph.input('values', values.shape)
            y = graph.norm('norm', x, weight.numpy(), bias.numpy(), 1e-6)
            path = GRAPH_DIR / '448/broadcast_check'
            graph.save(path, y)
            path.with_suffix('.json').write_text(json.dumps({'output_shape': y.shape}))
            with Myriad(path) as model:
                actual = model.infer_tensors({'values': values})
            error = np.abs(actual - target)
            rows.append({'tokens': count, 'expanded_constants': expanded, 'mae': float(error.mean()),
                         'max_error': float(error.max()), 'rows_with_error_above_0_1': int((error.max(-1) > .1).sum())})
            if expanded:
                assert error.max() < .05
    graph = IR('topk_values_check')
    values = np.random.default_rng(1).standard_normal((1, 1024)).astype(np.float16)
    x = graph.input('values', values.shape)
    y = graph.topk('topk', x, 10, 1)
    path = GRAPH_DIR / '448/topk_values_check'
    graph.save(path, y)
    path.with_suffix('.json').write_text(json.dumps({'output_shape': y.shape}))
    with Myriad(path) as model:
        actual = model.infer_tensors({'values': values})
    np.testing.assert_array_equal(actual, np.sort(values, axis=1)[:, -10:][:, ::-1].astype(np.float32))
    result = {'passed': True, 'input': 'seed-0 standard normal, rounded FP16',
              'operation': 'DINOv2 first LayerNorm, trained gamma and bias', 'cases': rows,
              'topk_values_output_with_unused_integer_indices': 'passed'}
    (ROOT / 'results/dino448_broadcast_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
