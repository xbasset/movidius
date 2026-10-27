"""Export the complete 448-square paper head, including query masking."""

import argparse
import json

import numpy as np

from movidius_lab.dino import GRAPH_DIR
from movidius_lab.legacy_ir import Tensor
from movidius_lab.pca_ir import pca_mask
from scripts.export_dino import export


def build(trials=3, iterations=64):
    g, query = export('features', size=448)
    layer = next(layer for layer in g.layers if layer.attrib['name'] == 'patch_features')
    port = layer.find('output/port')
    raw = Tensor(int(layer.attrib['id']), int(port.attrib['id']), (1, 1024, 384))
    mask = pca_mask(g, raw, iterations)
    enabled = g.input('mask_enabled', (1, 1, 1))
    disabled = g.power('mask_disabled', enabled, scale=-1, shift=1)
    mask = g.eltwise('optional_mask', g.eltwise('enabled_mask', mask, enabled, 'prod'), disabled, 'sum')
    axes = g.const('reference_axis', [2], 'I32')
    patch_axis = g.const('patch_axis', [1], 'I32')
    outputs = []
    for seed in range(trials):
        name = f'head{seed}'
        bank = g.input(f'reference_bank_{seed}', (1, 8192, 384))
        similarity = g.gemm(name + '/similarity', query, bank, transpose_b=True)
        distance = g.power(name + '/cosine_distance', similarity, scale=-1, shift=1)
        distance = g.node(name + '/nonnegative', 'ReLU', [distance], distance.shape)
        distance = g.node(name + '/nearest', 'ReduceMin', [distance, axes], (1, 1024, 1), {'keep_dims': 'true'})
        distance = g.eltwise(name + '/masked', distance, mask, 'prod')
        flat = g.reshape(name + '/flat', distance, (1, 1024))
        top = g.topk(name + '/top10', flat, 10, 1)
        score = g.node(name + '/score', 'ReduceMean', [top, patch_axis], (1, 1), {'keep_dims': 'true'})
        outputs.append(g.node(name + '/output', 'Concat', [flat, score], (1, 1025), {'axis': 1}))
    output = g.node('anomaly_output', 'Concat', outputs, (trials, 1025), {'axis': 0})
    return g, output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--trials', type=int, choices=[1, 3], default=3)
    parser.add_argument('--iterations', type=int, default=64)
    args = parser.parse_args()
    g, output = build(args.trials, args.iterations)
    path = GRAPH_DIR / f'448/paper_{args.trials}'
    g.save(path, output)
    meta = {'output_shape': output.shape, 'resolution': [448, 448], 'precision': 'FP16',
            'reference_trials': args.trials, 'shots_per_trial': 1, 'rotations_per_reference': 8,
            'pca': {'solver': 'normalized power iteration', 'iterations': args.iterations,
                    'centered_matrix_scale': 1/128, 'query_threshold': 10, 'morphology': 'author N-by-1 layout'},
            'vector_constants_expanded': True, 'weight_bytes': len(g.weights), 'ir_layers': len(g.layers),
            'distance': '1 minus cosine similarity', 'image_score': 'mean of ten largest patch distances'}
    path.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
