"""Check the 448 reference, complete device features, PCA masks and regressions."""

import ast
import argparse
from contextlib import nullcontext
import json

import cv2
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

from experiments.exp03_anomalydino.paper import MASKED, paper_mask
from movidius_lab.paths import sha256
from experiments.exp03_anomalydino.model import ROOT, DATA_DIR, GRAPH_DIR, DinoReference, prepare_image
from movidius_lab.legacy_ir import IR
from movidius_lab.myriad import Myriad
from experiments.exp03_anomalydino.pca_ir import pca_mask
from experiments.exp03_anomalydino.scripts.export_dino import export
from experiments.exp03_anomalydino.scripts.verify_dino_reference import official_model


def save(g, output, name):
    path = GRAPH_DIR / '448' / name
    g.save(path, output)
    path.with_suffix('.json').write_text(json.dumps({'output_shape': output.shape}))
    return path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    reference = DinoReference(size=448)
    original = official_model(reference)
    source = ast.parse((ROOT / 'third_party/anomalydino/src/backbones.py').read_text())
    wrapper = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'DINOv2Wrapper')
    method = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_background_mask')
    namespace = {'np': np, 'cv2': cv2, 'PCA': PCA}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'author_mask', 'exec'), namespace)
    examples = [DATA_DIR / c / 'test' / ('good/000.png') for c in ['bottle', *sorted(MASKED)]]
    features, rows = [], []
    context = nullcontext(None) if args.reference_only else Myriad(GRAPH_DIR / '448/features')
    with context as encoder, torch.inference_mode():
        for path in examples:
            pixels, _ = prepare_image(path, 448)
            raw = reference.features(pixels, normalize=False)
            official = original.get_intermediate_layers(torch.from_numpy(pixels))[0]
            error = float((raw - official).abs().max())
            assert error < 1e-4, error
            if args.reference_only:
                row = {'id': str(path.relative_to(DATA_DIR)), 'meta_feature_max_error': error}
                if path.parts[-4] in MASKED:
                    np.random.seed(0)
                    author = namespace['compute_background_mask'](None, raw.numpy()[0], (32, 32), masking_type=True)
                    assert np.array_equal(paper_mask(raw.numpy()[0], True), author)
                    row['author_mask_matches'] = True
                rows.append(row)
                continue
            target = F.normalize(raw, dim=-1).numpy()
            actual = encoder.infer_tensors({'pixels': pixels.astype(np.float16)})
            cosine = np.sum(actual * target, axis=-1) / np.linalg.norm(actual, axis=-1)
            assert cosine.mean() > .99 and np.abs(actual-target).max() < .15
            rows.append({'id': str(path.relative_to(DATA_DIR)), 'meta_feature_max_error': error,
                         'device_feature_mae': float(np.abs(actual-target).mean()),
                         'device_feature_max_error': float(np.abs(actual-target).max()),
                         'mean_feature_cosine': float(cosine.mean())})
            features.append(raw.numpy())
    g = IR('pca_validation')
    x = g.input('features', (1, 1024, 384))
    output = pca_mask(g, x)
    path = save(g, output, 'pca_validation')
    context = nullcontext(None) if args.reference_only else Myriad(path)
    with context as model:
        for raw, row in zip(features[1:], rows[1:]):
            expected = paper_mask(raw[0], True)
            np.random.seed(0)
            author = namespace['compute_background_mask'](None, raw[0], (32, 32), masking_type=True)
            assert np.array_equal(expected, author)
            actual = model.infer_tensors({'features': raw.astype(np.float16)}).ravel().astype(bool)
            row['pca_mask_pixels_different'] = int(np.count_nonzero(expected != actual))
            assert row['pca_mask_pixels_different'] <= 32
    g, output = export()
    path = GRAPH_DIR / '224_regression'
    g.save(path, output)
    hashes = {suffix: sha256(path.with_suffix(suffix)) for suffix in ['.xml', '.bin']}
    assert all(value == sha256((GRAPH_DIR / 'anomaly').with_suffix(suffix)) for suffix, value in hashes.items())
    result = {'passed': True, 'resolution': [448, 448], 'cases': rows, 'unchanged_224_graph_sha256': hashes,
              'pca': ('FP32 reference mask vs pinned author code' if args.reference_only else
                      'same raw FP32 features as input; FP16 device power iteration vs original randomized PCA'),
              'metric_validation': 'experiments.exp03_anomalydino.scripts.check_anomaly_metrics: sklearn and independent threshold PRO, ties and chunk boundaries'}
    result['reference_only'] = args.reference_only
    name = 'dino448_reference_verification' if args.reference_only else 'dino448_verification'
    (ROOT / f'experiments/exp03_anomalydino/results/448/{name}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
