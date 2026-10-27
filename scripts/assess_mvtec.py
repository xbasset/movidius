"""Publish complete 448-square measurements, provenance and paper comparison."""

import csv
import json

import numpy as np

from movidius_lab.anomalydino import CATEGORIES, cases
from movidius_lab.crivo import sha256
from movidius_lab.dino import ROOT, DATA_DIR, GRAPH_DIR, manifest

BACKENDS = ['myriad', 'cpu', 'mps']
METRICS = {'image_auroc': 'Image AUROC', 'image_f1_max': 'Image F1-max', 'image_ap': 'Image AP',
           'pixel_auroc': 'Pixel AUROC', 'pixel_f1_max': 'Pixel F1-max', 'aupro_0_3': 'AUPRO ≤ 0.3'}
PAPER = {'image_auroc': (.965, .004), 'image_f1_max': (.960, .002), 'image_ap': (.981, .003),
         'pixel_auroc': (.963, .001), 'pixel_f1_max': (.579, .008), 'aupro_0_3': (.917, .001)}


def read(name):
    return json.loads((ROOT / 'results' / name).read_text())


def write_report(result, inference):
    names = {'myriad': 'Myriad 2 stick, FP16', 'cpu': 'Mac CPU, FP32', 'mps': 'Mac GPU / Metal, FP32'}
    quality = result['quality']
    latency = result['single_head_latency']
    metric_rows = []
    for key, label in METRICS.items():
        paper_mean, paper_std = PAPER[key]
        values = [f'{quality[b]["mean"][key]*100:.2f} ± {quality[b]["std"][key]*100:.2f}' for b in BACKENDS]
        metric_rows.append(f'| {label} | {paper_mean*100:.1f} ± {paper_std*100:.1f} | ' + ' | '.join(values) + ' |')
    timing_rows = [f'| {names[b]} | {latency[b]["median_ms"]:.2f} ms | {latency[b]["p95_ms"]:.2f} ms |' for b in BACKENDS]
    categories = [f'| {row["category"]} | {row["test_images"]} | ' +
                  ' | '.join(f'{row["image_auroc"][b]*100:.2f}' for b in BACKENDS) + ' |'
                  for row in result['category_image_auroc']]
    parity = result['parity']['myriad_vs_cpu']
    support = result['foreground_output_support']
    execution = result['device_execution']
    self_scores = [x for row in inference['myriad']['categories'] for x in row['reference_self_scores']]
    text = f'''# Full MVTec AD: DINOv2-S + AnomalyDINO on the 2017 stick

**At 448 × 448, all 15 categories and all 1,725 test images run on Myriad 2.**
Three independent one-shot reference choices give **{quality['myriad']['mean']['image_auroc']*100:.2f}% mean image AUROC**,
compared with **{quality['cpu']['mean']['image_auroc']*100:.2f}% on the Mac** and **96.5% in the paper**.
The largest absolute mean difference from CPU across all six metrics is
**{max(abs(quality['myriad']['mean'][k]-quality['cpu']['mean'][k])*100 for k in METRICS):.3f} percentage points**.
The complete encoder, PCA mask, nearest-neighbor search and image score execute
on the stick. No test images were used for training or reference selection.

This is the full benchmark extension of the [224-square pilot](DINO_REPORT.md).
[Protocol and reproduction commands](DINO448_PROTOCOL.md) describe the pinned
sources, inference boundaries, solver approximation and metric definitions.

## Quality against the paper

Target: [Table 2, AnomalyDINO-S (448), one-shot, agnostic](https://arxiv.org/html/2405.14529v3#S4.T2).
All entries are percentages: mean ± population standard deviation of three
reference trials, after equal-weight category averaging within each trial.

| Metric | Paper | Stick | Mac CPU | Mac GPU |
|---|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

The Mac reference reproduces all six published mean values at the paper's
one-decimal precision. The device uses FP16 and replaces randomized SVD with
64 power iterations; its results are an implementation comparison rather than
a claim of bitwise equivalence. AUROC measures ranking quality. F1-max is an
oracle evaluation statistic over thresholds, not a selected deployment threshold.

## Single-reference latency

Measured on one Intel NCS1 / Myriad 2 and one Apple M2 Pro with 16 GiB RAM.
Each backend processes the same **150 batch-one requests**: ten evenly spaced
test IDs per category, the first reference and eight rotations, three warmups
per category. Quality inference and pixel evaluation have finished before timing.

| Backend | Median | p95 |
|---|---:|---:|
{chr(10).join(timing_rows)}

Prepared host tensor → returned 32 × 32 map and image score; includes conversion,
transfers, query masking and synchronization. Graph load, reference calibration,
image decoding/resizing and pixel-metric postprocessing are excluded. CPU uses
four PyTorch threads. The Metal baseline performs the model and matching on GPU
and the author's sklearn PCA on CPU. The stick calculates PCA in its fixed graph
even for categories where the resulting mask is disabled. Core ML and the
Apple Neural Engine are not benchmarked.

The Mac is {latency['myriad']['median_ms']/latency['cpu']['median_ms']:.1f}× faster on CPU and
{latency['myriad']['median_ms']/latency['mps']['median_ms']:.1f}× faster on Metal for these implementations.
Energy is excluded.

For quality evaluation, one encoder pass feeds three isolated one-shot heads:
**5,175 image/reference evaluations per backend**. Those joint timings in the
prediction archive are diagnostic throughput, not the single-head timings above.
On the matched latency requests, the maximum score difference between the
single-head graph and the first joint head is
**{max(v['score_max_error'] for v in result['single_head_vs_joint_first_head'].values()):.6f}** across the three backends.

## Per-category image AUROC

Mean of the three reference trials, in %. Small categories have the same weight
as larger categories in the final macro average.

| Category | Test images | Stick | CPU | GPU |
|---|---:|---:|---:|---:|
{chr(10).join(categories)}

The six metrics for every category/reference are in
[stick](results/dino448_myriad_metrics.json),
[CPU](results/dino448_cpu_metrics.json) and
[GPU](results/dino448_mps_metrics.json).

## Numerical drift and execution evidence

Against CPU, the stick's score mean absolute error is **{parity['score_mae']:.6f}**,
maximum **{parity['score_max_error']:.6f}**. Patch-distance mean absolute error is
**{parity['patch_distance_mae']:.6f}**, maximum **{parity['patch_distance_max_error']:.6f}**.
On the five masked categories, nonzero map support differs at
**{support['different_pixels']:,} / {support['compared_pixels']:,} positions ({support['disagreement_fraction']*100:.3f}%)**.
This combines solver differences, FP16 feature drift and zero rounding. The
separate same-input PCA checks isolate the solver in
[verification](results/dino448_verification.json).
Reference self-distances range from {min(self_scores):.6f} to {max(self_scores):.6f};
FP16 can yield a positive distance even for the reference image itself.

The [device profile](results/dino448_myriad.tsv) records
**{execution['executed_stages']} executed stages**, including
**{execution['executed_pca_stages']} PCA stages**, and **0 CPU kernel microseconds**.
The host bridge explicitly loads `MYRIAD` only. It has no fallback model.
The CPU still handles image preparation, tensor transport and metric evaluation.

At 1,025 tokens, our legacy runtime corrupted tail rows of constant-vector
broadcasts. Explicitly expanding the constants fixes this without changing the
model values; it grows the IR weight file to about 187 MB while retaining the
same 22.1M learned parameters. The
[broadcast reproducer](results/dino448_broadcast_verification.json) documents the
failure and correction. Firmware remains the stock RAM-loaded image.

[Independent source verification](results/dino448_reference_verification.json)
checks the explicit reference against Meta and the author's mask code.
The device verification also checks complete feature outputs and confirms that
the previous 224-square XML and weights remain unchanged. The streaming pixel
metric checks cover score ties and chunk boundaries against sklearn and direct
PRO thresholds. All original-resolution pixels are evaluated.

## Published evidence and scope

- [Assessment and graph/firmware hashes](results/dino448_assessment.json).
- [Per-image labels, three scores and joint timing for every backend](results/dino448_predictions.json).
- [Dataset revision and hashes of all selected files](results/dino448_dataset.json).
- Single-head latency requests: [stick](results/dino448_myriad_latency.json),
  [CPU](results/dino448_cpu_latency.json), [GPU](results/dino448_mps_latency.json).

No raw camera images, USB identifiers, private host metadata, pretrained weights
or generated graph binaries are published. Dataset/model terms are recorded in
[THIRD_PARTY.md](THIRD_PARTY.md). Reference choice, one physical device and this
specific runtime constrain the comparison; the reported standard deviations
cover reference variation only. The camera demo remains the 224-square pilot.

The historical result concerns **2017 hardware executing a later visual
Transformer and a 2024–2025 method**. It does not establish that their training
data, recipe or weights were available in 2017. The host runtime dates from 2020.
'''
    (ROOT / 'DINO448_REPORT.md').write_text(text)


def main():
    quality = {b: read(f'dino448_{b}_metrics.json') for b in BACKENDS}
    inference = {b: read(f'dino448_{b}_inference.json') for b in BACKENDS}
    latency = {b: read(f'dino448_{b}_latency.json') for b in BACKENDS}
    for b in BACKENDS:
        assert quality[b]['complete'] and quality[b]['test_images'] == 1725
        assert inference[b]['complete'] and sum(r['cases'] for r in inference[b]['categories']) == 1725
        assert {r['category'] for r in quality[b]['categories']} == set(CATEGORIES)
        assert len(latency[b]['rows']) == 150
        assert [r['id'] for r in latency[b]['rows']] == [r['id'] for r in latency['cpu']['rows']]
    predictions, parity, category_rows = [], {}, []
    score_error, patch_error = {b: [] for b in ['myriad', 'mps']}, {b: [] for b in ['myriad', 'mps']}
    for category in CATEGORIES:
        cached = {b: np.load(ROOT / f'assets/mvtec448/{b}/{category}.npz') for b in BACKENDS}
        ids = [r['id'] for r in cases(category)]
        for b, data in cached.items():
            assert data['ids'].tolist() == ids
            assert data['maps'].shape == (len(ids), 3, 32, 32)
            assert np.array_equal(data['gold'], cached['cpu']['gold'])
            assert np.isfinite(data['maps']).all() and np.isfinite(data['scores']).all()
            report = next(r for r in inference[b]['categories'] if r['category'] == category)
            assert sha256(ROOT / f'assets/mvtec448/{b}/{category}.npz') == report['cache_sha256']
        for b in score_error:
            score_error[b].append(np.abs(cached[b]['scores'] - cached['cpu']['scores']).ravel())
            patch_error[b].append(np.abs(cached[b]['maps'] - cached['cpu']['maps']).ravel())
        for i, identifier in enumerate(ids):
            predictions.append({'id': identifier, 'gold': int(cached['cpu']['gold'][i]),
                                'backends': {b: {'scores': cached[b]['scores'][i].tolist(),
                                                 'joint_three_head_ms': float(cached[b]['times'][i])} for b in BACKENDS}})
        category_metrics = {b: next(r for r in quality[b]['categories'] if r['category'] == category)
                            for b in BACKENDS}
        category_rows.append({'category': category, 'test_images': len(ids),
                              'image_auroc': {b: float(np.mean([r['image_auroc'] for r in category_metrics[b]['trials']]))
                                              for b in BACKENDS}})
    for b in score_error:
        s, p = np.concatenate(score_error[b]), np.concatenate(patch_error[b])
        parity[b + '_vs_cpu'] = {'score_mae': float(s.mean()), 'score_max_error': float(s.max()),
                                 'patch_distance_mae': float(p.mean()), 'patch_distance_max_error': float(p.max())}
    single_head_parity = {}
    for b in BACKENDS:
        times = [r['ms'] for r in latency[b]['rows']]
        assert np.isclose(np.median(times), latency[b]['median_ms'])
        expected = {r['id']: r['backends'][b]['scores'][0] for r in predictions}
        errors = [abs(row['score'] - expected[row['id']]) for row in latency[b]['rows']]
        single_head_parity[b] = {'score_mae': float(np.mean(errors)), 'score_max_error': float(np.max(errors))}
    with (ROOT / 'results/dino448_myriad.tsv').open() as stream:
        profile = list(csv.DictReader(stream, delimiter='\t'))
    active = [r for r in profile if r['status'] == '2']
    assert active and sum(int(r['cpu_us']) for r in profile) == 0
    assert all(any(prefix in r['layer'] for r in active) for prefix in ['layer11/', 'pca/', 'head0/'])
    graphs = {name: {suffix: sha256((GRAPH_DIR / '448' / name).with_suffix('.' + suffix)) for suffix in ['xml', 'bin']}
              for name in ['features', 'paper_1', 'paper_3']}
    masks = inference['myriad']['categories']
    support_diff = sum(r['mask_pixels_different_vs_cpu'] for r in masks)
    support_total = sum(r['mask_pixels_compared'] for r in masks)
    result = {'complete': True, 'resolution': [448, 448], 'categories': 15, 'test_images': 1725,
              'one_shot_trials': 3, 'quality': {b: {k: quality[b][k] for k in ['mean', 'std', 'macro_trials']} for b in BACKENDS},
              'single_head_latency': {b: {k: latency[b][k] for k in ['samples', 'median_ms', 'p95_ms', 'pca', 'timing']} for b in BACKENDS},
              'single_head_vs_joint_first_head': single_head_parity,
              'paper': {'source': 'https://arxiv.org/html/2405.14529v3#S4.T2', 'row': 'AnomalyDINO-S (448), one-shot, agnostic',
                        'mean_std': {k: list(v) for k, v in PAPER.items()}},
              'difference_from_paper_percentage_points': {b: {k: 100 * (quality[b]['mean'][k] - PAPER[k][0]) for k in PAPER} for b in BACKENDS},
              'category_image_auroc': category_rows, 'parity': parity,
              'foreground_output_support': {'different_pixels': support_diff, 'compared_pixels': support_total,
                    'disagreement_fraction': support_diff / support_total,
                    'definition': 'Nonzero patch-map support on five masked categories; combines PCA solver, FP16 backbone drift and zero rounding; not an isolated PCA-mask measurement'},
              'device_execution': {'backend': 'MYRIAD only', 'executed_stages': len(active), 'profile_cpu_us': 0,
                                   'executed_pca_stages': sum('pca/' in r['layer'] for r in active),
                                   'executed_head_stages': [r['layer'] for r in active if 'head0/' in r['layer']]},
              'graph_sha256': graphs, 'model': manifest(), 'hardware': {'stick': 'Intel NCS1 / Myriad 2 (MA2450)', 'mac': 'Apple M2 Pro, 16 GiB'},
              'verification': read('dino448_verification.json'), 'broadcast_verification': read('dino448_broadcast_verification.json'),
              'energy': 'excluded'}
    assert result['verification']['passed'] and result['broadcast_verification']['passed']
    firmware = ROOT / 'third_party/openvino-2020.3/bin/aarch64/Release/lib/usb-ma2450.mvcmd'
    result['firmware_sha256'] = sha256(firmware)
    (ROOT / 'results/dino448_assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    (ROOT / 'results/dino448_predictions.json').write_text(json.dumps({'trial_reference_indices': [0, 1, 2], 'rows': predictions}, indent=2) + '\n')
    dataset = json.loads((DATA_DIR / 'manifest_15cat.json').read_text())
    (ROOT / 'results/dino448_dataset.json').write_text(json.dumps(dataset, indent=2) + '\n')
    write_report(result, inference)
    print(json.dumps({k: result[k] for k in ['quality', 'single_head_latency', 'foreground_output_support']}, indent=2))


if __name__ == '__main__':
    main()
