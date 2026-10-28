"""The paper's 448-square, one-shot agnostic protocol and evaluation inputs."""

import cv2
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

from experiments.exp03_anomalydino.model import DATA_DIR, DinoReference, prepare_image
from experiments.exp03_anomalydino.scripts.fetch_mvtec import CATEGORIES

MASKED = {"capsule", "hazelnut", "pill", "screw", "toothbrush"}
ANGLES = range(0, 360, 45)


def cases(category):
    root = DATA_DIR / category
    return [{"id": str(p.relative_to(DATA_DIR)), "path": p,
             "gold": int(p.parent.name != "good"),
             "mask": root / "ground_truth" / p.parent.name / (p.stem + "_mask.png")}
            for p in sorted((root / "test").glob("*/*.png"))]


def rotated_inputs(path):
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    center = tuple(np.asarray(image.shape[1::-1]) / 2)
    from PIL import Image
    for angle in ANGLES:
        transform = cv2.getRotationMatrix2D(center, angle, 1.)
        rotated = cv2.warpAffine(image, transform, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_DEFAULT)
        yield prepare_image(Image.fromarray(rotated), 448)[0]


def paper_mask(raw, enabled):
    if not enabled:
        return np.ones(1024, dtype=bool)
    pc = PCA(n_components=1, svd_solver="randomized", random_state=0).fit_transform(raw.astype(np.float32))
    mask = pc > 10
    center = mask.reshape(32, 32)[6:25, 6:25]
    if center.sum() <= center.size * .35:
        mask = -pc > 10
    # Preserve the pinned author's flattened N-by-1 morphology layout.
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool).ravel()


class MacPaper:
    def __init__(self, device):
        self.reference = DinoReference(device, size=448)
        self.banks = []

    def calibrate(self, category, trials=3):
        self.banks = []
        for seed in range(trials):
            path = DATA_DIR / category / "train/good" / f"{seed:03}.png"
            bank = torch.cat([self.reference.features(pixels) for pixels in rotated_inputs(path)], dim=1)
            self.banks.append(bank)

    @torch.inference_mode()
    def infer(self, pixels, masked, trials=3):
        raw = self.reference.features(pixels, normalize=False)
        mask = paper_mask(raw[0].cpu().numpy(), masked)
        mask_tensor = torch.as_tensor(mask, device=raw.device)
        query = F.normalize(raw, dim=-1)
        results = []
        for bank in self.banks[:trials]:
            distance = (1 - query @ bank.transpose(-1, -2)).clamp_min(0).min(-1).values
            distance = distance * mask_tensor
            score = distance.topk(10, dim=-1).values.mean(-1, keepdim=True)
            results.append(torch.cat([distance, score], dim=-1))
        return torch.cat(results).cpu().numpy()


class MyriadPaper:
    """Host image preparation and transport; encoder, PCA and scoring are on MYRIAD."""
    def __init__(self, trials=3):
        from experiments.exp03_anomalydino.model import ROOT, GRAPH_DIR
        self.graph_dir = GRAPH_DIR / '448'
        self.cache = ROOT / 'assets/mvtec448/myriad_reference'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.trials = trials
        self.model = None
        self.self_scores = []

    def calibrate(self, category):
        import json
        from movidius_lab.paths import sha256
        from movidius_lab.myriad import Myriad
        self.close()
        path = self.cache / (category + '.npz')
        metadata = path.with_suffix('.json')
        graph = self.graph_dir / 'features'
        fingerprint = {'xml': sha256(graph.with_suffix('.xml')), 'bin': sha256(graph.with_suffix('.bin')),
                       'references': [sha256(DATA_DIR / category / 'train/good' / f'{seed:03}.png') for seed in range(3)]}
        if path.exists() and metadata.exists() and json.loads(metadata.read_text()) == fingerprint:
            banks = np.load(path)['banks']
        else:
            banks = []
            with Myriad(graph) as encoder:
                for seed in range(3):
                    features = []
                    for pixels in rotated_inputs(DATA_DIR / category / 'train/good' / f'{seed:03}.png'):
                        result = encoder.infer_tensors({'pixels': pixels.astype(np.float16)})
                        if not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result, axis=-1), 1., atol=.01):
                            raise ValueError('Invalid reference features from MYRIAD')
                        features.append(result.astype(np.float16))
                    banks.append(np.concatenate(features, axis=1))
                    print('myriad', category, f'reference {seed + 1}/3 complete', flush=True)
            banks = np.asarray(banks)
            np.savez_compressed(path, banks=banks)
            metadata.write_text(json.dumps(fingerprint, indent=2) + '\n')
        self.model = Myriad(self.graph_dir / f'paper_{self.trials}')
        for seed in range(self.trials):
            self.model.set_input(f'reference_bank_{seed}', banks[seed])
        self.model.set_input('mask_enabled', np.full((1, 1, 1), int(category in MASKED), dtype=np.float16))
        self.self_scores = []
        for seed in range(self.trials):
            pixels, _ = prepare_image(DATA_DIR / category / 'train/good' / f'{seed:03}.png', 448)
            score = float(self.infer(pixels, category in MASKED)[seed, -1])
            if not 0 <= score < .01:
                raise ValueError(f'Reference self-distance failed: {category} trial {seed}: {score}')
            self.self_scores.append(score)

    def infer(self, pixels, masked):
        self.model.set_input('pixels', pixels.astype(np.float16))
        return self.model.execute()

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None
