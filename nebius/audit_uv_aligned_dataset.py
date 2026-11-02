#!/usr/bin/env python3
"""Compare encoded UV targets and clipping on identical Nebius-generated faces."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from facekit.paths import DATA
from facekit.uv import decode_displacement, inverse_panel


def summarize(values):
    return {'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'p95': float(np.quantile(values, .95)), 'max': float(np.max(values))}


def audit(path):
    config = json.loads((path/'dataset.json').read_text())
    uv = np.load(path/'uv_mapping.npz')
    mask = uv['mask']; indices = uv['indices'][mask]; weights = uv['weights'][mask]
    clips = {'train': [], 'holdout': []}
    errors = []
    for meta_path in sorted((path/'samples').glob('*/complete.json')):
        meta = json.loads(meta_path.read_text())
        delta = np.load(meta_path.parent/'geometry.npz')['displacement_m']
        field = np.einsum('pi,pic->pc', weights, delta[indices])
        fraction = float(np.mean(np.abs(field * 1000) > config['bound_mm']))
        clips[meta['split']].append(fraction)
        if meta['split'] == 'holdout':
            panel = Image.open(meta_path.parent/'displacement.png').convert('RGB')
            decoded = decode_displacement(inverse_panel(panel, uv['bbox']), config['bound_mm'], config['encoding'])[mask]
            errors.append(float(np.mean(np.linalg.norm((decoded - field) * 1000, axis=1))))
    if (len(clips['train']), len(clips['holdout'])) != (400, 100):
        raise ValueError('Dataset has unexpected split sizes')
    return {'dataset': str(path), 'bound_mm': config['bound_mm'],
            'train_clipped_fraction': summarize(clips['train']),
            'holdout_clipped_fraction': summarize(clips['holdout']),
            'holdout_mean_uv_vector_reconstruction_error_mm': summarize(errors)}


def main():
    root = DATA/'datasets'
    report = {'method': 'Decode saved 512x512 displacement panels back into the UV atlas; compare each UV pixel to its own aligned geometric target. Inverse resizing and 8-bit quantization are included.',
              'original': audit(root/'faces500-similarity-linear-q97'),
              'corrected': audit(root/'faces500-uv-aligned-uvq97')}
    path = DATA/'audits/nebius-uv-alignment-comparison.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
