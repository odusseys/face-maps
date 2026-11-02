#!/usr/bin/env python3
"""Build UV-only similarity-aligned face targets with a train-only UV quantile."""
import argparse
import json
import shutil
from functools import lru_cache
from pathlib import Path
from PIL import Image

import numpy as np
import torch

from facekit.paths import DATA, ROOT
from facekit.uv import ATLAS, encode_displacement, panel

ALIGNMENT = 'rigid alignment plus optimal uniform scale to reference face'


def cached_map_renderer(mapping):
    """Cache repeated skin panels; displacement interpolation matches make_maps."""
    index,weights,mask,bbox=mapping
    sample_indices=index[mask]; sample_weights=weights[mask]

    @lru_cache(maxsize=32)
    def texture_panel(texture):
        image=Image.open(texture).convert('RGB').resize((ATLAS,ATLAS),Image.Resampling.LANCZOS)
        return panel(image,mask,bbox)

    def render(texture,displacement_m,bound_mm):
        field=np.zeros((ATLAS,ATLAS,3),dtype=np.float32)
        field[mask]=np.einsum('pi,pic->pc',sample_weights,displacement_m[sample_indices])*1000
        encoded=encode_displacement(field,bound_mm,'signed_linear')
        tex=texture_panel(str(texture)); disp=panel(Image.fromarray(encoded),mask,bbox)
        pair=Image.new('RGB',(1024,512));pair.paste(tex,(0,0));pair.paste(disp,(512,0))
        return tex,disp,pair,float(np.mean(np.abs(field[mask])>bound_mm))
    return render


def align(reference, vertices, ids):
    moving_center = vertices[ids].mean(axis=0)
    reference_center = reference[ids].mean(axis=0)
    moving = vertices - moving_center
    target = reference[ids] - reference_center
    u, _, vt = np.linalg.svd(moving[ids].T @ target)
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    rotated = moving[ids] @ rotation
    scale = float(np.sum(rotated * target) / np.sum(moving[ids] ** 2))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid similarity scale')
    delta = (scale * (moving @ rotation) + reference_center - reference).astype(np.float32)
    return delta, rotation, moving_center, reference_center, scale


def atomic_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def validate_alignment(reference, vertices, ids):
    expected = align(reference, vertices, ids)[0]
    # Rigid transforms and global scale must not change aligned face shape.
    angle = .31
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    transformed = 1.7 * (vertices @ rotation) + np.array([.2, -.3, .1])
    actual = align(reference, transformed, ids)[0]
    if not np.allclose(expected[ids], actual[ids], atol=1e-7, rtol=1e-5):
        raise AssertionError('Similarity alignment is not transform-invariant')
    outside = np.ones(len(vertices), dtype=bool)
    outside[ids] = False
    modified = vertices.copy()
    modified[outside] += np.array([.7, -.5, .3])
    actual = align(reference, modified, ids)[0]
    if not np.array_equal(expected[ids], actual[ids]):
        raise AssertionError('Unmapped vertices affected face alignment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DATA/'datasets/faces1000-visibleface-centered')
    parser.add_argument('--output', type=Path, default=DATA/'datasets/faces1000-visibleface-uv-paired')
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if not source.is_relative_to(ROOT.resolve()) or not out.is_relative_to(ROOT.resolve()) or source == out:
        parser.error('Source and destination must be distinct paths under FACE_SHARED_ROOT')
    reference_data = np.load(source/'reference.npz')
    reference = reference_data['vertices_m'].astype(np.float64)
    ids = np.unique(reference_data['triangle_vertices'])
    if not reference_data['face_mask'][ids].all():
        raise ValueError('UV triangles include vertices outside face mask')
    uv = np.load(source/'uv_mapping.npz')
    mapping = tuple(uv[k] for k in ('indices', 'weights', 'mask', 'bbox'))
    render_maps = cached_map_renderer(mapping)
    indices, weights = mapping[0][mapping[2]], mapping[1][mapping[2]]
    records = []
    for meta_path in sorted((source/'samples').glob('*/complete.json')):
        meta = json.loads(meta_path.read_text())
        with np.load(meta_path.parent/'geometry.npz') as geom:
            vertices = reference + geom['displacement_m'].astype(np.float64)
        if len(records) < 10:
            validate_alignment(reference, vertices, ids)
        delta, rotation, center, ref_center, scale = align(reference, vertices, ids)
        records.append((meta_path, meta, delta, rotation, center, ref_center, scale))
    train = [r for r in records if r[1]['split'] == 'train']
    holdout = [r for r in records if r[1]['split'] == 'holdout']
    if (len(records), len(train), len(holdout)) != (1000, 900, 100):
        raise ValueError(f'Unexpected split counts: {len(records)}, {len(train)}, {len(holdout)}')
    # Identical interpolation and float32 arithmetic to facekit.uv.make_maps.
    count = len(train) * len(indices) * 3
    values = np.empty(count, dtype=np.float32)
    width = len(indices) * 3
    for i, record in enumerate(train):
        values[i*width:(i+1)*width] = np.abs(
            np.einsum('pi,pic->pc', weights, record[2][indices]) * 1000).reshape(-1)
    pos = .97 * (count - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    values.partition((lo, hi))
    bound = float(values[lo] + (pos-lo) * (values[hi]-values[lo]))
    del values
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError('Invalid UV quantile')
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((source/'dataset.json').read_text())
    config.update({'bound_mm': bound, 'encoding': 'signed_linear', 'normalization_quantile': .97,
                   'normalization_source': str(source),
                   'normalization_population': 'training split interpolated face UV pixel XYZ components after UV-only similarity alignment',
                   'normalization_components': count, 'alignment_vertex_count': len(ids),
                   'alignment_vertex_selection': 'unique vertices of reference face UV triangles',
                   'geometry_alignment': ALIGNMENT, 'latent_modes': ['displacement_only'],
                   'count': len(records), 'smoke_only': False})
    config_path = out/'dataset.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        for key in ('normalization_source','bound_mm','alignment_vertex_count','normalization_components'):
            if previous.get(key) != config[key]:
                raise ValueError(f'Existing output has different {key}')
        config = previous
    else:
        atomic_json(config_path, config)
    print('BOUND', json.dumps({'bound_mm': bound, 'train': len(train), 'holdout': len(holdout),
          'uv_pixels': len(indices), 'components': count, 'alignment_vertices': len(ids)}), flush=True)
    for name in ('reference.npz','reference.blend','reference_skin.txt','uv_mapping.npz'):
        if not (out/name).exists():
            shutil.copy2(source/name, out/name)
    for index, (meta_path, meta, delta, rotation, center, ref_center, scale) in enumerate(records, 1):
        src, dst = meta_path.parent, out/'samples'/meta_path.parent.name
        dst.mkdir(parents=True, exist_ok=True)
        if (dst/'complete.json').exists():
            continue
        for name in ('portrait_hair.png','portrait_bare.png','render_complete.json'):
            shutil.copy2(src/name, dst/name)
        # Source vertices here are already rigidly aligned; this second fit is
        # mathematically equivalent to fitting the original mesh directly.
        np.savez_compressed(dst/'geometry.npz', displacement_m=delta, rotation=rotation,
                            moving_center=center, reference_center=ref_center,
                            similarity_scale=np.float64(scale))
        tex, disp, maps, clipped = render_maps(meta['texture'], delta, bound)
        tex.save(dst/'texture.png'); disp.save(dst/'displacement.png'); maps.save(dst/'maps.png')
        previous = torch.load(src/'latents.pt', map_location='cpu', weights_only=True)
        torch.save({k: previous[k] for k in ('face_hair','face_bare')}, dst/'latents.pt')
        meta.update({'clipped_coordinate_fraction': clipped, 'displacement_encoding': 'signed_linear',
                     'bound_mm': bound, 'similarity_scale': scale,
                     'geometry_alignment': 'UV-mapped-vertex similarity', 'paired_map_latent_available': False})
        atomic_json(dst/'complete.json', meta)
        if index % 25 == 0 or index == len(records):
            print('MAPS', index, len(records), flush=True)
    state_path = out/'dataset_complete.json'
    if not state_path.exists():
        atomic_json(state_path, {'count': len(records), 'bound_mm': bound, 'encoding': 'signed_linear',
                                 'displacement_latents_ready': False, 'paired_map_latents_ready': False})
    print('DATASET_READY', out, flush=True)


if __name__ == '__main__':
    main()
