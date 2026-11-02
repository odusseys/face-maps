#!/usr/bin/env python3
"""Verify and mark the moderate UV dataset as fixed-texture training ready."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from facekit.paths import DATA


def atomic_json(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    source=DATA/'datasets/faces500-moderate-centered'
    dataset=DATA/'datasets/faces500-moderate-uv-fixed-texture'
    manifest=json.loads((source/'portrait_manifest.json').read_text())
    config=json.loads((dataset/'dataset.json').read_text())
    state=json.loads((dataset/'dataset_complete.json').read_text())
    if not state.get('displacement_latents_ready'):raise ValueError('Target latents are incomplete')
    paths=sorted((dataset/'samples').glob('*/complete.json'))
    if len(paths)!=500:raise ValueError('Expected 500 samples')
    texture=Path(manifest['texture']).resolve(); digest=hashlib.sha256(texture.read_bytes()).hexdigest()
    if digest!=manifest['fixed_texture_sha256']:raise ValueError('Texture checksum mismatch')
    splits=collections.Counter()
    for p in paths:
        meta=json.loads(p.read_text());splits[meta['split']]+=1
        if Path(meta['texture']).resolve()!=texture or meta.get('fixed_texture_sha256')!=digest:
            raise ValueError(f'Inconsistent texture: {p.parent.name}')
        if not (dataset/'latents_displacement'/(p.parent.name+'.pt')).exists():raise ValueError(f'Missing target latent {p.parent.name}')
    if dict(splits)!={'train':400,'holdout':100}:raise ValueError(f'Bad splits: {splits}')
    preview=DATA/'previews/moderate-shapes-20261101/samples'
    for i in range(20):
        name=f'{i:05d}'
        a=np.load(preview/name/'geometry.npz')['displacement_m']
        b=np.load(source/'samples'/name/'geometry.npz')['displacement_m']
        if not np.array_equal(a,b):raise ValueError(f'Preview geometry changed: {name}')
    config.update({'fixed_texture_path':str(texture),'fixed_texture_sha256':digest,
                   'shape_distribution':'moderate variation preview, random hair',
                   'preview_geometry_verified':20})
    state.update({'fixed_texture_path':str(texture),'fixed_texture_sha256':digest,
                  'preview_geometry_verified':20})
    atomic_json(dataset/'dataset.json',config);atomic_json(dataset/'dataset_complete.json',state)
    print('MODERATE_DATASET_READY',json.dumps({'path':str(dataset),'count':len(paths),
          'splits':dict(splits),'one_texture':True,'preview_geometry_verified':20,
          'bound_mm':config['bound_mm']}),flush=True)


if __name__=='__main__':main()
