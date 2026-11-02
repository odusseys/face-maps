#!/usr/bin/env python3
"""Render a separate reproducible preview of 20 more varied MakeHuman faces."""
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from facekit.paths import DATA
from facekit.bootstrap import assets
from nebius.build_uv_aligned_dataset import align


def sample_diversity(reference, ids, paths):
    features=[]
    for path in paths:
        delta=np.load(path)['displacement_m']
        features.append(align(reference, reference+delta, ids)[0][ids])
    a=np.stack(features)
    return np.array([np.linalg.norm((a[i]-a[j])*1000,axis=1).mean()
                     for i in range(len(a)) for j in range(i)])


def main():
    output=DATA/'previews/moderate-shapes-20261101'
    original=DATA/'datasets/faces500-centered'
    output.mkdir(parents=True,exist_ok=True)
    if not (output/'reference.npz').exists():shutil.copy2(original/'reference.npz',output/'reference.npz')
    blender,mpfb,asset_root=assets()
    code=Path(__file__).resolve().parent/'blender_moderate_shapes.py'
    env=os.environ.copy()
    env.update({'BLENDER_USER_RESOURCES':str(DATA/'blender_user'), 'BLENDER_USER_EXTENSIONS':str(DATA/'blender_extensions'),
                'CUDA_CACHE_PATH':str(DATA/'cache/cuda'), 'BLENDER_USER_CONFIG':str(DATA/'blender_config'),
                'BLENDER_USER_SCRIPTS':str(DATA/'blender_scripts')})
    def worker(index):
        conf={'output':str(output),'assets':str(asset_root),'mpfb':str(mpfb), 'seed':202611011,
              'device':'CUDA','samples':20,'threads':4,'indices':list(range(index*10,(index+1)*10))}
        path=output/f'worker-{index}.json';path.write_text(json.dumps(conf,indent=2))
        log=output/f'worker-{index}.log'
        with log.open('w') as f:
            subprocess.run([str(blender),'-b','-t','4','--python-exit-code','1','--python',str(code),'--',str(path)],
                           env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('WORKER_COMPLETE',index,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker,range(2)))
    paths=[output/'samples'/f'{i:05d}' for i in range(20)]
    if any(not (p/'render_complete.json').is_file() for p in paths):raise ValueError('Missing rendered face')
    tile=320; header=28
    canvas=Image.new('RGB',(tile*5,(tile+header)*4),(28,28,28));draw=ImageDraw.Draw(canvas)
    for i,path in enumerate(paths):
        x=(i%5)*tile;y=(i//5)*(tile+header)
        face=Image.open(path/'portrait_bare.png').convert('RGBA')
        background=Image.new('RGBA',face.size,(48,48,48,255)); background.alpha_composite(face)
        canvas.paste(background.convert('RGB').resize((tile,tile),Image.Resampling.LANCZOS),(x,y+header))
        draw.text((x+8,y+6),f'{i+1:02}',fill=(255,255,255))
    sheet=output/'moderate-shapes-contact-sheet.png';canvas.save(sheet)
    ref=np.load(output/'reference.npz'); vertices=ref['vertices_m'];ids=np.unique(ref['triangle_vertices'])
    wide=sample_diversity(vertices,ids,[p/'geometry.npz' for p in paths])
    old=list(sorted((original/'samples').glob('*/geometry.npz')))
    rng=np.random.default_rng(778);baseline=[]
    for _ in range(10):
        baseline.extend(sample_diversity(vertices,ids,[old[i] for i in rng.choice(len(old),20,replace=False)]))
    report={'seed':202611011,'count':20,'presentation':'bald, same skin and lighting',
            'metric':'mean per-vertex Euclidean distance after UV-only similarity alignment, in mm',
            'preview_pairwise_median_mm':float(np.median(wide)),
            'existing_pairwise_median_mm':float(np.median(baseline)),
            'preview_pairwise_mean_mm':float(np.mean(wide)), 'existing_pairwise_mean_mm':float(np.mean(baseline)),
            'comparison_baseline_random_groups':10,'sheet':str(sheet)}
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print('PREVIEW_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
