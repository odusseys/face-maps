#!/usr/bin/env python3
"""Generate 500 moderately varied faces with one fixed skin material."""
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from facekit.paths import DATA
from facekit.bootstrap import assets


def main():
    source=DATA/'datasets/faces500-centered'
    output=DATA/'datasets/faces500-moderate-centered'
    output.mkdir(parents=True,exist_ok=True)
    if not (output/'reference.npz').exists():shutil.copy2(source/'reference.npz',output/'reference.npz')
    blender,mpfb,asset_root=assets()
    script=Path(__file__).resolve().parent/'blender_moderate_training.py'
    env=os.environ.copy()
    env.update({'BLENDER_USER_RESOURCES':str(DATA/'blender_user'), 'BLENDER_USER_EXTENSIONS':str(DATA/'blender_extensions'),
                'CUDA_CACHE_PATH':str(DATA/'cache/cuda'), 'BLENDER_USER_CONFIG':str(DATA/'blender_config'),
                'BLENDER_USER_SCRIPTS':str(DATA/'blender_scripts')})
    seed=202611011
    missing=[i for i in range(500) if not (output/'samples'/f'{i:05d}'/'render_complete.json').exists()]
    chunks=[missing[i:i+10] for i in range(0,len(missing),10)]
    def run(item):
        number,indices=item
        config={'output':str(output),'assets':str(asset_root),'mpfb':str(mpfb), 'seed':seed,
                'device':'CUDA','samples':20,'threads':3,'indices':indices}
        path=output/f'worker-{number:03d}.json';path.write_text(json.dumps(config,indent=2))
        with (output/f'worker-{number:03d}.log').open('w') as f:
            subprocess.run([str(blender),'-b','-t','3','--python-exit-code','1','--python',str(script),'--',str(path)],
                           env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('RENDERED_BATCH',number,len(indices),'last',indices[-1],flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run,enumerate(chunks)))
    count=sum((output/'samples'/f'{i:05d}'/'render_complete.json').exists() for i in range(500))
    if count!=500:raise RuntimeError(f'Expected 500 complete renders, got {count}')
    print('RENDERS_COMPLETE',count,flush=True)


if __name__=='__main__':main()
