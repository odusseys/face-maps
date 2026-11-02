#!/usr/bin/env python3
"""Generate 1000 moderately varied faces with diverse skin and hair."""
import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from facekit.paths import DATA
from facekit.bootstrap import assets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=1000)
    args=parser.parse_args()
    if not 1<=args.limit<=1000:parser.error('limit must be between 1 and 1000')
    source=DATA/'datasets/faces500-centered'
    output=DATA/'datasets/faces1000-visibleface-centered'
    output.mkdir(parents=True,exist_ok=True)
    if not (output/'reference.npz').exists():shutil.copy2(source/'reference.npz',output/'reference.npz')
    blender,mpfb,asset_root=assets()
    script=Path(__file__).resolve().parent/'blender_diverse_paired.py'
    env=os.environ.copy()
    env.update({'BLENDER_USER_RESOURCES':str(DATA/'blender_user'), 'BLENDER_USER_EXTENSIONS':str(DATA/'blender_extensions'),
                'CUDA_CACHE_PATH':str(DATA/'cache/cuda'), 'BLENDER_USER_CONFIG':str(DATA/'blender_config'),
                'BLENDER_USER_SCRIPTS':str(DATA/'blender_scripts')})
    seed=202611012
    missing=[i for i in range(args.limit) if not (output/'samples'/f'{i:05d}'/'render_complete.json').exists()]
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
        futures=[pool.submit(run,item) for item in enumerate(chunks)]
        try:
            for future in as_completed(futures):future.result()
        except Exception:
            for future in futures:future.cancel()
            raise
    count=sum((output/'samples'/f'{i:05d}'/'render_complete.json').exists() for i in range(args.limit))
    if count!=args.limit:raise RuntimeError(f'Expected {args.limit} complete renders, got {count}')
    print('RENDERS_COMPLETE',count,flush=True)


if __name__=='__main__':main()
