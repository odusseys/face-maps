"""Generate canonical expressions and matching portraits as a new dataset."""
import json,os,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from facekit.paths import DATA
from facekit.bootstrap import assets
from nebius.expressions.render import expression_assets

def main():
    source=DATA/'datasets/faces1000-visibleface-centered';neutral=DATA/'datasets/faces1000-visibleface-uv-paired'
    out=DATA/'datasets/faces1000-canonical-expression-256';out.mkdir(parents=True,exist_ok=True)
    pilot=DATA/'runs/expression-canonical-pilot/samples'
    for i in range(6):
        src=pilot/f'{i:05d}';dst=out/'samples'/src.name
        if not (src/'render_complete.json').exists():raise ValueError('Canonical pilot is not complete')
        if not (dst/'render_complete.json').exists():
            dst.mkdir(parents=True,exist_ok=True)
            for name in ('portrait_expression_hair.png','portrait_expression_bare.png','raw_expression.npz','canonical_expression.npz','render_complete.json'):shutil.copy2(src/name,dst/name)
    blender,mpfb,asset_root=assets();units=expression_assets()
    missing=[i for i in range(1000) if not (out/'samples'/f'{i:05d}'/'render_complete.json').exists()]
    chunks=[missing[i:i+10] for i in range(0,len(missing),10)]
    env=os.environ.copy();env.update({'BLENDER_USER_RESOURCES':str(DATA/'blender_user'),'BLENDER_USER_EXTENSIONS':str(DATA/'blender_extensions'),
      'BLENDER_USER_CONFIG':str(DATA/'blender_config'),'BLENDER_USER_SCRIPTS':str(DATA/'blender_scripts'),'CUDA_CACHE_PATH':str(DATA/'cache/cuda')})
    def work(item):
        n,indices=item;cfg={'source':str(source),'neutral_source':str(neutral),'output':str(out),'assets':str(asset_root),
            'mpfb':str(mpfb),'units':str(units),'indices':indices,'canonical':True}
        path=out/f'worker-{n:03d}.json';path.write_text(json.dumps(cfg))
        with (out/f'worker-{n:03d}.log').open('w') as log:
            subprocess.run([str(blender),'-b','-t','3','--python-exit-code','1','--python',str(Path(__file__).with_name('blender_render.py')),'--',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('CANONICAL_RENDER_BATCH',n,'count',len(indices),flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(work,item) for item in enumerate(chunks)]
        for f in as_completed(futures):f.result()
    print('CANONICAL_RENDERS_READY',1000,flush=True)

if __name__=='__main__':main()
