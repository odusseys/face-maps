"""Capture procedural neutral and expressive eyes for true holdout previews."""
import json,os,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from facekit.paths import DATA
from facekit.bootstrap import assets
from nebius.expressions.render import expression_assets

def main():
    source=DATA/'datasets/faces1000-visibleface-centered';neutral=DATA/'datasets/faces1000-visibleface-uv-paired'
    canonical=DATA/'datasets/faces1000-canonical-expression-256';dataset=DATA/'datasets/faces1000-canonical-head-256'
    root=DATA/'runs/head-holdout-eyes';root.mkdir(parents=True,exist_ok=True);blender,mpfb,asset_root=assets();units=expression_assets()
    indices=[int(p.parent.name) for p in sorted((canonical/'samples').glob('*/complete.json')) if json.loads(p.read_text())['split']=='holdout']
    if len(indices)!=100:raise ValueError('Expected 100 holdout identities')
    env=os.environ.copy();env.update({'BLENDER_USER_RESOURCES':str(DATA/'blender_user'),'BLENDER_USER_EXTENSIONS':str(DATA/'blender_extensions'),
      'BLENDER_USER_CONFIG':str(DATA/'blender_config'),'BLENDER_USER_SCRIPTS':str(DATA/'blender_scripts'),'CUDA_CACHE_PATH':str(DATA/'cache/cuda')})
    chunks=[indices[i:i+10] for i in range(0,len(indices),10)];tasks=[(kind,i,chunk) for kind in ('neutral','expression') for i,chunk in enumerate(chunks)]
    def run(item):
        kind,n,chunk=item;output=root/kind;output.mkdir(parents=True,exist_ok=True)
        cfg={'source':str(source),'neutral_source':str(neutral),'output':str(output),'assets':str(asset_root),'mpfb':str(mpfb),'units':str(units),
          'indices':chunk,'canonical':True,'capture_eyes':True,'capture_eyes_only':True}
        if kind=='neutral':cfg.update({'neutral_only':True,'expression_override':{}})
        file=root/f'{kind}-{n:02d}.json';file.write_text(json.dumps(cfg))
        with (root/f'{kind}-{n:02d}.log').open('w') as log:
            subprocess.run([str(blender),'-b','-t','3','--python-exit-code','1','--python',str(Path(__file__).with_name('blender_render.py')),'--',str(file)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('EYES_CAPTURED',kind,n,len(chunk),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(run,item) for item in tasks]):f.result()
    for index in indices:
        ident=f'{index:05d}';dst=dataset/'samples'/ident;dst.mkdir(parents=True,exist_ok=True)
        with np.load(source/'samples'/ident/'geometry.npz') as a,np.load(neutral/'samples'/ident/'geometry.npz') as b:
            for kind in ('neutral','expression'):
                world=np.load(root/kind/'samples'/ident/'eyes_world.npy')
                aligned=(((world-a['moving_center'])@a['rotation']+a['reference_center']-b['moving_center'])@b['rotation'])*float(b['similarity_scale'])+b['reference_center']
                if aligned.shape!=(1064,3) or not np.isfinite(aligned).all():raise ValueError(f'Invalid eyes: {ident}')
                np.save(dst/('gt_'+kind+'_eyes.npy'),aligned.astype(np.float32))
    (dataset/'holdout_eyes_complete.json').write_text(json.dumps({'count':len(indices)}));print('HOLDOUT_EYES_READY',len(indices),flush=True)

if __name__=='__main__':main()
