"""Resume 256px expression renders of the existing 1,000 identities on Nebius."""
import argparse,hashlib,json,os,subprocess,urllib.request,zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from facekit.paths import DATA
from facekit.bootstrap import assets

PACK_SHA256='d113107bd7eb59f3af4df6fc0ec29bfcc593f496d0b336aec14f086a80ce7146'

def expression_assets():
    archive=DATA/'assets/faceunits01.zip';root=DATA/'assets/faceunits01'
    if not archive.exists():
        tmp=archive.with_suffix('.zip.tmp')
        urllib.request.urlretrieve('https://files2.makehumancommunity.org/functional/faceunits01.zip',tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest()!=PACK_SHA256:raise ValueError('Unexpected Face Units archive hash')
        tmp.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=PACK_SHA256:raise ValueError('Face Units archive hash mismatch')
    units=root/'targets/faceunits'
    if not units.is_dir():
        root.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if not (root/info.filename).resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe archive path')
            z.extractall(root)
    return units

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=1000);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if not 1<=args.limit<=1000 or not 1<=args.workers<=8:raise ValueError('invalid limit/workers')
    source=DATA/'datasets/faces1000-visibleface-centered'; out=DATA/'datasets/faces1000-shape-expression-256'
    out.mkdir(parents=True,exist_ok=True);blender,mpfb,asset_root=assets()
    units=expression_assets()
    missing=[i for i in range(args.limit) if not (out/'samples'/f'{i:05d}'/'render_complete.json').exists()]
    chunks=[missing[i:i+10] for i in range(0,len(missing),10)]
    env=os.environ.copy();env.update({'BLENDER_USER_RESOURCES':str(DATA/'blender_user'),
        'BLENDER_USER_EXTENSIONS':str(DATA/'blender_extensions'),'BLENDER_USER_CONFIG':str(DATA/'blender_config'),
        'BLENDER_USER_SCRIPTS':str(DATA/'blender_scripts'),'CUDA_CACHE_PATH':str(DATA/'cache/cuda')})
    def work(item):
        n,indices=item; cfg={'source':str(source),'output':str(out),'assets':str(asset_root),'mpfb':str(mpfb),'units':str(units),'indices':indices}
        path=out/f'worker-{n:03d}.json';path.write_text(json.dumps(cfg))
        with (out/f'worker-{n:03d}.log').open('w') as log:
            subprocess.run([str(blender),'-b','-t','3','--python-exit-code','1','--python',str(Path(__file__).with_name('blender_render.py')),'--',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('RENDER_BATCH',n,'completed',len(indices),flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(work,item) for item in enumerate(chunks)]
        for f in as_completed(futures):f.result()
    print('RENDERS_READY',args.limit,flush=True)

if __name__=='__main__':main()
