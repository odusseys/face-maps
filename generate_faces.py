#!/usr/bin/env python3
"""Generate a reproducible 1,000-face MakeHuman/FLUX.2 dataset on shared storage."""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from facekit.paths import ROOT, DATA, MODEL, MODEL_ID, MODEL_REVISION, BOUND_MM
os.environ.setdefault('HF_HOME',str(ROOT/'face_maps'/'cache'/'huggingface'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'face_maps'/'cache'))
from facekit.bootstrap import assets, checkpoint


def run_blender(blender, config, output, label):
    path=output/f'worker_{label}.json'; path.write_text(json.dumps(config))
    env=os.environ.copy(); env['BLENDER_USER_RESOURCES']=str(DATA/'blender_user')
    env['BLENDER_USER_EXTENSIONS']=str(DATA/'blender_extensions')
    env['CUDA_CACHE_PATH']=str(DATA/'cache'/'cuda')
    env['BLENDER_USER_CONFIG']=str(DATA/'blender_config')
    env['BLENDER_USER_SCRIPTS']=str(DATA/'blender_scripts')
    command=[str(blender),'-b','-t',str(config['threads']),'--python-exit-code','1',
             '--python',str(Path(__file__).parent/'facekit'/'blender_generate.py'),'--',str(path)]
    subprocess.run(command,env=env,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DATA/'datasets'/'faces1000')
    p.add_argument('--count',type=int,default=1000); p.add_argument('--seed',type=int,default=20261031)
    p.add_argument('--holdout',type=int,default=100); p.add_argument('--render-workers',type=int,default=1)
    p.add_argument('--render-threads',type=int,default=6); p.add_argument('--render-samples',type=int,default=20)
    p.add_argument('--render-chunk-size',type=int,default=20,help='Restart Blender after this many faces to bound memory')
    p.add_argument('--device',choices=['CUDA','CPU'],default='CUDA')
    p.add_argument('--bound-mm',type=float,default=BOUND_MM)
    args=p.parse_args(); out=args.output.resolve()
    if not out.is_relative_to(ROOT.resolve()): p.error('output must be on the shared FACE_SHARED_ROOT mount')
    if args.count<1 or not 0<=args.holdout<args.count or args.bound_mm<=0 or args.render_workers<1 or args.render_chunk_size<1: p.error('invalid count, holdout, bound, or workers')
    print('Loading numerical libraries and FLUX.2 VAE support',flush=True)
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLFlux2
    from facekit.uv import raster_reference,make_maps
    from facekit.latent import vae_encode
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for FLUX.2 VAE encoding')
    out.mkdir(parents=True,exist_ok=True)
    settings={'seed':args.seed,'count':args.count,'holdout':args.holdout,'bound_mm':args.bound_mm,
              'model':MODEL_ID,'model_revision':MODEL_REVISION,'model_path':str(MODEL),
              'encoding':'signed_log1p_9','map_size':[1024,512], 'portrait_size':[512,512],
              'render_samples':args.render_samples,'render_device':args.device,
              'camera':'head-relative-hair-bounds-v2',
              'displacement_axes':'XYZ in rigidly aligned reference coordinates',
              'interpolation':'barycentric on reference UV triangles', 'units':'metres'}
    config_path=out/'dataset.json'
    if config_path.exists() and json.loads(config_path.read_text())!=settings:
        raise ValueError('Existing dataset configuration differs; choose a new output directory')
    config_path.write_text(json.dumps(settings,indent=2))
    print('Preparing assets on shared storage',flush=True)
    blender,mpfb,asset_root=assets()
    config={'output':str(out),'assets':str(asset_root),'mpfb':str(mpfb),
            'seed':args.seed,'device':args.device,'samples':args.render_samples,'threads':args.render_threads}
    if not all((out/name).exists() for name in ('reference.npz','reference.blend','reference_skin.txt')):
        run_blender(blender,{**config,'reference':True},out,'reference')
    missing=[i for i in range(args.count) if not (out/'samples'/f'{i:05d}'/'render_complete.json').is_file()]
    chunks=[missing[i:i+args.render_chunk_size] for i in range(0,len(missing),args.render_chunk_size)]
    with ThreadPoolExecutor(max_workers=args.render_workers) as pool:
        futures=[pool.submit(run_blender,blender,{**config,'indices':ids},out,i) for i,ids in enumerate(chunks) if ids]
        for f in futures: f.result()
    print('Rasterizing reference UV mapping',flush=True)
    ref=np.load(out/'reference.npz'); mapping=raster_reference(ref)
    np.savez_compressed(out/'uv_mapping.npz',indices=mapping[0],weights=mapping[1],mask=mapping[2],bbox=mapping[3])
    holdout=set(np.random.default_rng(args.seed+1).choice(args.count,args.holdout,replace=False).tolist())
    print('Preparing FLUX.2 VAE checkpoint',flush=True)
    checkpoint(vae_only=True)
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval()
    for idx in range(args.count):
        folder=out/'samples'/f'{idx:05d}'
        if (folder/'complete.json').exists(): continue
        meta=json.loads((folder/'render_complete.json').read_text()); geom=np.load(folder/'geometry.npz')
        tex,disp,maps,clipped=make_maps(meta['texture'],geom['displacement_m'],mapping,args.bound_mm)
        tex.save(folder/'texture.png'); disp.save(folder/'displacement.png'); maps.save(folder/'maps.png')
        latents={}
        for name,file in [('face_hair','portrait_hair.png'),('face_bare','portrait_bare.png'),('maps','maps.png')]:
            image=Image.open(folder/file).convert('RGBA'); black=Image.new('RGBA',image.size,(0,0,0,255)); black.alpha_composite(image)
            image=black.convert('RGB'); image.save(folder/file)
            latents[name]=vae_encode(vae,image)
        torch.save(latents,folder/'latents.pt')
        meta.update({'split':'holdout' if idx in holdout else 'train','clipped_coordinate_fraction':clipped})
        (folder/'complete.json').write_text(json.dumps(meta,indent=2))
        print('COMPLETE',idx,'split',meta['split'],'clipped',clipped,flush=True)
    print('DATASET_READY',out,flush=True)


if __name__=='__main__': main()
