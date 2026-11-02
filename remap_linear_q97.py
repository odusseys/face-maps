#!/usr/bin/env python3
"""Rebuild a centered face dataset with one global linear 97th-percentile scale."""
import argparse
import json
import os
import shutil
from pathlib import Path
from facekit.paths import ROOT, DATA, MODEL
os.environ.setdefault('HF_HOME',str(DATA/'cache'/'huggingface'))
os.environ.setdefault('XDG_CACHE_HOME',str(DATA/'cache'))


def estimate(source):
    import numpy as np
    ref=np.load(source/'reference.npz'); mask=ref['face_mask']
    paths=sorted((source/'samples').glob('*/complete.json'))
    if not paths: raise ValueError('No completed source samples')
    values=[]
    for meta in paths:
        with np.load(meta.parent/'geometry.npz') as z:
            values.append(np.abs(z['displacement_m'][mask]).reshape(-1)*1000)
    values=np.concatenate(values)
    bound=float(np.quantile(values,.97))
    if not np.isfinite(bound) or bound<=0: raise ValueError('Invalid 97th-percentile bound')
    return bound,len(paths),int(values.size),float(np.mean(values>bound))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DATA/'datasets'/'faces500-centered')
    parser.add_argument('--output',type=Path,default=DATA/'datasets'/'faces500-centered-linear-q97')
    parser.add_argument('--sample-limit',type=int,default=0,help='Use only first N outputs for smoke testing; scale still uses all source samples')
    parser.add_argument('--estimate-only',action='store_true')
    args=parser.parse_args(); source=args.source.resolve(); out=args.output.resolve()
    if not source.is_relative_to(ROOT.resolve()) or not out.is_relative_to(ROOT.resolve()) or source==out:
        parser.error('source and output must be distinct paths on shared storage')
    bound,count,components,clip=estimate(source)
    print('GLOBAL_Q97',json.dumps({'bound_mm':bound,'samples':count,'components':components,
                                  'vertex_components_above_bound_fraction':clip}),flush=True)
    if args.estimate_only: return
    if args.sample_limit<0: parser.error('sample-limit must be nonnegative')
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLFlux2
    from facekit.bootstrap import checkpoint
    from facekit.latent import vae_encode
    from facekit.uv import make_maps
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for VAE encoding')
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads((source/'dataset.json').read_text())
    config.update({'bound_mm':bound,'encoding':'signed_linear',
                   'normalization_quantile':.97,'normalization_source':str(source),
                   'normalization_population':'absolute XYZ coordinate displacements at all face vertices across all source samples',
                   'normalization_components':components,'vertex_components_above_bound_fraction':clip})
    paths=sorted((source/'samples').glob('*/complete.json'))
    if args.sample_limit: paths=paths[:args.sample_limit]; config['count']=len(paths); config['smoke_only']=True
    path=out/'dataset.json'
    if path.exists() and json.loads(path.read_text())!=config: raise ValueError('Output configuration differs; use a new output directory')
    path.write_text(json.dumps(config,indent=2))
    for name in ('reference.npz','reference.blend','reference_skin.txt','uv_mapping.npz'):
        if not (out/name).exists(): shutil.copy2(source/name,out/name)
    if all((out/'samples'/p.parent.name/'complete.json').exists() for p in paths):
        (out/'dataset_complete.json').write_text(json.dumps({'count':len(paths),'bound_mm':bound,'encoding':'signed_linear'},indent=2))
        print('DATASET_READY',out,flush=True); return
    mapping_data=np.load(source/'uv_mapping.npz')
    mapping=(mapping_data['indices'],mapping_data['weights'],mapping_data['mask'],mapping_data['bbox'])
    checkpoint(vae_only=True)
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval()
    for meta_path in paths:
        src=meta_path.parent; dst=out/'samples'/src.name; dst.mkdir(parents=True,exist_ok=True)
        if (dst/'complete.json').exists(): continue
        meta=json.loads(meta_path.read_text()); geom=np.load(src/'geometry.npz')
        for name in ('portrait_hair.png','portrait_bare.png','geometry.npz','render_complete.json'):
            shutil.copy2(src/name,dst/name)
        tex,disp,maps,clipped=make_maps(meta['texture'],geom['displacement_m'],mapping,bound,encoding='signed_linear')
        tex.save(dst/'texture.png'); disp.save(dst/'displacement.png'); maps.save(dst/'maps.png')
        latents={name:vae_encode(vae,Image.open(dst/file)) for name,file in
                 [('face_hair','portrait_hair.png'),('face_bare','portrait_bare.png'),('maps','maps.png')]}
        expected={'face_hair':(128,32,32),'face_bare':(128,32,32),'maps':(128,32,64)}
        for name,value in latents.items():
            if tuple(value.shape)!=expected[name] or not bool(torch.isfinite(value).all()):
                raise ValueError(f'Invalid {name} latent for {src.name}')
        torch.save(latents,dst/'latents.pt')
        meta.update({'clipped_coordinate_fraction':clipped,'displacement_encoding':'signed_linear','bound_mm':bound})
        temp=dst/'complete.json.tmp';temp.write_text(json.dumps(meta,indent=2));temp.replace(dst/'complete.json')
        print('COMPLETE',src.name,'split',meta['split'],'clipped',clipped,flush=True)
    (out/'dataset_complete.json').write_text(json.dumps({'count':len(paths),'bound_mm':bound,'encoding':'signed_linear'},indent=2))
    print('DATASET_READY',out,flush=True)


if __name__=='__main__': main()
