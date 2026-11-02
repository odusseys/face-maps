#!/usr/bin/env python3
"""Encode 512px displacement maps to a separate, resumable FLUX.2 latent set."""
import argparse
import json
import os
from pathlib import Path
from facekit.paths import ROOT, DATA, MODEL
os.environ.setdefault('HF_HOME',str(DATA/'cache'/'huggingface'))
os.environ.setdefault('XDG_CACHE_HOME',str(DATA/'cache'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DATA/'datasets'/'faces500-centered-linear-q97')
    p.add_argument('--output',type=Path)
    p.add_argument('--limit',type=int,default=0,help='Encode first N samples for a smoke test')
    args=p.parse_args(); output=args.output or args.dataset/'latents_displacement'
    if not args.dataset.resolve().is_relative_to(ROOT.resolve()) or not output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Dataset and output must be on shared storage')
    if args.limit<0: raise ValueError('limit must be nonnegative')
    samples=sorted((args.dataset/'samples').glob('*/complete.json'))
    if not samples: raise ValueError('No completed samples')
    if args.limit: samples=samples[:args.limit]
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLFlux2
    from facekit.bootstrap import checkpoint
    from facekit.latent import vae_encode
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    output.mkdir(parents=True,exist_ok=True)
    missing=[p for p in samples if not (output/(p.parent.name+'.pt')).exists()]
    if missing:
        checkpoint(vae_only=True)
        vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval()
        for meta in missing:
            img=Image.open(meta.parent/'displacement.png').convert('RGB')
            if img.size!=(512,512): raise ValueError(f'Unexpected image size for {meta.parent.name}: {img.size}')
            z=vae_encode(vae,img)
            if tuple(z.shape)!=(128,32,32) or not bool(torch.isfinite(z).all()): raise ValueError(f'Invalid latent for {meta.parent.name}')
            path=output/(meta.parent.name+'.pt'); temp=path.with_suffix('.tmp')
            torch.save(z,temp); temp.replace(path)
            print('ENCODED',meta.parent.name,flush=True)
    # Validate existing files too, so interrupted or stale outputs never silently enter training.
    for meta in samples:
        z=torch.load(output/(meta.parent.name+'.pt'),map_location='cpu',weights_only=True)
        if tuple(z.shape)!=(128,32,32) or not bool(torch.isfinite(z).all()): raise ValueError(f'Invalid saved latent for {meta.parent.name}')
    manifest={'dataset':str(args.dataset.resolve()),'count':len(samples),'model':str(MODEL),'input':'displacement.png','shape':[128,32,32], 'smoke_only':bool(args.limit)}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    complete=args.dataset/'dataset_complete.json'
    if not args.limit and output.resolve()==(args.dataset/'latents_displacement').resolve() and complete.exists():
        state=json.loads(complete.read_text())
        if 'displacement_latents_ready' in state:
            state['displacement_latents_ready']=True
            temp=complete.with_suffix('.json.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(complete)
    print('LATENTS_READY',json.dumps(manifest),flush=True)


if __name__=='__main__': main()
