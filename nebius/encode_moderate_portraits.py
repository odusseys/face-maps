#!/usr/bin/env python3
"""Prepare 500 moderate-shape renders for UV dataset construction."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLFlux2
from facekit.paths import DATA, MODEL
from facekit.bootstrap import checkpoint
from facekit.latent import vae_encode


def atomic_json(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    source=DATA/'datasets/faces500-moderate-centered'
    reference=DATA/'datasets/faces500-centered'
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for portrait encoding')
    paths=[source/'samples'/f'{i:05d}' for i in range(500)]
    if any(not (p/'render_complete.json').is_file() for p in paths):raise ValueError('500 renders required')
    for name in ('reference.blend','reference_skin.txt','uv_mapping.npz'):
        if not (source/name).exists():shutil.copy2(reference/name,source/name)
    info=json.loads((reference/'dataset.json').read_text())
    info.update({'seed':202611011,'count':500,'holdout':100,'portrait_skin':'young_caucasian_female',
                 'shape_distribution':'moderate variation preview, random hair', 'latent_modes':['portrait_only']})
    config_path=source/'dataset.json'
    if config_path.exists() and json.loads(config_path.read_text())!=info:raise ValueError('Dataset settings changed')
    atomic_json(config_path,info)
    holdout=set(np.random.default_rng(info['seed']+1).choice(500,100,replace=False).tolist())
    textures={json.loads((p/'render_complete.json').read_text())['texture'] for p in paths}
    if len(textures)!=1:raise ValueError('Portraits must share the same texture')
    texture=Path(textures.pop());digest=hashlib.sha256(texture.read_bytes()).hexdigest()
    checkpoint(vae_only=True)
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval()
    for i,p in enumerate(paths):
        if (p/'complete.json').exists():continue
        meta=json.loads((p/'render_complete.json').read_text());features={}
        for key,name in (('face_hair','portrait_hair.png'),('face_bare','portrait_bare.png')):
            image=Image.open(p/name).convert('RGBA');background=Image.new('RGBA',image.size,(0,0,0,255))
            background.alpha_composite(image);rgb=background.convert('RGB');rgb.save(p/name)
            features[key]=vae_encode(vae,rgb)
            if tuple(features[key].shape)!=(128,32,32) or not torch.isfinite(features[key]).all():
                raise ValueError(f'Invalid latent: {p.name} {key}')
        tmp=p/'latents.pt.tmp';torch.save(features,tmp);tmp.replace(p/'latents.pt')
        meta.update({'split':'holdout' if i in holdout else 'train','fixed_texture_sha256':digest})
        atomic_json(p/'complete.json',meta)
        if (i+1)%25==0:print('PORTRAITS_ENCODED',i+1,len(paths),flush=True)
    atomic_json(source/'portrait_manifest.json',{'count':500,'train':400,'holdout':100,
                 'texture':str(texture),'fixed_texture_sha256':digest})
    print('PORTRAITS_READY',source,flush=True)


if __name__=='__main__':main()
