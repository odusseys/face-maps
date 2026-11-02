#!/usr/bin/env python3
"""Prepare displacement-only targets with identical portrait skin texture."""
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from facekit.paths import DATA, MODEL, ROOT
from facekit.bootstrap import checkpoint
from facekit.latent import vae_encode


def atomic_json(path, value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def fixed_rgb(path):
    img=Image.open(path).convert('RGBA')
    background=Image.new('RGBA',img.size,(0,0,0,255));background.alpha_composite(img)
    return background.convert('RGB')


def verify_render(old, new):
    original=json.loads((old/'render_complete.json').read_text())
    rendered=json.loads((new/'render_complete.json').read_text())
    for key in ('index','seed','gender','age','race_weights','targets','hair','camera'):
        if original[key]!=rendered[key]:raise ValueError(f'{old.name} changed {key}')
    if rendered['skin']!='young_caucasian_female':raise ValueError(f'{old.name} has incorrect skin')
    a=np.load(old/'geometry.npz')['displacement_m']; b=np.load(new/'geometry.npz')['displacement_m']
    if not np.array_equal(a,b):raise ValueError(f'{old.name} geometry changed')
    return rendered


def main():
    from diffusers import AutoencoderKLFlux2
    original=DATA/'datasets/faces500-centered'
    source=DATA/'datasets/faces500-uv-aligned-uvq97'
    renders=DATA/'previews/faces500-fixed-texture-renders/samples'
    output=DATA/'datasets/faces500-uv-aligned-fixed-texture'
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for encoding portrait latents')
    source_state=json.loads((source/'dataset_complete.json').read_text())
    if not source_state.get('displacement_latents_ready'):raise ValueError('Source displacement latents incomplete')
    records=sorted((source/'samples').glob('*/complete.json'))
    if len(records)!=500:raise ValueError('Expected 500 corrected source faces')
    verified=[]
    for p in records:
        rendered=verify_render(original/'samples'/p.parent.name,renders/p.parent.name)
        verified.append(rendered)
    textures={v['texture'] for v in verified}
    if len(textures)!=1:raise ValueError('Rendered faces do not share one diffuse texture')
    texture=Path(textures.pop()).resolve()
    if not texture.is_file() or not texture.is_relative_to(ROOT.resolve()):raise ValueError('Texture unavailable')
    digest=hashlib.sha256(texture.read_bytes()).hexdigest()
    print('ALL_RENDERED_GEOMETRIES_MATCH',len(records),'texture_sha256',digest,flush=True)
    output.mkdir(parents=True,exist_ok=True)
    config=json.loads((source/'dataset.json').read_text())
    config.update({'latent_modes':['displacement_only'],'fixed_texture_path':str(texture),
                   'fixed_texture_sha256':digest,'fixed_texture_source_dataset':str(source),
                   'portrait_generation':'original face parameters and camera, fixed skin, fresh FLUX.2 portrait latents'})
    config_path=output/'dataset.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Output configuration does not match; choose a new directory')
    atomic_json(config_path,config)
    for name in ('reference.npz','reference.blend','reference_skin.txt','uv_mapping.npz'):
        if not (output/name).exists():shutil.copy2(source/name,output/name)
    target_dir=output/'latents_displacement';target_dir.mkdir(exist_ok=True)
    checkpoint(vae_only=True)
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval()
    for i, (meta_path, render_meta) in enumerate(zip(records,verified),1):
        name=meta_path.parent.name; dst=output/'samples'/name; dst.mkdir(parents=True,exist_ok=True)
        src=meta_path.parent; render=renders/name
        if (dst/'complete.json').is_file():continue
        for file in ('geometry.npz','displacement.png'):
            shutil.copy2(src/file,dst/file)
        shutil.copy2(render/'render_complete.json',dst/'render_complete.json')
        target=target_dir/(name+'.pt')
        if not target.is_file():shutil.copy2(source/'latents_displacement'/(name+'.pt'),target)
        features={}
        for key,filename in (('face_hair','portrait_hair.png'),('face_bare','portrait_bare.png')):
            img=fixed_rgb(render/filename);img.save(dst/filename)
            features[key]=vae_encode(vae,img)
            if tuple(features[key].shape)!=(128,32,32) or not torch.isfinite(features[key]).all():
                raise ValueError(f'Invalid {name} {key} portrait latent')
        tmp=dst/'latents.pt.tmp';torch.save(features,tmp);tmp.replace(dst/'latents.pt')
        meta=json.loads(meta_path.read_text())
        meta.update({'skin':render_meta['skin'],'texture':str(texture),'fixed_texture_sha256':digest,
                     'paired_map_latent_available':False})
        atomic_json(dst/'complete.json',meta)
        if i%25==0 or i==len(records):print('ENCODED',i,len(records),flush=True)
    state={'count':len(records),'bound_mm':config['bound_mm'],'encoding':config['encoding'],
           'displacement_latents_ready':True,'paired_map_latents_ready':False,
           'fixed_texture_path':str(texture),'fixed_texture_sha256':digest,
           'render_geometry_verified_against_original':True}
    atomic_json(output/'dataset_complete.json',state)
    print('FIXED_TEXTURE_DATASET_READY',str(output),flush=True)


if __name__=='__main__':main()
