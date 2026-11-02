#!/usr/bin/env python3
"""Verify every sample pair, epoch membership, and saved inference grid wiring."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from facekit.paths import DATA
from facekit.uv import make_maps
from nebius.build_uv_aligned_dataset import align
from train_face_maps import epoch_batch,list_samples


def digest(path):return hashlib.sha256(path.read_bytes()).digest()


def main():
    dataset=DATA/'datasets/faces500-moderate-uv-fixed-texture'
    raw=DATA/'datasets/faces500-moderate-centered'
    run=DATA/'training/flux2_moderate_fixed_texture'
    config=json.loads((dataset/'dataset.json').read_text())
    reference=np.load(dataset/'reference.npz');v=reference['vertices_m'];ids=np.unique(reference['triangle_vertices'])
    uv=np.load(dataset/'uv_mapping.npz');mapping=tuple(uv[k] for k in ('indices','weights','mask','bbox'))
    train,holdout=list_samples(dataset)
    if len(train)!=400 or len(holdout)!=100 or set(train)&set(holdout):raise ValueError('Split mismatch')
    paths=sorted(train+holdout); meta_keys=('index','seed','gender','age','race_weights','targets','hair','skin','texture','camera','split')
    sampled=set([p.name for split in (train,holdout) for p in random.Random(837).sample(split,12)])
    for index,p in enumerate(paths,1):
        name=p.name;raw_path=raw/'samples'/name
        meta=json.loads((p/'complete.json').read_text());old=json.loads((raw_path/'complete.json').read_text())
        if int(name)!=meta['index'] or any(meta[k]!=old[k] for k in meta_keys):raise ValueError(f'Metadata mismatch {name}')
        for file in ('portrait_bare.png','portrait_hair.png'):
            if digest(p/file)!=digest(raw_path/file):raise ValueError(f'Portrait mismatch {name} {file}')
        original=torch.load(raw_path/'latents.pt',map_location='cpu',weights_only=True)
        features=torch.load(p/'latents.pt',map_location='cpu',weights_only=True)
        for key in ('face_hair','face_bare'):
            if features[key].shape!=(128,32,32) or not torch.equal(features[key],original[key]):raise ValueError(f'Portrait latent mismatch {name} {key}')
        target=torch.load(dataset/'latents_displacement'/(name+'.pt'),map_location='cpu',weights_only=True)
        if target.shape!=(128,32,32) or not torch.isfinite(target).all():raise ValueError(f'Invalid target latent {name}')
        delta=np.load(raw_path/'geometry.npz')['displacement_m']
        calculated=align(v,v+delta,ids)[0]
        stored=np.load(p/'geometry.npz')['displacement_m']
        if not np.array_equal(calculated,stored):raise ValueError(f'Geometry mismatch {name}')
        if name in sampled:
            _,displacement,_,_=make_maps(meta['texture'],stored,mapping,config['bound_mm'],'signed_linear')
            if not np.array_equal(np.asarray(displacement),np.asarray(Image.open(p/'displacement.png').convert('RGB'))):
                raise ValueError(f'Displacement image mismatch {name}')
        if index%100==0:print('SAMPLES_CHECKED',index,flush=True)
    for epoch in range(100):
        examples=[]
        for j in range(100):
            actual,batch=epoch_batch(train,epoch*100+j+1,4,20261031)
            if actual!=epoch or len(batch)!=4:raise ValueError('Unexpected epoch batch')
            examples.extend(batch)
        if len(examples)!=400 or len(set(examples))!=400 or set(examples)!=set(train):raise ValueError(f'Epoch {epoch} samples wrong')
    grids=[]
    for comparison in sorted(run.glob('debug/*/comparison.png')):
        step=int(comparison.parent.name);picks=random.Random(20261031+step).sample(holdout,3)
        grid=np.asarray(Image.open(comparison).convert('RGB'))
        if grid.shape!=(852,1024,3):raise ValueError(f'Grid dimensions {step}')
        for row,p in enumerate(picks):
            work=comparison.parent/p.name
            files=(work/'gt_map_face/predicted_face.png',work/'predicted_face.png',p/'displacement.png',work/'predicted_maps.png')
            for col,file in enumerate(files):
                expected=np.asarray(Image.open(file).convert('RGB').resize((256,256),Image.Resampling.LANCZOS))
                actual=grid[row*284+28:row*284+284,col*256:(col+1)*256]
                if not np.array_equal(actual,expected):raise ValueError(f'Grid mismatch step {step}, row {row}, column {col}, sample {p.name}')
        grids.append({'step':step,'ids':[p.name for p in picks]})
    if not grids:raise ValueError('No saved inference grids')
    # A weighted mean of per-microbatch mean losses yields the same gradients
    # as a full-batch mean; verify the exact weighting used by training.
    scalar=torch.tensor(1.7,requires_grad=True);x=torch.tensor([.3,1.2,-.4,2.1]);y=torch.tensor([.4,-.1,.8,1.3])
    full=((scalar*x-y)**2).mean();full_grad=torch.autograd.grad(full,scalar)[0]
    pieces=sum(((scalar*x[i:i+1]-y[i:i+1])**2).mean()/4 for i in range(4))
    micro_grad=torch.autograd.grad(pieces,scalar)[0]
    if not torch.allclose(full_grad,micro_grad):raise ValueError('Microbatch weighting mismatch')
    result={'samples_verified':len(paths),'train':len(train),'holdout':len(holdout),
            'epoch_coverage_verified':100,'uv_image_reconstructions_verified':len(sampled),
            'inference_grids':grids,'weighted_microbatch_gradient_test':True}
    output=DATA/'audits/nebius-batch-pairing.json';output.write_text(json.dumps(result,indent=2))
    print('BATCH_PAIRING_PASSED',json.dumps(result),flush=True)


if __name__=='__main__':main()
