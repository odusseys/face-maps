#!/usr/bin/env python3
"""Verify diverse identity pairing, camera framing and paired-map latents."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from facekit.uv import make_maps


def validate(dataset,deep=True):
    dataset=Path(dataset).resolve();cfg=json.loads((dataset/'dataset.json').read_text())
    state=json.loads((dataset/'dataset_complete.json').read_text())
    if not state.get('paired_map_latents_ready') or 'paired_maps' not in cfg.get('latent_modes',[]):
        raise ValueError('Paired-map target latents are not complete')
    paths=sorted((dataset/'samples').glob('*/complete.json'))
    if len(paths)!=1000 or state['count']!=1000:raise ValueError('Expected 1000 complete identities')
    split=collections.Counter();skins=collections.Counter();hair=collections.Counter();textures={};distances=[];geometry_hashes=set()
    for i,p in enumerate(paths):
        meta=json.loads(p.read_text());folder=p.parent
        if int(folder.name)!=meta['index']:raise ValueError(f'Identity mismatch: {folder.name}')
        split[meta['split']]+=1;skins[meta['skin']]+=1;hair[meta['hair']]+=1
        cam=meta['camera'];bbox=cam['hair_face_bounds']
        if (cam['hair_projection']!='perspective' or cam['bare_projection']!='orthographic' or
            abs(cam['bare_ortho_scale_m']-.43)>1e-8 or min(bbox)<.04 or max(bbox)>.96):
            raise ValueError(f'Invalid camera or face framing: {folder.name}')
        distances.append(cam['hair_distance_m']);tex=Path(meta['texture'])
        shape=np.load(folder/'geometry.npz')['displacement_m']
        geometry_hashes.add(hashlib.sha256(shape.tobytes()).digest())
        if str(tex) not in textures:textures[str(tex)]=hashlib.sha256(tex.read_bytes()).hexdigest()
        if meta['texture_sha256']!=textures[str(tex)]:raise ValueError(f'Texture mismatch: {folder.name}')
        features=torch.load(folder/'latents.pt',map_location='cpu',weights_only=True)
        for key,shape in (('face_hair',(128,32,32)),('face_bare',(128,32,32)),('maps',(128,32,64))):
            t=features.get(key)
            if not isinstance(t,torch.Tensor) or tuple(t.shape)!=shape or not bool(torch.isfinite(t).all()):
                raise ValueError(f'Invalid {key} latent: {folder.name}')
        if Image.open(folder/'maps.png').size!=(1024,512):raise ValueError(f'Invalid paired map: {folder.name}')
    if dict(split)!={'train':900,'holdout':100} or len(skins)<15 or len(hair)<6 or len(set(textures.values()))<15:
        raise ValueError(f'Unexpected split or diversity: {split}, skins={len(skins)}, hair={len(hair)}, textures={len(set(textures.values()))}')
    if len(geometry_hashes)!=1000:raise ValueError('Duplicate face geometry found')
    if max(distances)-min(distances)<.2:raise ValueError('Camera distances did not vary')
    checked=0
    if deep:
        uv=np.load(dataset/'uv_mapping.npz');mapping=tuple(uv[k] for k in ('indices','weights','mask','bbox'))
        for j in np.linspace(0,999,24,dtype=int):
            p=paths[j];meta=json.loads(p.read_text());folder=p.parent
            delta=np.load(folder/'geometry.npz')['displacement_m']
            texture,disp,pair,_=make_maps(meta['texture'],delta,mapping,cfg['bound_mm'],cfg['encoding'])
            for name,expected in (('texture.png',texture),('displacement.png',disp),('maps.png',pair)):
                if not np.array_equal(np.asarray(expected),np.asarray(Image.open(folder/name).convert('RGB'))):
                    raise ValueError(f'Map does not match identity: {folder.name} {name}')
            checked+=1
    return {'count':len(paths),'train':split['train'],'holdout':split['holdout'],
            'skins':len(skins),'hairs':len(hair),'distinct_textures':len(set(textures.values())),
            'hair_distance_m':{'min':min(distances),'median':float(np.median(distances)),'max':max(distances)},
            'unique_geometries':len(geometry_hashes),'paired_maps_reconstructed':checked,'target_shape':[128,32,64],'condition_shape':[128,32,32]}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    args=p.parse_args();result=validate(args.dataset)
    out=args.dataset/'paired_validation.json';out.write_text(json.dumps(result,indent=2))
    print('PAIRED_DATASET_VALIDATED',json.dumps(result),flush=True)


if __name__=='__main__':main()
