#!/usr/bin/env python3
"""Create a separate, scale-invariant displacement-only dataset on shared storage.

Run encode_displacement_latents.py on the result before training. Portrait
latents are reused; paired texture/displacement latents must be recomputed
separately before using paired-map training.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from facekit.paths import ROOT, DATA
from facekit.similarity import normalize_aligned_displacement
from facekit.uv import make_maps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=DATA/'datasets'/'faces500-centered-linear-q97')
    p.add_argument('--output',type=Path,default=DATA/'datasets'/'faces500-similarity-linear-q97')
    p.add_argument('--sample-limit',type=int,default=0)
    p.add_argument('--estimate-only',action='store_true')
    args=p.parse_args();source=args.source.resolve();out=args.output.resolve()
    if not source.is_relative_to(ROOT.resolve()) or not out.is_relative_to(ROOT.resolve()) or source==out:
        p.error('Source and output must be distinct paths on shared storage')
    if args.sample_limit<0:p.error('sample-limit must be nonnegative')
    ref=np.load(source/'reference.npz');reference=ref['vertices_m'];mask=ref['face_mask']
    records=[];values=[]
    for meta_path in sorted((source/'samples').glob('*/complete.json')):
        meta=json.loads(meta_path.read_text());geom=np.load(meta_path.parent/'geometry.npz')
        delta,scale=normalize_aligned_displacement(reference,geom['displacement_m'],mask)
        records.append((meta_path,meta,delta,scale))
        if meta['split']=='train':values.append(np.abs(delta[mask]).reshape(-1)*1000)
    if not records or not values:raise ValueError('No complete samples or training samples')
    all_values=np.concatenate(values);bound=float(np.quantile(all_values,.97))
    if not np.isfinite(bound) or bound<=0:raise ValueError('Invalid displacement bound')
    summary={'bound_mm':bound,'train_samples':len(values),'samples':len(records),
             'normalization':'similarity-aligned signed linear','quantile':.97}
    print('SHAPE_BOUND',json.dumps(summary),flush=True)
    if args.estimate_only:return
    if args.sample_limit:records=records[:args.sample_limit]
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads((source/'dataset.json').read_text())
    config.update({'bound_mm':bound,'encoding':'signed_linear','normalization_quantile':.97,
                   'normalization_source':str(source),'normalization_population':'training split face vertices after similarity alignment',
                   'normalization_components':len(all_values),'vertex_components_above_bound_fraction':float(np.mean(all_values>bound)),
                   'geometry_alignment':'rigid alignment plus optimal uniform scale to reference face',
                   'latent_modes':['displacement_only'],'count':len(records),'smoke_only':bool(args.sample_limit)})
    config_path=out/'dataset.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Output configuration differs; use a new directory')
    config_path.write_text(json.dumps(config,indent=2))
    for name in ('reference.npz','reference.blend','reference_skin.txt','uv_mapping.npz'):
        if not (out/name).exists():shutil.copy2(source/name,out/name)
    mapping_data=np.load(source/'uv_mapping.npz')
    mapping=tuple(mapping_data[k] for k in ('indices','weights','mask','bbox'))
    for meta_path,meta,delta,scale in records:
        src=meta_path.parent;dst=out/'samples'/src.name;dst.mkdir(parents=True,exist_ok=True)
        if (dst/'complete.json').exists():continue
        for name in ('portrait_hair.png','portrait_bare.png','render_complete.json'):
            shutil.copy2(src/name,dst/name)
        with np.load(src/'geometry.npz') as source_geom:
            geom={k:source_geom[k] for k in source_geom.files}
        geom.update(displacement_m=delta,similarity_scale=np.float64(scale))
        np.savez_compressed(dst/'geometry.npz',**geom)
        tex,disp,maps,clipped=make_maps(meta['texture'],delta,mapping,bound,encoding='signed_linear')
        tex.save(dst/'texture.png');disp.save(dst/'displacement.png');maps.save(dst/'maps.png')
        previous=torch.load(src/'latents.pt',map_location='cpu',weights_only=True)
        inputs={k:previous[k] for k in ('face_hair','face_bare')}
        torch.save(inputs,dst/'latents.pt')
        meta.update({'clipped_coordinate_fraction':clipped,'displacement_encoding':'signed_linear',
                     'bound_mm':bound,'similarity_scale':scale,'geometry_alignment':'similarity',
                     'paired_map_latent_available':False})
        temp=dst/'complete.json.tmp';temp.write_text(json.dumps(meta,indent=2));temp.replace(dst/'complete.json')
        print('COMPLETE',src.name,'scale',scale,flush=True)
    (out/'dataset_complete.json').write_text(json.dumps({'count':len(records),'bound_mm':bound,'encoding':'signed_linear',
          'displacement_latents_ready':False,'paired_map_latents_ready':False},indent=2))
    print('DATASET_READY',str(out),flush=True)


if __name__=='__main__':main()
