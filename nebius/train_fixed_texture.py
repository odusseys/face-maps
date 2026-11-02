#!/usr/bin/env python3
"""Enforce a single verified texture for Nebius displacement-only training."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--displacement-only',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--log-every',type=int,default=200)
    parser.add_argument('--validate-only',action='store_true')
    args,_=parser.parse_known_args()
    if not args.displacement_only:raise ValueError('This launcher is for displacement-only training')
    if args.log_every!=200:raise ValueError('Displacement inference comparisons must be logged every 200 steps')
    config=json.loads((args.dataset/'dataset.json').read_text())
    state=json.loads((args.dataset/'dataset_complete.json').read_text())
    texture=Path(config['fixed_texture_path']).resolve();expected=config['fixed_texture_sha256']
    if not texture.is_file() or hashlib.sha256(texture.read_bytes()).hexdigest()!=expected:
        raise ValueError('Fixed texture is missing or has changed')
    if not state.get('displacement_latents_ready') or state.get('fixed_texture_sha256')!=expected:
        raise ValueError('Fixed-texture dataset is not ready')
    records=list((args.dataset/'samples').glob('*/complete.json'))
    if len(records)!=state['count'] or len(records)!=500:raise ValueError('Dataset sample count is incorrect')
    for path in records:
        meta=json.loads(path.read_text())
        if Path(meta['texture']).resolve()!=texture or meta.get('fixed_texture_sha256')!=expected:
            raise ValueError(f'Inconsistent texture for {path.parent.name}')
    print('FIXED_TEXTURE_VERIFIED',len(records),str(texture),'INFERENCE_INTERVAL',args.log_every,flush=True)
    if args.validate_only:return
    runpy.run_path(str(Path(__file__).resolve().parent.parent/'train_face_maps.py'),run_name='__main__')


if __name__=='__main__':main()
