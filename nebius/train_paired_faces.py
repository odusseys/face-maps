#!/usr/bin/env python3
"""Guard the Nebius paired-texture/displacement training entry point."""
import argparse
import json
import runpy
from pathlib import Path
from nebius.validate_paired_dataset import validate


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--displacement-only',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--log-every',type=int,default=200)
    parser.add_argument('--validate-only',action='store_true')
    args,_=parser.parse_known_args()
    if args.displacement_only:raise ValueError('Use --no-displacement-only for paired maps')
    if args.log_every!=200:raise ValueError('Inference logging interval must be 200')
    summary=validate(args.dataset,deep=True)
    print('PAIRED_TRAINING_INPUT_VERIFIED',json.dumps(summary),flush=True)
    if args.validate_only:return
    runpy.run_path(str(Path(__file__).resolve().parent.parent/'train_face_maps.py'),run_name='__main__')


if __name__=='__main__':main()
