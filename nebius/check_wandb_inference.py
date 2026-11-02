#!/usr/bin/env python3
"""Check Forge run history and uploaded inference images at requested steps."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import wandb
from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--steps',type=int,nargs='+',default=[200,400])
    parser.add_argument('--width',type=int,default=1024)
    parser.add_argument('--height',type=int,default=852)
    args=parser.parse_args()
    run=wandb.Api(timeout=60).run(args.run)
    seen={}
    for row in run.scan_history(keys=['train/global_step','holdout/comparison'],page_size=1000):
        step=int(row['train/global_step']);value=row['holdout/comparison']
        if step not in args.steps:continue
        if not isinstance(value,dict) or not value.get('path'):raise ValueError(f'Invalid W&B image at {step}')
        with tempfile.TemporaryDirectory(prefix='face-wandb-check-') as temp:
            downloaded=run.file(value['path']).download(root=temp,replace=True)
            with Image.open(downloaded.name) as image:
                pixels=np.asarray(image.convert('RGB'))
                size=list(image.size)
                if size!=[args.width,args.height] or float(pixels.std())<5:raise ValueError(f'Invalid W&B image at {step}: {size}')
            seen[step]={'file':value['path'],'size':size,'pixel_std':float(pixels.std())}
    missing=set(args.steps)-set(seen)
    if missing:raise ValueError(f'Inference history missing steps {sorted(missing)}')
    print('WANDB_INFERENCE_VERIFIED',json.dumps({'run':args.run,'state':run.state,'steps':seen}),flush=True)


if __name__=='__main__':main()
