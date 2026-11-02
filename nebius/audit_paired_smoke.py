#!/usr/bin/env python3
"""Check all seven paired-smoke grid panels match their own holdout identity."""
import json
import random
import numpy as np
import torch
from PIL import Image
from facekit.paths import DATA
from train_face_maps import list_samples


def main():
    ds=DATA/'datasets/faces500-moderate-uv-fixed-texture'
    run=DATA/'training/nebius-paired-wiring-smoke-v2';step=3
    _,holdout=list_samples(ds);picks=random.Random(20261031+step).sample(holdout,3)
    grid=np.asarray(Image.open(run/'debug'/f'{step:07d}'/'comparison.png').convert('RGB'))
    if grid.shape!=(852,1792,3):raise ValueError(f'Invalid grid shape {grid.shape}')
    checks=[]
    for row,p in enumerate(picks):
        work=run/'debug'/f'{step:07d}'/p.name
        gt=Image.open(p/'maps.png').convert('RGB');pred=Image.open(work/'predicted_maps.png').convert('RGB')
        if gt.size!=(1024,512) or pred.size!=(1024,512):raise ValueError('Paired maps have incorrect size')
        panels=[Image.open(p/'portrait_hair.png'),Image.open(p/'portrait_bare.png'),
                Image.open(work/'predicted_face.png'),gt.crop((0,0,512,512)),pred.crop((0,0,512,512)),
                gt.crop((512,0,1024,512)),pred.crop((512,0,1024,512))]
        for col,im in enumerate(panels):
            expected=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS))
            actual=grid[row*284+28:row*284+284,col*256:(col+1)*256]
            if not np.array_equal(actual,expected):raise ValueError(f'Mismatched row {row} panel {col}, {p.name}')
        f=torch.load(p/'latents.pt',map_location='cpu',weights_only=True)
        if tuple(f['face_hair'].shape)!=(128,32,32) or tuple(f['maps'].shape)!=(128,32,64):
            raise ValueError('Unexpected paired training latent shape')
        checks.append({'id':p.name,'panels_matched':7})
    result={'step':step,'grid_shape':list(grid.shape),'target_shape':[128,32,64],
            'condition':'face_hair','rows':checks}
    out=DATA/'audits/nebius-paired-smoke-wiring.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2));print('PAIRED_SMOKE_WIRING_PASSED',json.dumps(result),flush=True)


if __name__=='__main__':main()
