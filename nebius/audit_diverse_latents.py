#!/usr/bin/env python3
"""Re-encode paired and portrait images to independently audit sample pairing."""
import json
import random
import torch
from PIL import Image
from diffusers import AutoencoderKLFlux2
from facekit.paths import DATA,MODEL
from facekit.latent import vae_encode


def main():
    ds=DATA/'datasets/faces1000-visibleface-uv-paired'
    paths=sorted((ds/'samples').glob('*/complete.json'))
    groups={s:[p.parent for p in paths if json.loads(p.read_text())['split']==s] for s in ('train','holdout')}
    rng=random.Random(11722);chosen=rng.sample(groups['train'],6)+rng.sample(groups['holdout'],6)
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    checks=[]
    for p in chosen:
        features=torch.load(p/'latents.pt',map_location='cpu',weights_only=True)
        for key,name in (('face_hair','portrait_hair.png'),('face_bare','portrait_bare.png'),('maps','maps.png')):
            recoded=vae_encode(vae,Image.open(p/name).convert('RGB'));saved=features[key]
            difference=(recoded.float()-saved.float()).abs()
            maximum=float(difference.max());mean=float(difference.mean())
            if maximum>.002:raise ValueError(f'Latent mismatches same-identity image: {p.name} {key}, {maximum}')
            checks.append({'id':p.name,'key':key,'max_abs':maximum,'mean_abs':mean})
        print('PAIRED_LATENTS_REENCODED',p.name,flush=True)
    result={'identities':len(chosen),'checks':checks}
    out=DATA/'audits/nebius-diverse-paired-latent-audit.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2));print('PAIRED_LATENT_AUDIT_PASSED',len(chosen),flush=True)


if __name__=='__main__':main()
