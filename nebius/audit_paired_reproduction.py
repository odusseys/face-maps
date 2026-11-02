#!/usr/bin/env python3
"""Reproduce the paired smoke's hair-conditioned preview from its checkpoint."""
import json,random
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLFlux2,Flux2Transformer2DModel
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from facekit.paths import DATA,MODEL
from facekit.latent import vae_decode
from facekit.train_helpers import infer,prompt_embedding
from train_face_maps import add_lora,list_samples


def main():
    ds=DATA/'datasets/faces500-moderate-uv-fixed-texture';run=DATA/'training/nebius-paired-wiring-smoke-v2';step=3
    state=torch.load(run/'checkpoint-0000003.pt',map_location='cpu',weights_only=False)
    if state['step']!=3 or state['displacement_only']:raise ValueError('Incorrect smoke checkpoint')
    model=Flux2Transformer2DModel.from_pretrained(str(MODEL),subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    add_lora(model);model.eval()
    if set(state['lora'])!=set(get_peft_model_state_dict(model)):raise ValueError('Checkpoint mismatch')
    if set_peft_model_state_dict(model,state['lora']).unexpected_keys:raise ValueError('Unexpected weights')
    del state
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    embeds=prompt_embedding(MODEL,False);_,holdout=list_samples(ds);picks=random.Random(20261031+step).sample(holdout,3)
    checks=[]
    for i,p in enumerate(picks):
        cond=torch.load(p/'latents.pt',map_location='cpu',weights_only=True)['face_hair'].unsqueeze(0).to('cuda',torch.bfloat16)
        with torch.inference_mode():generated=vae_decode(vae,infer(model,cond,embeds,MODEL,28,20261031+step+i,False))[0]
        saved=Image.open(run/'debug'/f'{step:07d}'/p.name/'predicted_maps.png').convert('RGB')
        a=np.asarray(generated,dtype=np.int16);b=np.asarray(saved,dtype=np.int16)
        if generated.size!=(1024,512):raise ValueError('Incorrect generated paired-map dimensions')
        err=np.abs(a-b);entry={'id':p.name,'mean_pixel_difference':float(err.mean()),'max_pixel_difference':int(err.max())}
        if entry['mean_pixel_difference']>.25:raise ValueError(f'Preview reproduction mismatch: {entry}')
        checks.append(entry);print('PAIRED_PREVIEW_REPRODUCED',json.dumps(entry),flush=True)
    result={'checkpoint_step':step,'hair_conditioned':True,'paired_dimensions':[1024,512],'samples':checks}
    out=DATA/'audits/nebius-paired-smoke-reproduction.json';out.write_text(json.dumps(result,indent=2))
    print('PAIRED_REPRODUCTION_PASSED',json.dumps(result),flush=True)


if __name__=='__main__':main()
