#!/usr/bin/env python3
"""Reproduce a saved holdout preview from its checkpoint, without editing the run."""
import json
import random
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLFlux2,Flux2Transformer2DModel
from peft import set_peft_model_state_dict
from facekit.paths import DATA,MODEL
from facekit.latent import vae_decode
from facekit.train_helpers import infer,inference_schedule,prompt_embedding
from train_face_maps import add_lora,list_samples


def main():
    ds=DATA/'datasets/faces500-moderate-uv-fixed-texture';run=DATA/'training/flux2_moderate_fixed_texture'
    candidates=[p for p in sorted(run.glob('checkpoint-*.pt')) if (run/'debug'/p.stem.removeprefix('checkpoint-')/'comparison.png').is_file()]
    if not candidates:raise ValueError('No finished checkpoint and preview')
    ckpt=candidates[-1];state=torch.load(ckpt,map_location='cpu',weights_only=False);step=state['step']
    model=Flux2Transformer2DModel.from_pretrained(str(MODEL),subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    add_lora(model);model.eval();result=set_peft_model_state_dict(model,state['lora'])
    if result.unexpected_keys:raise ValueError('Unexpected adapter weights')
    del state
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    embeds=prompt_embedding(MODEL,True);_,holdout=list_samples(ds)
    picks=random.Random(20261031+step).sample(holdout,3);checks=[]
    for i,p in enumerate(picks):
        condition=torch.load(p/'latents.pt',map_location='cpu',weights_only=True)['face_bare'].unsqueeze(0).to('cuda',torch.bfloat16)
        with torch.inference_mode():image=vae_decode(vae,infer(model,condition,embeds,MODEL,28,20261031+step+i,True))[0]
        saved=np.asarray(Image.open(run/'debug'/f'{step:07d}'/p.name/'predicted_maps.png').convert('RGB'),dtype=np.int16)
        fresh=np.asarray(image,dtype=np.int16);diff=np.abs(fresh-saved)
        checks.append({'id':p.name,'seed':20261031+step+i,'max_pixel_difference':int(diff.max()),
                       'mean_pixel_difference':float(diff.mean()),'fraction_different':float(np.mean(diff!=0))})
        print('PREVIEW_REPRODUCED',json.dumps(checks[-1]),flush=True)
    target=torch.load(ds/'latents_displacement'/(picks[0].name+'.pt'),map_location='cpu',weights_only=True).to('cuda',torch.bfloat16)
    noise=torch.randn(target.shape,device='cuda',dtype=target.dtype,generator=torch.Generator(device='cuda').manual_seed(4321))
    schedule=inference_schedule(MODEL,28,target.shape,'cuda');diffs=[]
    for i,t in enumerate(schedule.timesteps):
        q=(t.to(target.dtype)/1000);trained=(1-q)*target+q*noise
        sigma=schedule.sigmas[i].to('cuda');ideal=((1-sigma)*target.float()+sigma*noise.float()).to(target.dtype)
        diffs.append((trained.float()-ideal.float()).square().mean())
    rmse=torch.stack(diffs).sqrt()
    report={'checkpoint_step':step,'preview_reproduction':checks,'training_mixture_vs_exact_sampler_sigma':{
        'rmse_mean':float(rmse.mean()),'rmse_max':float(rmse.max()),'latent_target_rms':float(target.float().square().mean().sqrt()),
        'description':'Difference from BF16 timestep and mixture arithmetic, relative to mixing the same BF16 latents at the scheduler float32 sigma.'}}
    out=DATA/'audits/nebius-saved-preview-reproduction.json';out.write_text(json.dumps(report,indent=2));print('AUDIT_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
