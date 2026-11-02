#!/usr/bin/env python3
"""Compare cached training inputs and preview inference to the FLUX.2 pipeline."""
import json
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from facekit.paths import DATA, MODEL
from facekit.latent import vae_encode
from facekit.train_helpers import infer, inference_schedule, predict, prompt_embedding
from train_face_maps import add_lora


def describe(a,b):
    d=(a.float()-b.float()).abs()
    return {'max_abs':float(d.max()),'mean_abs':float(d.mean()),
            'rmse':float((d.square().mean()).sqrt()),'exact':bool(torch.equal(a,b))}


def main():
    ds=DATA/'datasets/faces500-moderate-uv-fixed-texture'
    train=DATA/'training/flux2_moderate_fixed_texture'
    records=sorted((ds/'samples').glob('*/complete.json'))
    for_split={split:[p.parent for p in records if json.loads(p.read_text())['split']==split] for split in ('train','holdout')}
    picks=for_split['train'][:2]+for_split['holdout'][:2]
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    checks=[]
    for p in picks:
        features=torch.load(p/'latents.pt',map_location='cpu',weights_only=True)
        for key,file in (('face_hair','portrait_hair.png'),('face_bare','portrait_bare.png')):
            reencoded=vae_encode(vae,Image.open(p/file).convert('RGB'))
            checks.append({'id':p.name,'kind':key,**describe(reencoded,features[key])})
        target=torch.load(ds/'latents_displacement'/(p.name+'.pt'),map_location='cpu',weights_only=True)
        reencoded=vae_encode(vae,Image.open(p/'displacement.png').convert('RGB'))
        checks.append({'id':p.name,'kind':'displacement',**describe(reencoded,target)})
    print('CACHE_CHECKS',json.dumps(checks),flush=True)
    if max(x['max_abs'] for x in checks) > .002:raise ValueError('Cached latent differs from source image')
    sample=picks[-1]; condition=torch.load(sample/'latents.pt',map_location='cpu',weights_only=True)['face_bare'].unsqueeze(0).to('cuda',torch.bfloat16)
    target=torch.load(ds/'latents_displacement'/(sample.name+'.pt'),map_location='cpu',weights_only=True).unsqueeze(0).to('cuda',torch.bfloat16)
    shape=target.shape; schedule=inference_schedule(MODEL,28,shape,'cuda')
    native=FlowMatchEulerDiscreteScheduler.from_pretrained(str(MODEL),subfolder='scheduler',local_files_only=True)
    sigmas=np.linspace(1.,1/28,28)
    if getattr(native.config,'use_flow_sigmas',False):sigmas=None
    native_ts,_=retrieve_timesteps(native,28,'cuda',sigmas=sigmas,mu=compute_empirical_mu(image_seq_len=shape[-2]*shape[-1],num_steps=28))
    if not torch.equal(schedule.timesteps,native_ts):raise ValueError('Training and native pipeline schedules differ')
    quant=(schedule.timesteps.to(torch.bfloat16)/1000).float()
    exact=schedule.sigmas[:-1].float().to('cuda')
    sigma_difference=describe(quant,exact)
    gen=torch.Generator(device='cuda').manual_seed(5321)
    noise=torch.randn(shape,device='cuda',dtype=target.dtype,generator=gen)
    # Test the Euler update against the exact flow target in float32.
    x0=target.float();eps=noise.float();oracle=[]
    for i in range(len(schedule.timesteps)):
        sigma=schedule.sigmas[i].float().to('cuda'); nxt=schedule.sigmas[i+1].float().to('cuda')
        actual=((1-sigma)*x0+sigma*eps)+(nxt-sigma)*(eps-x0)
        expected=(1-nxt)*x0+nxt*eps
        oracle.append(float((actual-expected).abs().max()))
    print('SCHEDULE_MATCH',len(native_ts),'sigma_rounding',json.dumps(sigma_difference),'oracle_max_abs',max(oracle),flush=True)
    if max(oracle)>2e-6:raise ValueError('Flow direction is inconsistent with Euler step')
    weights=sorted(train.glob('checkpoint-*.pt'))
    if not weights:raise RuntimeError('No completed checkpoint available')
    checkpoint=weights[-1]; state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=Flux2Transformer2DModel.from_pretrained(str(MODEL),subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    add_lora(model); model.eval()
    if set(state['lora'])!=set(get_peft_model_state_dict(model)):raise ValueError('Checkpoint keys differ')
    loaded=set_peft_model_state_dict(model,state['lora'])
    if loaded.unexpected_keys:raise ValueError('Unexpected LoRA checkpoint weights')
    step=state['step'];del state
    embeds=prompt_embedding(MODEL,True)
    t=torch.full((1,),.8,device='cuda',dtype=torch.bfloat16)
    with torch.inference_mode():
        actual=predict(model,noise,condition,t,embeds)
        packed=Flux2KleinPipeline._pack_latents(noise)
        packed_cond=Flux2KleinPipeline._pack_latents(condition)
        ids=Flux2KleinPipeline._prepare_latent_ids(noise).to('cuda')
        cond_ids=Flux2KleinPipeline._prepare_image_ids([condition]).to('cuda')
        text_ids=Flux2KleinPipeline._prepare_text_ids(embeds).to('cuda')
        native_pred=model(hidden_states=torch.cat((packed,packed_cond),dim=1),timestep=t,guidance=None,
                          encoder_hidden_states=embeds,txt_ids=text_ids,img_ids=torch.cat((ids,cond_ids),dim=1),return_dict=False)[0]
        expected=native_pred[:,:packed.shape[1]].transpose(1,2).reshape_as(noise)
    forward=describe(actual,expected)
    if not forward['exact']:raise ValueError(f'Predict differs from independently assembled transformer: {forward}')
    seed=773
    with torch.inference_mode():
        custom=infer(model,condition,embeds,MODEL,28,seed,True)
        initial=torch.randn(shape,device='cuda',dtype=condition.dtype,generator=torch.Generator(device='cuda').manual_seed(seed))
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(MODEL),subfolder='scheduler',local_files_only=True)
        pipe=Flux2KleinPipeline(scheduler=scheduler,vae=vae,text_encoder=None,tokenizer=None,transformer=model,is_distilled=False)
        pipe.set_progress_bar_config(disable=True)
        # Supply the exact cached portrait features to isolate the sampler.
        def cached_image_latents(self,images,batch_size,generator,device,dtype):
            return (Flux2KleinPipeline._pack_latents(condition),Flux2KleinPipeline._prepare_image_ids([condition]).to(device))
        pipe.prepare_image_latents=types.MethodType(cached_image_latents,pipe)
        captured={}
        def callback(pipeline,index,timestep,kwargs):
            if index==27:captured['last']=kwargs['latents'].detach().clone()
            return kwargs
        pipe(image=Image.open(sample/'portrait_bare.png').convert('RGB'),height=512,width=512,
             prompt_embeds=embeds,guidance_scale=1.,num_inference_steps=28,latents=initial,
             output_type='latent',callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents'])
    custom_packed=Flux2KleinPipeline._pack_latents(custom)
    sampler=describe(custom_packed,captured['last'])
    print('SAMPLER_COMPARISON',json.dumps(sampler),flush=True)
    if sampler['max_abs']>.001:raise ValueError('Preview sampler differs from standard FLUX.2 sampler')
    report={'checkpoint_step':step,'sample':sample.name,'cache_checks':checks,'native_schedule_exact':True,
            'sigma_bfloat16_rounding':sigma_difference,'oracle_flow_euler_max_abs':max(oracle),
            'native_transformer_forward':forward,'native_sampler_final_latent':sampler,
            'scope':'Same cached portrait latents and CFG 1; training samples either hair/bare and preview uses bare only. Correctness does not establish conditioning quality.'}
    out=DATA/'audits/nebius-inference-consistency.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print('AUDIT_COMPLETE',out,flush=True)


if __name__=='__main__':main()
