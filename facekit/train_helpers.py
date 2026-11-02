"""FLUX.2 image-conditioning, flow objective, and debug sampling helpers."""
import numpy as np
import torch
from diffusers import Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
from .prompt_cache import PAIRED_PROMPT,load_prompt

PROMPT=PAIRED_PROMPT


def prompt_embedding(model, displacement_only=False):
    return load_prompt(model,displacement_only,device='cuda')


def predict(model,noisy,condition,t,embeds):
    pipe=Flux2KleinPipeline; b=noisy.shape[0]
    target_ids=pipe._prepare_latent_ids(noisy).to(noisy.device)
    cond_ids=pipe._prepare_image_ids([condition[:1]]).to(noisy.device).expand(b,-1,-1)
    z=pipe._pack_latents(noisy); cond=pipe._pack_latents(condition)
    text=embeds.expand(b,-1,-1)
    text_ids=pipe._prepare_text_ids(text).to(noisy.device)
    result=model(hidden_states=torch.cat([z,cond],1),timestep=t, guidance=None,
                 encoder_hidden_states=text,txt_ids=text_ids,img_ids=torch.cat([target_ids,cond_ids],1),return_dict=False)[0]
    return result[:,:z.shape[1]].permute(0,2,1).reshape_as(noisy)


def inference_schedule(model_path, steps, shape, device):
    """Build the same shifted scheduler used by the preview sampler."""
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(model_path),subfolder='scheduler')
    sigmas=np.linspace(1.0,1/steps,steps)
    if getattr(scheduler.config,'use_flow_sigmas',False): sigmas=None
    mu=compute_empirical_mu(image_seq_len=shape[-1]*shape[-2],num_steps=steps)
    scheduler.set_timesteps(sigmas=sigmas,num_inference_steps=steps,device=device,mu=mu)
    return scheduler


def flow_loss(model,target,condition,embeds,timesteps):
    # Uniformly sample the exact shifted timesteps at which inference evaluates
    # the model, including the pure-noise first step but excluding terminal zero.
    b=target.shape[0]
    index=torch.randint(len(timesteps),(b,),device=target.device)
    t=timesteps[index].to(target.dtype)/1000
    noise=torch.randn_like(target); sigma=t[:,None,None,None]
    noisy=(1-sigma)*target+sigma*noise
    prediction=predict(model,noisy,condition,t,embeds)
    return torch.nn.functional.mse_loss(prediction.float(),(noise-target).float())


@torch.no_grad()
def infer(model,condition,embeds,model_path,steps,seed,displacement_only=False):
    shape=(condition.shape[0],condition.shape[1],condition.shape[2],condition.shape[3]*(1 if displacement_only else 2))
    rng=torch.Generator(device=condition.device).manual_seed(seed)
    z=torch.randn(shape,device=condition.device,dtype=condition.dtype,generator=rng)
    scheduler=inference_schedule(model_path,steps,shape,condition.device)
    for t in scheduler.timesteps:
        tv=t.expand(z.shape[0]).to(z.dtype)/1000
        pred=predict(model,z,condition,tv,embeds)
        z=scheduler.step(pred,t,z,return_dict=False)[0].to(condition.dtype)
    return z
