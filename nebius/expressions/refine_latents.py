"""Optimize VAE latents for faithful 256px target or conditioning images."""
import numpy as np,torch
from PIL import Image
from diffusers import Flux2KleinPipeline
from facekit.latent import vae_encode

STEPS=80
LEARNING_RATE=.04
REGULARIZATION=.0001

def portrait_weight(images):
    values=[]
    for image in images:
        pixels=np.asarray(image.convert('RGB'));values.append(.1+.9*np.any(pixels>8,axis=2).astype(np.float32))
    return torch.from_numpy(np.stack(values))[:,None]

def map_weight(uv):
    from facekit.uv import ATLAS,panel
    coverage=np.asarray(panel(Image.new('RGB',(ATLAS,ATLAS),'white'),uv['mask'],uv['bbox']).resize((256,256),Image.Resampling.LANCZOS),np.float32)[:,:,0]/255
    return torch.from_numpy(.1+.9*np.tile(coverage,(1,3)))[None,None]

def refine(vae,images,weights,steps=STEPS):
    if not images:raise ValueError('No images to refine')
    if any(im.size!=images[0].size for im in images):raise ValueError('Image sizes must match')
    vae.requires_grad_(False)
    target=torch.stack([torch.from_numpy(np.asarray(im.convert('RGB'),np.float32).copy()).permute(2,0,1)/127.5-1 for im in images]).to(vae.device)
    weights=weights.to(vae.device)
    if weights.shape[0] not in (1,len(images)) or weights.shape[-2:]!=target.shape[-2:]:raise ValueError('Invalid reconstruction weight shape')
    original=torch.stack([vae_encode(vae,im) for im in images]).to(vae.device,torch.float32)
    z=torch.nn.Parameter(original.clone());optimizer=torch.optim.Adam([z],lr=LEARNING_RATE)
    mean=vae.bn.running_mean.view(1,-1,1,1).to(z);std=(vae.bn.running_var.view(1,-1,1,1).to(z)+vae.config.batch_norm_eps).sqrt()
    def image_from_latent(x):return vae.decode(Flux2KleinPipeline._unpatchify_latents(x*std+mean),return_dict=False)[0]
    with torch.enable_grad():
        with torch.no_grad():before=((image_from_latent(original)-target).square()*weights).mean(dim=(1,2,3))
        best=original.clone();best_loss=before.clone()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True);image=image_from_latent(z)
            per_image=((image-target).square()*weights).mean(dim=(1,2,3))
            if step and step%10==0:
                with torch.no_grad():
                    better=per_image<best_loss;best[better]=z.detach()[better];best_loss[better]=per_image[better]
            loss=per_image.mean()+REGULARIZATION*(z-original).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite latent refinement loss')
            loss.backward();optimizer.step()
        with torch.no_grad():
            final=((image_from_latent(z.detach())-target).square()*weights).mean(dim=(1,2,3));better=final<best_loss
            best[better]=z.detach()[better];result=best.to(torch.float16)
            after=((image_from_latent(result.float())-target).square()*weights).mean(dim=(1,2,3))
            worse=after>before*1.001+1e-9
            result[worse]=original.to(torch.float16)[worse];after[worse]=before[worse]
    if not torch.isfinite(result).all():raise ValueError('Nonfinite optimized latent')
    stats={'before':before.detach().cpu().tolist(),'after':after.detach().cpu().tolist(),
           'delta_rms':((result.float()-original).square().mean(dim=(1,2,3)).sqrt()).detach().cpu().tolist(),
           'unchanged_count':int(((result.float()-original).square().mean(dim=(1,2,3))==0).sum())}
    return result.cpu(),stats
