"""FLUX.2 VAE latents: mode, patchification, then model batch normalization."""
import torch
import numpy as np
from PIL import Image


def vae_encode(vae, image):
    from diffusers import Flux2KleinPipeline
    x = torch.from_numpy(np.asarray(image.convert('RGB'), dtype=np.float32).copy())
    x = (x.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1).to(vae.device, vae.dtype)
    with torch.inference_mode():
        z = vae.encode(x).latent_dist.mode()
        z = Flux2KleinPipeline._patchify_latents(z)
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(z)
        std = (vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).sqrt().to(z)
        return ((z - mean) / std).squeeze(0).cpu().to(torch.float16)


def vae_decode(vae, z):
    from diffusers import Flux2KleinPipeline
    z = z.to(vae.device, vae.dtype)
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(z)
    std = (vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).sqrt().to(z)
    with torch.inference_mode():
        z = Flux2KleinPipeline._unpatchify_latents(z * std + mean)
        x = vae.decode(z, return_dict=False)[0].float().clamp(-1, 1)
    return [Image.fromarray(np.uint8(((v.permute(1,2,0).cpu().numpy()+1)*127.5).round())) for v in x]
