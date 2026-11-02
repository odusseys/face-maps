"""Measure refinement of the 256px conditioning portrait VAE latent."""
import json,time
import numpy as np,torch
from PIL import Image,ImageDraw
from diffusers import AutoencoderKLFlux2,Flux2KleinPipeline
from facekit.paths import DATA,MODEL
from facekit.latent import vae_encode,vae_decode

def main():
    ds=DATA/'datasets/faces1000-canonical-expression-256';out=DATA/'runs/portrait-vae-inversion-study';out.mkdir(parents=True,exist_ok=True)
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval();vae.requires_grad_(False)
    mean=vae.bn.running_mean.view(1,-1,1,1).to('cuda');std=(vae.bn.running_var.view(1,-1,1,1).to('cuda')+vae.config.batch_norm_eps).sqrt()
    names=['00000','00114','00390'];grid=Image.new('RGB',(256*3,284*len(names)),'black');draw=ImageDraw.Draw(grid);results=[]
    for row,ident in enumerate(names):
        im=Image.open(ds/'samples'/ident/'portrait_expression_hair.png').convert('RGB');rgb=np.asarray(im,np.float32).copy();fg=np.any(rgb>8,axis=2)
        target=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to('cuda')/127.5-1
        weight=torch.from_numpy(.1+.9*fg.astype(np.float32)).to('cuda')[None,None]
        z0=vae_encode(vae,im).unsqueeze(0).to('cuda',torch.float32);z=torch.nn.Parameter(z0.clone());opt=torch.optim.Adam([z],lr=.04);start=time.monotonic()
        for step in range(80):
            opt.zero_grad(set_to_none=True);decoded=vae.decode(Flux2KleinPipeline._unpatchify_latents(z*std+mean),return_dict=False)[0]
            loss=((decoded-target).square()*weight).mean()+.0001*(z-z0).square().mean();loss.backward();opt.step()
        images=[im,vae_decode(vae,z0)[0],vae_decode(vae,z.detach().to(torch.float16))[0]]
        for col,(label,image) in enumerate(zip(('Input','VAE original','VAE refined'),images)):
            draw.text((col*256+4,row*284+5),ident+' '+label,fill='white');grid.paste(image,(col*256,row*284+28))
            if col:results.append({'sample':ident,'type':label,'foreground_mse':float(np.mean((np.asarray(image,np.float32)[fg]-rgb[fg])**2))})
        print('PORTRAIT_INVERSION',ident,json.dumps(results[-2:]),'seconds',round(time.monotonic()-start,2),flush=True)
    grid.save(out/'comparison.png');(out/'results.json').write_text(json.dumps(results,indent=2));print('PORTRAIT_INVERSION_READY',out,flush=True)

if __name__=='__main__':main()
