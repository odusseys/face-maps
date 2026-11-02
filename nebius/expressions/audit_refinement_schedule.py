"""Compare shorter latent-refinement schedules with the accepted 80-step result."""
import json,time
import numpy as np,torch
from PIL import Image
from diffusers import AutoencoderKLFlux2,Flux2KleinPipeline
from facekit.paths import DATA,MODEL
from facekit.latent import vae_encode,vae_decode
from nebius.expressions.refine_latents import map_weight

def main():
    ds=DATA/'datasets/faces1000-canonical-head-256';uv=np.load(ds/'uv_mapping.npz');vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval().requires_grad_(False)
    mean=vae.bn.running_mean.view(1,-1,1,1).to('cuda');std=(vae.bn.running_var.view(1,-1,1,1).to('cuda')+vae.config.batch_norm_eps).sqrt();weight=map_weight(uv).to('cuda');results=[]
    for ident in ('00000','00001'):
        p=ds/'samples'/ident;image=Image.open(p/'maps.png').convert('RGB');target=torch.from_numpy(np.asarray(image,np.float32).copy()).permute(2,0,1).unsqueeze(0).to('cuda')/127.5-1
        z0=vae_encode(vae,image).unsqueeze(0).to('cuda',torch.float32);saved=json.loads((p/'refinement.json').read_text())
        def decode(z):return vae.decode(Flux2KleinPipeline._unpatchify_latents(z*std+mean),return_dict=False)[0]
        for mode in ('constant','decay','fast_decay'):
            z=torch.nn.Parameter(z0.clone());opt=torch.optim.Adam([z],lr=.04);start=time.monotonic()
            for step in range(40):
                if mode=='decay':opt.param_groups[0]['lr']=.04 if step<15 else (.02 if step<25 else .01)
                if mode=='fast_decay':opt.param_groups[0]['lr']=.06 if step<10 else (.025 if step<20 else .01)
                opt.zero_grad(set_to_none=True);im=decode(z);loss=((im-target).square()*weight).mean()+.0001*(z-z0).square().mean();loss.backward();opt.step()
            with torch.no_grad():score=float(((decode(z.to(torch.float16).float())-target).square()*weight).mean())
            result={'sample':ident,'schedule':mode,'seconds':time.monotonic()-start,'loss_ratio_to_accepted_80':score/saved['maps_after'],'loss':score,'accepted_loss':saved['maps_after']};results.append(result)
            print('SCHEDULE_RESULT',json.dumps(result),flush=True)
    out=DATA/'runs/head-refinement-schedule-study.json';out.write_text(json.dumps(results,indent=2));print('SCHEDULE_STUDY_READY',out,flush=True)

if __name__=='__main__':main()
