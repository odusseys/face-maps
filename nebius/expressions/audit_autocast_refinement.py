"""Test faster mixed-precision latent refinement against FP32 decoding."""
import json,time
import numpy as np,torch
from PIL import Image
from diffusers import AutoencoderKLFlux2,Flux2KleinPipeline
from facekit.paths import DATA,MODEL
from facekit.bootstrap import assets
from facekit.latent import vae_encode,vae_decode
from nebius.expressions.refine_latents import map_weight
from nebius.expressions.debug import render_pair

def main():
    ds=DATA/'datasets/faces1000-canonical-head-256';out=DATA/'runs/head-mixed-precision-study';out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ds/'dataset.json').read_text());ids=np.unique(np.load(ds/'reference.npz')['triangle_vertices']);uv=np.load(ds/'uv_mapping.npz')
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval().requires_grad_(False)
    mean=vae.bn.running_mean.view(1,-1,1,1).to('cuda');std=(vae.bn.running_var.view(1,-1,1,1).to('cuda')+vae.config.batch_norm_eps).sqrt();weights=map_weight(uv).to('cuda');blender,_,_=assets();results=[]
    for ident in ('00000','00114'):
        p=ds/'samples'/ident;im=Image.open(p/'maps.png').convert('RGB');pixels=np.asarray(im,np.float32).copy()
        target=torch.from_numpy(pixels).permute(2,0,1).unsqueeze(0).to('cuda')/127.5-1
        z0=vae_encode(vae,im).unsqueeze(0).to('cuda',torch.float32);z=torch.nn.Parameter(z0.clone());opt=torch.optim.Adam([z],lr=.04);start=time.monotonic()
        for _ in range(80):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                image=vae.decode(Flux2KleinPipeline._unpatchify_latents(z*std+mean),return_dict=False)[0]
                loss=((image.float()-target).square()*weights).mean()+.0001*(z-z0).square().mean()
            loss.backward();opt.step()
        elapsed=time.monotonic()-start;decoded=vae_decode(vae,z.detach().to(torch.float16))[0];work=out/ident;render_pair(ds,p,decoded,blender,work,config)
        a=np.load(work/'gt_neutral.npy');b=np.load(work/'pred_neutral.npy');c=np.load(work/'gt_expression.npy');d=np.load(work/'pred_expression.npy')
        shape=np.linalg.norm(a[ids]-b[ids],axis=1)*1000;expr=np.linalg.norm((c-a)[ids]-(d-b)[ids],axis=1)*1000
        result={'sample':ident,'seconds':elapsed,'shape_error_mm_mean':float(shape.mean()),'expression_error_mm_mean':float(expr.mean()),
          'expression_error_mm_p95':float(np.percentile(expr,95)),'map_pixel_mae':float(np.abs(np.asarray(decoded,np.float32)-pixels).mean())}
        results.append(result);print('MIXED_PRECISION_REFINEMENT',json.dumps(result),flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2));print('MIXED_PRECISION_STUDY_READY',out,flush=True)

if __name__=='__main__':main()
