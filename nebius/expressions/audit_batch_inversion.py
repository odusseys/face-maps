"""Benchmark latent refinement on diverse head maps without training."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from diffusers import AutoencoderKLFlux2,Flux2KleinPipeline
from facekit.paths import DATA,MODEL
from facekit.bootstrap import assets
from facekit.latent import vae_encode,vae_decode
from facekit.uv import ATLAS,panel
from nebius.expressions.debug import render_pair

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,default=DATA/'runs/canonical-uv-coverage-audit/head-island-pilot')
    parser.add_argument('--output',type=Path,default=DATA/'runs/head-vae-batch-study');args=parser.parse_args()
    ds=args.dataset;out=args.output;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ds/'dataset.json').read_text());ref=np.load(ds/'reference.npz');ids=np.unique(ref['triangle_vertices'])
    uv=np.load(ds/'uv_mapping.npz');coverage=np.asarray(panel(Image.new('RGB',(ATLAS,ATLAS),'white'),uv['mask'],uv['bbox']).resize((256,256),Image.Resampling.LANCZOS),np.float32)[:,:,0]/255
    weight=torch.from_numpy(.1+.9*np.tile(coverage,(1,3))).to('cuda')[None,None]
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval();vae.requires_grad_(False)
    mean=vae.bn.running_mean.view(1,-1,1,1).to('cuda');std=(vae.bn.running_var.view(1,-1,1,1).to('cuda')+vae.config.batch_norm_eps).sqrt()
    blender,_,_=assets();results=[];samples=['00000','00001','00002','00003','00114','00390','00420','00831']
    for start in range(0,len(samples),4):
        names=samples[start:start+4];images=[Image.open(ds/'samples'/n/'maps.png').convert('RGB') for n in names]
        target=torch.stack([torch.from_numpy(np.asarray(im,np.float32).copy()).permute(2,0,1)/127.5-1 for im in images]).to('cuda')
        z0=torch.stack([vae_encode(vae,im) for im in images]).to('cuda',torch.float32);z=torch.nn.Parameter(z0.clone());opt=torch.optim.Adam([z],lr=.04)
        torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        for step in range(80):
            opt.zero_grad(set_to_none=True);image=vae.decode(Flux2KleinPipeline._unpatchify_latents(z*std+mean),return_dict=False)[0]
            loss=((image-target).square()*weight).mean()+.0001*(z-z0).square().mean();loss.backward();opt.step()
        print('BATCH_BENCHMARK',names,'seconds',round(time.monotonic()-t,2),'peak_gib',round(torch.cuda.max_memory_allocated()/1024**3,2),flush=True)
        for i,ident in enumerate(names):
            for kind,latent in [('baseline',z0[i:i+1]),('refined',z[i:i+1].detach().to(torch.float16))]:
                decoded=vae_decode(vae,latent)[0];work=out/ident/kind;render_pair(ds,ds/'samples'/ident,decoded,blender,work,cfg)
                a=np.load(work/'gt_neutral.npy');b=np.load(work/'pred_neutral.npy');c=np.load(work/'gt_expression.npy');d=np.load(work/'pred_expression.npy')
                shape=np.linalg.norm(a[ids]-b[ids],axis=1)*1000;expr=np.linalg.norm((c-a)[ids]-(d-b)[ids],axis=1)*1000
                result={'sample':ident,'kind':kind,'shape_error_mm_mean':float(shape.mean()),'expression_error_mm_mean':float(expr.mean()),
                  'expression_error_mm_p95':float(np.percentile(expr,95)),'map_mae':float(np.abs(np.asarray(decoded,np.float32)-np.asarray(images[i],np.float32)).mean())}
                results.append(result);print('BATCH_RESULT',json.dumps(result),flush=True)
        del z,opt,target,z0;torch.cuda.empty_cache()
    (out/'results.json').write_text(json.dumps(results,indent=2));print('BATCH_STUDY_READY',out,flush=True)

if __name__=='__main__':main()
