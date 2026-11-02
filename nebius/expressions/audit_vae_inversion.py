"""Try optimizing VAE latents at the existing 256px target resolution."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from diffusers import AutoencoderKLFlux2,Flux2KleinPipeline
from facekit.paths import DATA,MODEL
from facekit.bootstrap import assets
from facekit.latent import vae_encode,vae_decode
from facekit.uv import ATLAS,panel
from nebius.expressions.debug import render_pair

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,default=DATA/'datasets/faces1000-canonical-expression-256')
    parser.add_argument('--output',type=Path,default=DATA/'runs/canonical-vae-latent-optimization')
    parser.add_argument('--factor',type=float,default=1.0);parser.add_argument('--steps',type=int,default=80)
    parser.add_argument('--regularization',type=float,default=.0001);args=parser.parse_args()
    if args.factor not in (1.0,1.5,2.0) or args.steps not in (80,160,320):raise ValueError('Unsupported study settings')
    ds=args.dataset;out=args.output;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ds/'dataset.json').read_text());ids=np.unique(np.load(ds/'reference.npz')['triangle_vertices']);uv=np.load(ds/'uv_mapping.npz')
    coverage=np.asarray(panel(Image.new('RGB',(ATLAS,ATLAS),'white'),uv['mask'],uv['bbox']).resize((256,256),Image.Resampling.LANCZOS),np.float32)[:,:,0]/255
    coverage=np.tile(coverage,(1,3));size=(int(768*args.factor),int(256*args.factor))
    if args.factor!=1:coverage=np.asarray(Image.fromarray(coverage).resize(size,Image.Resampling.BILINEAR))
    weight=torch.from_numpy(.1+.9*coverage).to('cuda')[None,None]
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval();vae.requires_grad_(False)
    mean=vae.bn.running_mean.view(1,-1,1,1).to('cuda');std=(vae.bn.running_var.view(1,-1,1,1).to('cuda')+vae.config.batch_norm_eps).sqrt()
    blender,_,_=assets();results=[];samples=['00000','00114'];grid=Image.new('RGB',(256*4,284*len(samples)),'black');draw=ImageDraw.Draw(grid)
    for row,ident in enumerate(samples):
        p=ds/'samples'/ident;target=Image.open(p/'maps.png').convert('RGB')
        model_target=target if args.factor==1 else target.resize(size,Image.Resampling.LANCZOS)
        target_t=torch.from_numpy(np.asarray(model_target,np.float32).copy()).permute(2,0,1).unsqueeze(0).to('cuda')/127.5-1
        z0=vae_encode(vae,model_target).unsqueeze(0).to('cuda',dtype=torch.float32);z=torch.nn.Parameter(z0.clone());optimizer=torch.optim.Adam([z],lr=.04)
        snapshots={0:z0.detach().clone()};start=time.monotonic()
        requested=(10,30,args.steps)
        for step in range(1,args.steps+1):
            optimizer.zero_grad(set_to_none=True)
            raw=Flux2KleinPipeline._unpatchify_latents(z*std+mean);decoded=vae.decode(raw,return_dict=False)[0]
            loss=((decoded-target_t).square()*weight).mean()+args.regularization*(z-z0).square().mean()
            loss.backward();optimizer.step()
            if step in requested:
                snapshots[step]=z.detach().clone();print('OPTIMIZATION',ident,step,'loss',float(loss.detach()),'seconds',round(time.monotonic()-start,2),flush=True)
        faces=[]
        for step in (0,*requested):
            latent=snapshots[step];decoded=vae_decode(vae,latent)[0];work=out/ident/f'step-{step:03d}';work.mkdir(parents=True,exist_ok=True);decoded.save(work/'maps.png')
            if args.factor!=1:decoded=decoded.resize((768,256),Image.Resampling.LANCZOS)
            rendered=render_pair(ds,p,decoded,blender,work,cfg);faces.append(rendered[3])
            a=np.load(work/'gt_neutral.npy');b=np.load(work/'pred_neutral.npy');c=np.load(work/'gt_expression.npy');d=np.load(work/'pred_expression.npy')
            shape=np.linalg.norm(a[ids]-b[ids],axis=1)*1000;expr=np.linalg.norm((c-a)[ids]-(d-b)[ids],axis=1)*1000
            result={'sample':ident,'steps':step,'factor':args.factor,'shape_error_mm_mean':float(shape.mean()),'expression_error_mm_mean':float(expr.mean()),
                'expression_error_mm_p95':float(np.percentile(expr,95)),'latent_delta_rms':float((latent-z0).square().mean().sqrt()),
                'map_pixel_mae':float(np.abs(np.asarray(decoded,np.float32)-np.asarray(target,np.float32)).mean())}
            results.append(result);print('INVERSION_RESULT',json.dumps(result),flush=True)
        for col,(label,image) in enumerate(zip(['VAE original','Optimized 10','Optimized 30',f'Optimized {args.steps}'],faces)):
            draw.text((col*256+4,row*284+5),ident+' '+label,fill='white');grid.paste(image,(col*256,row*284+28))
    grid.save(out/'comparison.png');(out/'results.json').write_text(json.dumps(results,indent=2));print('INVERSION_STUDY_READY',out,flush=True)

if __name__=='__main__':main()
