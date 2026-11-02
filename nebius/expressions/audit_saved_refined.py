"""Render actual saved refined latents and compare them with true geometry."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from diffusers import AutoencoderKLFlux2
from facekit.paths import DATA,MODEL
from facekit.bootstrap import assets
from facekit.latent import vae_decode
from nebius.expressions.debug import render_pair

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--samples',nargs='+',default=['00000','00001','00002','00003','00114','00390','00420','00831'])
    parser.add_argument('--output',type=Path,default=DATA/'runs/final-refined-audit');args=parser.parse_args()
    ds=DATA/'datasets/faces1000-canonical-head-256';out=args.output;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ds/'dataset.json').read_text());ref=np.load(ds/'reference.npz');ids=np.unique(ref['triangle_vertices']);blender,_,_=assets()
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval()
    grid=Image.new('RGB',(256*4,284*len(args.samples)),'black');draw=ImageDraw.Draw(grid);results=[]
    for row,ident in enumerate(args.samples):
        sample=ds/'samples'/ident;z=torch.load(sample/'latents.pt',map_location='cpu',weights_only=True)
        target=vae_decode(vae,z['maps'].unsqueeze(0))[0];condition=vae_decode(vae,z['condition'].unsqueeze(0))[0]
        work=out/ident;faces=render_pair(ds,sample,target,blender,work,cfg)
        a=np.load(work/'gt_neutral.npy');b=np.load(work/'pred_neutral.npy');c=np.load(work/'gt_expression.npy');d=np.load(work/'pred_expression.npy')
        shape=np.linalg.norm(a[ids]-b[ids],axis=1)*1000;expr=np.linalg.norm((c-a)[ids]-(d-b)[ids],axis=1)*1000
        stats=json.loads((sample/'refinement.json').read_text());result={'sample':ident,'shape_error_mm_mean':float(shape.mean()),
          'expression_error_mm_mean':float(expr.mean()),'expression_error_mm_p95':float(np.percentile(expr,95)),
          'condition_loss_ratio':stats['condition_after']/stats['condition_before'],'maps_loss_ratio':stats['maps_after']/stats['maps_before']}
        if not np.isfinite(list(result.values())[1:]).all():raise ValueError(f'Nonfinite results {ident}')
        results.append(result);print('SAVED_REFINEMENT_AUDIT',json.dumps(result),flush=True)
        for col,(label,image) in enumerate(zip(['Input','Refined input VAE','GT expression','Refined inverse'],[Image.open(sample/'portrait_expression_hair.png'),condition,faces[2],faces[3]])):
            x=col*256;y=row*284;draw.text((x+3,y+5),ident+' '+label,fill='white');grid.paste(image.convert('RGB'),(x,y+28))
    grid.save(out/'comparison.png');(out/'results.json').write_text(json.dumps(results,indent=2));print('SAVED_REFINEMENT_AUDIT_READY',out,flush=True)

if __name__=='__main__':main()
