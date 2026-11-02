"""Audit VAE target-map round trips and their reconstructed meshes."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from PIL import Image
from diffusers import AutoencoderKLFlux2
from facekit.paths import DATA,MODEL
from facekit.bootstrap import assets
from facekit.latent import vae_encode,vae_decode
from nebius.expressions.debug import render_pair

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,default=DATA/'runs/expression-canonical-pilot')
    parser.add_argument('--output',type=Path,default=DATA/'runs/expression-canonical-vae-audit')
    parser.add_argument('--samples',nargs='+',default=['00000','00001','00002','00003','00004','00005']);args=parser.parse_args()
    ds=args.dataset;out=args.output;out.mkdir(parents=True,exist_ok=True);cfg=json.loads((ds/'dataset.json').read_text())
    ids=np.unique(np.load(ds/'reference.npz')['triangle_vertices']);blender,_,_=assets()
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval()
    results=[]
    for ident in args.samples:
        sample=ds/'samples'/ident;target=Image.open(sample/'maps.png').convert('RGB')
        latent=vae_encode(vae,target);decoded=vae_decode(vae,latent.unsqueeze(0))[0];work=out/ident;work.mkdir(parents=True,exist_ok=True);decoded.save(work/'vae-maps.png')
        render_pair(ds,sample,decoded,blender,work,cfg)
        a=np.load(work/'gt_neutral.npy');b=np.load(work/'pred_neutral.npy');c=np.load(work/'gt_expression.npy');d=np.load(work/'pred_expression.npy')
        neutral=np.linalg.norm(a[ids]-b[ids],axis=1)*1000;expression=np.linalg.norm((c-a)[ids]-(d-b)[ids],axis=1)*1000
        result={'sample':ident,'neutral_vertex_error_mm_mean':float(neutral.mean()),'expression_vertex_error_mm_mean':float(expression.mean()),'expression_vertex_error_mm_p95':float(np.percentile(expression,95))}
        results.append(result);print('VAE_INVERSE_SAMPLE',json.dumps(result),flush=True)
    (out/'audit.json').write_text(json.dumps(results,indent=2));print('VAE_INVERSE_READY',out,flush=True)

if __name__=='__main__':main()
