"""Encode 256px expression portraits and three-panel targets; never train."""
import argparse,json,os
from pathlib import Path
from facekit.paths import DATA,MODEL
os.environ.setdefault('HF_HOME',str(DATA/'cache/huggingface'))
os.environ.setdefault('XDG_CACHE_HOME',str(DATA/'cache'))

def main():
    import numpy as np,torch
    from PIL import Image
    from diffusers import AutoencoderKLFlux2
    from facekit.latent import vae_encode,vae_decode
    from facekit.uv import ATLAS,panel
    from facekit.bootstrap import checkpoint
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--dataset',type=Path,default=DATA/'datasets/faces1000-shape-expression-256');args=parser.parse_args()
    out=args.dataset
    if not args.preflight and not (out/'maps_complete.json').is_file():raise ValueError('Maps not ready')
    records=sorted((out/'samples').glob('*/complete.json'))
    if args.preflight:
        if len(records)<2:raise ValueError('Need two completed samples')
        records=records[:2]
    elif len(records)!=1000:raise ValueError('Expected 1000 records')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    checkpoint(vae_only=True)
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval()
    for i,meta in enumerate(records,1):
        p=meta.parent;file=p/'latents.pt'
        if file.exists():
            data=torch.load(file,map_location='cpu',weights_only=True)
            if tuple(data['condition'].shape)==(128,16,16) and tuple(data['maps'].shape)==(128,16,48):continue
        condition=vae_encode(vae,Image.open(p/'portrait_expression_hair.png'))
        target=vae_encode(vae,Image.open(p/'maps.png'))
        if tuple(condition.shape)!=(128,16,16) or tuple(target.shape)!=(128,16,48) or not torch.isfinite(condition).all() or not torch.isfinite(target).all():raise ValueError(p)
        tmp=file.with_suffix('.pt.tmp');torch.save({'condition':condition,'maps':target},tmp);tmp.replace(file)
        if i%50==0:print('ENCODED',i,flush=True)
    if args.preflight:print('PREFLIGHT_LATENTS_READY',len(records),flush=True);return
    uv=np.load(out/'uv_mapping.npz')
    coverage=np.asarray(panel(Image.new('RGB',(ATLAS,ATLAS),'white'),uv['mask'],uv['bbox']).resize((256,256),Image.Resampling.LANCZOS))[:,:,0]>=254
    if not coverage.any():raise ValueError('Empty face UV coverage')
    checks=[]
    for i in (0,199,399,599,799,999):
        p=records[i].parent;data=torch.load(p/'latents.pt',map_location='cpu',weights_only=True)
        img=np.asarray(vae_decode(vae,data['maps'].unsqueeze(0))[0],dtype=np.float32)
        truth=np.asarray(Image.open(p/'maps.png').convert('RGB'),dtype=np.float32)
        mse=[float(np.mean(((img[:,j*256:(j+1)*256]-truth[:,j*256:(j+1)*256])**2)[coverage])) for j in range(3)]
        if not np.isfinite(mse).all() or max(mse)>500:raise ValueError(f'VAE roundtrip {p.name}: {mse}')
        portrait=np.asarray(Image.open(p/'portrait_expression_hair.png').convert('RGB'),dtype=np.float32)
        portrait_dec=np.asarray(vae_decode(vae,data['condition'].unsqueeze(0))[0],dtype=np.float32)
        foreground=np.any(portrait>4,axis=2)
        portrait_mse=float(np.mean(((portrait_dec-portrait)**2)[foreground]))
        if not np.isfinite(portrait_mse) or portrait_mse>500:raise ValueError(f'Portrait VAE roundtrip {p.name}: {portrait_mse}')
        checks.append({'sample':p.name,'face_uv_panel_mse':mse,'portrait_foreground_mse':portrait_mse})
    manifest={'count':len(records),'condition_shape':[128,16,16],'target_shape':[128,16,48],'vae_roundtrips':checks}
    tmp=out/'dataset_complete.json.tmp';tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(out/'dataset_complete.json')
    print('DATASET_READY',json.dumps(manifest),flush=True)

if __name__=='__main__':main()
