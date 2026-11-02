#!/usr/bin/env python3
"""Resume encoding both target variants for the scale-normalized face dataset."""
import argparse
import json
import os
from pathlib import Path

from facekit.paths import ROOT, DATA, MODEL
os.environ.setdefault('HF_HOME',str(DATA/'cache'/'huggingface'))
os.environ.setdefault('XDG_CACHE_HOME',str(DATA/'cache'))


def atomic_json(path, value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def valid_tensor(value, shape):
    import torch
    return isinstance(value,torch.Tensor) and tuple(value.shape)==shape and bool(torch.isfinite(value).all())


def verify_roundtrips(vae,dataset,records):
    """Decode six encoded pairs and check both panels against their inputs."""
    import numpy as np
    import torch
    from PIL import Image
    from facekit.latent import vae_decode
    from facekit.uv import ATLAS,panel
    uv=np.load(dataset/'uv_mapping.npz')
    mask=np.asarray(panel(Image.new('RGB',(ATLAS,ATLAS),'white'),uv['mask'],uv['bbox']))[:,:,0]>=254
    results=[]
    for index in sorted(set([0,99,199,299,399,len(records)-1])):
        sample=records[index].parent
        single=torch.load(dataset/'latents_displacement'/(sample.name+'.pt'),map_location='cpu',weights_only=True)
        paired=torch.load(sample/'latents.pt',map_location='cpu',weights_only=True)['maps']
        single_dec=np.asarray(vae_decode(vae,single.unsqueeze(0))[0],dtype=np.float32)
        paired_dec=np.asarray(vae_decode(vae,paired.unsqueeze(0))[0],dtype=np.float32)
        single_gt=np.asarray(Image.open(sample/'displacement.png').convert('RGB'),dtype=np.float32)
        paired_gt=np.asarray(Image.open(sample/'maps.png').convert('RGB'),dtype=np.float32)
        scores={'id':sample.name,'displacement_mse':float(np.mean((single_dec[mask]-single_gt[mask])**2)),
                'paired_texture_mse':float(np.mean((paired_dec[:,:512][mask]-paired_gt[:,:512][mask])**2)),
                'paired_displacement_mse':float(np.mean((paired_dec[:,512:][mask]-paired_gt[:,512:][mask])**2))}
        if max(scores[k] for k in ('displacement_mse','paired_texture_mse','paired_displacement_mse'))>500:
            raise RuntimeError(f'VAE round-trip verification failed: {scores}')
        results.append(scores);print('ROUNDTRIP',json.dumps(scores),flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DATA/'datasets'/'faces500-similarity-linear-q97')
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args();dataset=args.dataset.resolve()
    if not dataset.is_relative_to(ROOT.resolve()) or args.limit<0:parser.error('Invalid shared dataset or limit')
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLFlux2
    from facekit.bootstrap import checkpoint
    from facekit.latent import vae_encode
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    config_path=dataset/'dataset.json';config=json.loads(config_path.read_text())
    if config.get('geometry_alignment')!='rigid alignment plus optimal uniform scale to reference face':
        raise ValueError('Expected a similarity-normalized dataset')
    source=Path(config['normalization_source']);records=sorted((dataset/'samples').glob('*/complete.json'))
    state_path=dataset/'dataset_complete.json';state=json.loads(state_path.read_text())
    if len(records)!=state['count'] or len(records)!=500:raise ValueError('Expected a complete 500-sample dataset')
    if args.limit:records=records[:args.limit]
    output=dataset/'latents_displacement';output.mkdir(parents=True,exist_ok=True)
    checkpoint(vae_only=True)
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval()
    for index,meta_path in enumerate(records,1):
        sample=meta_path.parent;original=source/'samples'/sample.name
        sample_path=sample/'latents.pt';features=torch.load(sample_path,map_location='cpu',weights_only=True)
        original_features=torch.load(original/'latents.pt',map_location='cpu',weights_only=True)
        for key in ('face_bare','face_hair'):
            if not valid_tensor(features.get(key),(128,32,32)) or not torch.equal(features[key],original_features[key]):
                raise ValueError(f'Invalid or changed portrait latent: {sample.name} {key}')
        del original_features
        if not valid_tensor(features.get('maps'),(128,32,64)):
            img=Image.open(sample/'maps.png').convert('RGB')
            if img.size!=(1024,512):raise ValueError(f'Invalid paired image size: {sample.name}')
            features['maps']=vae_encode(vae,img)
            if not valid_tensor(features['maps'],(128,32,64)):raise ValueError(f'Invalid paired latent: {sample.name}')
            temp=sample_path.with_suffix('.pt.tmp');torch.save(features,temp);temp.replace(sample_path)
        path=output/(sample.name+'.pt')
        saved=torch.load(path,map_location='cpu',weights_only=True) if path.exists() else None
        if not valid_tensor(saved,(128,32,32)):
            img=Image.open(sample/'displacement.png').convert('RGB')
            if img.size!=(512,512):raise ValueError(f'Invalid displacement image size: {sample.name}')
            latent=vae_encode(vae,img)
            if not valid_tensor(latent,(128,32,32)):raise ValueError(f'Invalid displacement latent: {sample.name}')
            temp=path.with_suffix('.pt.tmp');torch.save(latent,temp);temp.replace(path)
        meta=json.loads(meta_path.read_text());meta['paired_map_latent_available']=True;atomic_json(meta_path,meta)
        print('ENCODED',index,len(records),sample.name,flush=True)
        if index==2:print('SMOKE_TWO_SAMPLES_PASSED',flush=True)
    if not args.limit:
        verification=verify_roundtrips(vae,dataset,records)
        manifest={'dataset':str(dataset),'count':len(records),'model':str(MODEL),
                  'inputs':['displacement.png','maps.png'],'displacement_shape':[128,32,32],
                  'paired_shape':[128,32,64],'portrait_latents_verified_unchanged':True,
                  'roundtrip_verification':verification}
        atomic_json(output/'manifest.json',manifest)
        config['latent_modes']=['displacement_only','paired_maps'];atomic_json(config_path,config)
        state.update({'displacement_latents_ready':True,'paired_map_latents_ready':True})
        atomic_json(state_path,state)
        print('ALL_LATENTS_READY',json.dumps(manifest),flush=True)
    else:print('SMOKE_LATENTS_READY',len(records),flush=True)


if __name__=='__main__':main()
