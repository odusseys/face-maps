"""Verify FLUX conditioning, batching, scheduler and debug grid; no optimizer."""
import json
import torch
from pathlib import Path
from PIL import Image
from diffusers import AutoencoderKLFlux2,Flux2Transformer2DModel
from facekit.paths import DATA,MODEL
from facekit.latent import vae_decode
from facekit.bootstrap import assets,checkpoint
from facekit.train_helpers import predict,inference_schedule
from nebius.expressions.prompt import load
from nebius.expressions.train import infer
from nebius.expressions.debug import render_pair,make_grid

def main():
    ds=DATA/'datasets/faces1000-shape-expression-256';config=json.loads((ds/'dataset.json').read_text())
    samples=[ds/'samples'/f'{i:05d}' for i in (0,1)];data=[torch.load(p/'latents.pt',map_location='cpu',weights_only=True) for p in samples]
    assert all(tuple(x['condition'].shape)==(128,16,16) and tuple(x['maps'].shape)==(128,16,48) for x in data)
    checkpoint(vae_only=False,include_text=False);embeds=load();torch.manual_seed(431)
    model=Flux2Transformer2DModel.from_pretrained(str(MODEL),subfolder='transformer',torch_dtype=torch.bfloat16).to('cuda').eval()
    condition=torch.stack([x['condition'] for x in data]).to('cuda',dtype=torch.bfloat16)
    noise=torch.randn((1,128,16,48),device='cuda',dtype=torch.bfloat16);t=torch.tensor([.7],device='cuda',dtype=torch.bfloat16)
    with torch.inference_mode():
        individual=[predict(model,noise,condition[i:i+1],t,embeds) for i in range(2)]
        batched=predict(model,noise.expand(2,-1,-1,-1),condition,t.expand(2),embeds)
        errors=[float((batched[i].float()-individual[i][0].float()).abs().mean()) for i in range(2)]
        conditioning=float((individual[0].float()-individual[1].float()).abs().mean())
        if not torch.isfinite(batched).all() or max(errors)>.03 or conditioning<1e-5:raise ValueError(f'Batch/condition mismatch: {errors}, {conditioning}')
        latent=infer(model,condition[:1],embeds,2,123)
    if tuple(latent.shape)!=(1,128,16,48) or not torch.isfinite(latent).all():raise ValueError('Invalid inference result')
    del model,condition,individual,batched;torch.cuda.empty_cache()
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval()
    image=vae_decode(vae,latent)[0]
    if image.size!=(768,256):raise ValueError('Invalid decoded dimensions')
    work=DATA/'runs/expression-forward-check';work.mkdir(parents=True,exist_ok=True);image.save(work/'untrained_prediction.png')
    blender,_,_=assets();faces=render_pair(ds,samples[0],image,blender,work/'render',config)
    target=Image.open(samples[0]/'maps.png').convert('RGB');panels=[]
    for j in range(3):panels.extend([target.crop((j*256,0,(j+1)*256,256)),image.crop((j*256,0,(j+1)*256,256))])
    grid=make_grid([faces+panels]);grid.save(work/'comparison.png')
    report={'sample_ids':[p.name for p in samples],'batch_vs_individual_mean_abs':errors,'conditioning_difference_mean_abs':conditioning,
      'output_latent_shape':list(latent.shape),'decoded_size':list(image.size),'inference_steps':2,'trained':False}
    (work/'results.json').write_text(json.dumps(report,indent=2));print('FORWARD_CHECK',json.dumps(report),flush=True)

if __name__=='__main__':main()
