"""Create refined target and portrait latents for the canonical head dataset."""
import argparse,json,os,time
import numpy as np,torch
from PIL import Image
from diffusers import AutoencoderKLFlux2
from facekit.paths import DATA,MODEL
from nebius.expressions.refine_latents import STEPS,refine,map_weight,portrait_weight

def alive(pid):
    if not pid:return True
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def atomic_json(path,data):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--builder-pid',type=int);p.add_argument('--batch-size',type=int,default=8);args=p.parse_args()
    if not 1<=args.batch_size<=8:raise ValueError('Invalid batch size')
    ds=DATA/'datasets/faces1000-canonical-head-256';cfg=json.loads((ds/'dataset.json').read_text())
    if cfg['latent_refinement']['steps']!=STEPS:raise ValueError('Dataset/refinement settings disagree')
    uv=np.load(ds/'uv_mapping.npz');weights=map_weight(uv);vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    processed=set()
    for f in (ds/'samples').glob('*/refinement.json'):
        if (f.parent/'latents.pt').is_file():processed.add(f.parent.name)
    while True:
        ready=sorted(p.parent for p in (ds/'samples').glob('*/complete.json') if p.parent.name not in processed)
        if not ready:
            if (ds/'maps_complete.json').is_file():break
            if not alive(args.builder_pid):raise RuntimeError('Map builder exited without completing')
            time.sleep(4);continue
        batch=ready[:args.batch_size]
        if len(batch)<args.batch_size and not (ds/'maps_complete.json').is_file() and alive(args.builder_pid):
            time.sleep(2);continue
        started=time.monotonic();portraits=[Image.open(p/'portrait_expression_hair.png').convert('RGB') for p in batch]
        maps=[Image.open(p/'maps.png').convert('RGB') for p in batch]
        if any(x.size!=(256,256) for x in portraits) or any(x.size!=(768,256) for x in maps):raise ValueError('Bad image dimensions')
        conditions,cstats=refine(vae,portraits,portrait_weight(portraits));targets,tstats=refine(vae,maps,weights)
        if tuple(conditions.shape)!=(len(batch),128,16,16) or tuple(targets.shape)!=(len(batch),128,16,48):raise ValueError('Bad latent dimensions')
        for i,p in enumerate(batch):
            file=p/'latents.pt';tmp=file.with_suffix('.pt.tmp');torch.save({'condition':conditions[i],'maps':targets[i]},tmp);tmp.replace(file)
            stat={'steps':STEPS,'condition_before':cstats['before'][i],'condition_after':cstats['after'][i],
              'condition_delta_rms':cstats['delta_rms'][i],'maps_before':tstats['before'][i],
              'maps_after':tstats['after'][i],'maps_delta_rms':tstats['delta_rms'][i]}
            atomic_json(p/'refinement.json',stat);processed.add(p.name)
        print('REFINED',len(processed),'of 1000','batch',len(batch),'seconds',round(time.monotonic()-started,1),flush=True)
    if len(processed)!=1000:raise ValueError(f'Expected 1000 refined samples, got {len(processed)}')
    stats=[json.loads((ds/'samples'/i/'refinement.json').read_text()) for i in sorted(processed)]
    manifest={'count':len(processed),'condition_shape':[128,16,16],'target_shape':[128,16,48],
      'refinement_steps':STEPS,'mean_condition_loss_ratio':float(np.mean([x['condition_after']/x['condition_before'] for x in stats])),
      'mean_map_loss_ratio':float(np.mean([x['maps_after']/x['maps_before'] for x in stats]))}
    atomic_json(ds/'dataset_complete.json',manifest);print('REFINED_DATASET_READY',json.dumps(manifest),flush=True)

if __name__=='__main__':main()
