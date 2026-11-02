#!/usr/bin/env python3
"""Train rank-16 FLUX.2 attention/MLP LoRA from portraits to UV maps.

This script is intentionally not executed by dataset generation. Effective batch
size defaults to four; micro-batches allow it to fit on smaller GPUs.
"""
import argparse, math, json, os, random, time, uuid
from pathlib import Path
from facekit.paths import ROOT, DATA, MODEL
os.environ.setdefault('HF_HOME',str(DATA/'cache'/'huggingface'))
os.environ.setdefault('XDG_CACHE_HOME',str(DATA/'cache'))
os.environ.setdefault('WANDB_DIR',str(DATA/'wandb'))
os.environ.setdefault('WANDB_CACHE_DIR',str(DATA/'cache'/'wandb'))
os.environ.setdefault('WANDB_CONFIG_DIR',str(DATA/'wandb_config'))


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DATA/'datasets'/'faces500-centered-linear-q97')
    p.add_argument('--output',type=Path,default=DATA/'training'/'flux2_face_lora')
    p.add_argument('--wandb-project',default='flux2-face-uv-lora-20261031')
    p.add_argument('--wandb-name',default=None)
    p.add_argument('--displacement-only',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--displacement-latents',type=Path,default=None)
    p.add_argument('--batch-size',type=int,default=4); p.add_argument('--micro-batch-size',type=int,default=1)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--steps',type=int,help='Optional cap, for a smoke test or partial run')
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--log-every',type=int,default=200); p.add_argument('--inference-steps',type=int,default=28)
    p.add_argument('--checkpoint-every',type=int,default=50)
    p.add_argument('--resume-latest',action='store_true')
    p.add_argument('--seed',type=int,default=20261031); p.add_argument('--resume',type=Path)
    return p.parse_args()


def list_samples(dataset):
    records=[]
    for f in sorted((dataset/'samples').glob('*/complete.json')):
        meta=json.loads(f.read_text()); records.append((f.parent,meta['split']))
    train=[p for p,s in records if s=='train']; holdout=[p for p,s in records if s=='holdout']
    if len(holdout)!=100: raise ValueError(f'Expected exactly 100 holdout samples, found {len(holdout)}')
    if not train: raise ValueError('No completed training samples')
    return train,holdout


def epoch_batch(train,step,batch_size,seed):
    """Visit each training example exactly once per deterministically shuffled epoch."""
    steps_per_epoch=math.ceil(len(train)/batch_size)
    epoch=(step-1)//steps_per_epoch
    batch_index=(step-1)%steps_per_epoch
    shuffled=list(train)
    random.Random(seed+epoch).shuffle(shuffled)
    return epoch,shuffled[batch_index*batch_size:(batch_index+1)*batch_size]


def add_lora(model):
    import torch
    from peft import LoraConfig
    targets=[name for name,layer in model.named_modules() if isinstance(layer,torch.nn.Linear)
             and any(part in name for part in ('.attn.','.ff.','.ff_context.'))]
    if not targets or not any('.ff.' in n for n in targets) or not any('.attn.' in n for n in targets):
        raise RuntimeError('Could not identify FLUX.2 attention and MLP linear layers')
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=16,lora_alpha=16,init_lora_weights='gaussian',target_modules=targets))
    for p in model.parameters():
        if p.requires_grad: p.data=p.data.float()
    model.enable_gradient_checkpointing()
    return targets


def save_checkpoint(model,optimizer,step,args):
    import torch
    from peft import get_peft_model_state_dict
    path=args.output/f'checkpoint-{step:07d}.pt'; temp=path.with_suffix('.tmp')
    state={k:v.detach().cpu() for k,v in get_peft_model_state_dict(model).items()}
    torch.save({'step':step,'lora':state,'optimizer':optimizer.state_dict(), 'displacement_only':args.displacement_only,
                'training_noise_schedule':'uniform inference timesteps','inference_steps':args.inference_steps,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                'python_rng':random.getstate()},temp)
    temp.replace(path)
    print('CHECKPOINT',path,flush=True)


def restore_checkpoint(model,optimizer,path,displacement_only=False):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    if ckpt.get('displacement_only',False) != displacement_only:
        raise ValueError('Checkpoint target mode differs from the requested target mode')
    saved=ckpt['lora']; expected=get_peft_model_state_dict(model)
    if set(saved)!=set(expected):
        raise ValueError(f'LoRA checkpoint keys do not match model: missing={len(set(expected)-set(saved))}, extra={len(set(saved)-set(expected))}')
    mismatched=[k for k in expected if saved[k].shape!=expected[k].shape]
    if mismatched: raise ValueError(f'LoRA checkpoint shape mismatch: {mismatched[:3]}')
    result=set_peft_model_state_dict(model,saved)
    if result.unexpected_keys: raise ValueError(f'Unexpected LoRA weights: {result.unexpected_keys[:3]}')
    optimizer.load_state_dict(ckpt['optimizer'])
    random.setstate(ckpt['python_rng']); torch.set_rng_state(ckpt['torch_rng']); torch.cuda.set_rng_state(ckpt['cuda_rng'])
    print('RESUMED',str(path),'STEP',ckpt['step'],flush=True)
    return ckpt['step']


def image_grid(rows,displacement_only=False,gt_map_face=False):
    from PIL import Image,ImageDraw
    labels=([('GT face from map' if gt_map_face else 'GT face'),'Predicted face','GT displacement','Predicted displacement'] if displacement_only else
            ['Input: hair, varied distance','GT face: standard view','Predicted face: standard view',
             'GT texture','Predicted texture','GT displacement','Predicted displacement'])
    size=256; header=28
    grid=Image.new('RGB',(size*len(labels),len(rows)*(size+header)),(0,0,0)); draw=ImageDraw.Draw(grid)
    for r,row in enumerate(rows):
        for col,(label,img) in enumerate(zip(labels,row)):
            x=col*size; y=r*(size+header); draw.text((x+4,y+5),label,fill='white')
            grid.paste(img.convert('RGB').resize((size,size),Image.Resampling.LANCZOS),(x,y+header))
    return grid


def evaluate(model,vae,embeds,holdout,args,step):
    import torch
    from PIL import Image
    from facekit.train_helpers import infer
    from facekit.latent import vae_decode
    from facekit.debug_render import render_debug
    from facekit.bootstrap import assets
    picks=random.Random(args.seed+step).sample(holdout,3)
    blender,_,_=assets(); rows=[]
    settings=json.loads((args.dataset/'dataset.json').read_text())
    gt_map_face=settings.get('geometry_alignment')=='rigid alignment plus optimal uniform scale to reference face'
    model.eval(); vae.to('cuda')
    for i,path in enumerate(picks):
        data=torch.load(path/'latents.pt',map_location='cpu',weights_only=True)
        condition_key='face_bare' if args.displacement_only else 'face_hair'
        condition=data[condition_key].unsqueeze(0).to('cuda',dtype=torch.bfloat16)
        pred=infer(model,condition,embeds,MODEL,args.inference_steps,args.seed+step+i,args.displacement_only)
        image=vae_decode(vae,pred)[0]
        work=args.output/'debug'/f'{step:07d}'/path.name
        work.mkdir(parents=True,exist_ok=True); image.save(work/'predicted_maps.png')
        if args.displacement_only:
            meta=json.loads((path/'complete.json').read_text())
            face=render_debug(args.dataset,path,image,blender,work,settings['bound_mm'],settings['encoding'],texture_path=meta['texture'])
            gt_map=Image.open(path/'displacement.png').convert('RGB')
            if gt_map_face:
                gt_face=render_debug(args.dataset,path,gt_map,blender,work/'gt_map_face',
                                     settings['bound_mm'],settings['encoding'],texture_path=meta['texture'])
            else:
                gt_face=Image.open(path/'portrait_bare.png')
            rows.append([gt_face,face,gt_map,image])
        else:
            face=render_debug(args.dataset,path,image,blender,work,settings['bound_mm'],settings['encoding'])
            gt=Image.open(path/'maps.png').convert('RGB')
            gt_face=Image.open(path/'portrait_bare.png').convert('RGB')
            rows.append([Image.open(path/'portrait_hair.png'),gt_face,face,gt.crop((0,0,512,512)),
                         image.crop((0,0,512,512)),gt.crop((512,0,1024,512)),image.crop((512,0,1024,512))])
    result=image_grid(rows,args.displacement_only,gt_map_face); dest=args.output/'debug'/f'{step:07d}'/'comparison.png'; result.save(dest)
    vae.to('cpu'); torch.cuda.empty_cache(); model.train()
    return result,'Rows: '+', '.join(p.name for p in picks)


def main():
    args=parse()
    if args.displacement_latents is None: args.displacement_latents=args.dataset/'latents_displacement'
    if not args.dataset.resolve().is_relative_to(ROOT.resolve()) or not args.output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Dataset and training outputs must be on FACE_SHARED_ROOT')
    if args.batch_size<1 or args.micro_batch_size<1 or args.batch_size%args.micro_batch_size:
        raise ValueError('batch-size must be a positive multiple of micro-batch-size')
    if args.epochs<1 or args.log_every<1 or args.checkpoint_every<1 or args.inference_steps<1 or (args.steps is not None and args.steps<1):
        raise ValueError('epochs, step cap, intervals, and inference steps must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    for key in ('WANDB_DIR','WANDB_CACHE_DIR','WANDB_CONFIG_DIR'): Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    import torch, wandb
    from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
    from facekit.bootstrap import checkpoint
    from facekit.train_helpers import flow_loss, inference_schedule, prompt_embedding
    from facekit.prompt_cache import prompt_for
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    train,holdout=list_samples(args.dataset)
    dataset_config=json.loads((args.dataset/'dataset.json').read_text())
    if not args.displacement_only and 'latent_modes' in dataset_config and 'paired_maps' not in dataset_config['latent_modes']:
        raise ValueError('This dataset has no paired-map latents; recompute them before paired-map training')
    if args.displacement_only:
        missing=[p.name for p in train+holdout if not (args.displacement_latents/(p.name+'.pt')).is_file()]
        if missing: raise ValueError(f'Missing displacement-only latents for {len(missing)} samples, e.g. {missing[:3]}')
        target_shape=torch.load(args.displacement_latents/(train[0].name+'.pt'),map_location='cpu',weights_only=True).shape
    else:
        target_shape=torch.load(train[0]/'latents.pt',map_location='cpu',weights_only=True)['maps'].shape
    schedule=inference_schedule(MODEL,args.inference_steps,target_shape,'cuda')
    train_timesteps=schedule.timesteps
    print('TRAIN_NOISE_SCHEDULE',json.dumps({'sampling':'uniform inference timesteps','timesteps':train_timesteps.cpu().tolist()}),flush=True)
    total_steps=args.epochs*math.ceil(len(train)/args.batch_size)
    if args.steps is not None: total_steps=min(total_steps,args.steps)
    print('TRAIN_CONFIG',json.dumps({'train':len(train),'holdout':len(holdout),'epochs':args.epochs,
          'batch_size':args.batch_size,'steps':total_steps,'dataset':str(args.dataset)}),flush=True)
    checkpoint(vae_only=False,include_text=False)
    embeds=prompt_embedding(MODEL,args.displacement_only)
    model=Flux2Transformer2DModel.from_pretrained(str(MODEL),subfolder='transformer',torch_dtype=torch.bfloat16).to('cuda')
    targets=add_lora(model); model.train()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.learning_rate)
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).eval()
    start=0
    if not args.resume and args.resume_latest:
        saved=sorted(args.output.glob('checkpoint-*.pt'))
        if saved: args.resume=saved[-1]
    if args.resume:
        start=restore_checkpoint(model,optimizer,args.resume,args.displacement_only)
        if start>total_steps: raise ValueError(f'Checkpoint step {start} exceeds target {total_steps}')
    run_id_file=args.output/'wandb-run-id.txt'
    run_id=run_id_file.read_text().strip() if args.resume and run_id_file.exists() else uuid.uuid4().hex[:16]
    run=wandb.init(project=args.wandb_project,name=args.wandb_name,id=run_id,resume='allow',dir=os.environ['WANDB_DIR'],config={
        'dataset':str(args.dataset),'batch_size':args.batch_size,'micro_batch_size':args.micro_batch_size,
        'rank':16,'lora_targets':targets,'epochs':args.epochs,'steps':total_steps,'learning_rate':args.learning_rate,
        'prompt':str(embeds.shape),'prompt_text':prompt_for(args.displacement_only),
        'prompt_embedding_shape':list(embeds.shape),'target_shape':list(target_shape),
        'conditioning':'random hair or bare' if args.displacement_only else 'hair at varied distance',
        'displacement_only':args.displacement_only,'displacement_latents':str(args.displacement_latents),
        'displacement_encoding':dataset_config.get('encoding'),'displacement_bound_mm':dataset_config.get('bound_mm'),
        'training_noise_schedule':'uniform inference timesteps','inference_steps':args.inference_steps,
        'inference_log_interval':args.log_every,'checkpoint_interval':args.checkpoint_every})
    run_id_file.write_text(run_id)
    if args.resume:
        wandb.config.update({'epochs':args.epochs,'steps':total_steps,
                             'training_noise_schedule':'uniform inference timesteps',
                             'inference_steps':args.inference_steps},allow_val_change=True)
    wandb.define_metric('train/global_step')
    wandb.define_metric('*',step_metric='train/global_step')
    try:
        for step in range(start+1,total_steps+1):
            started=time.monotonic()
            optimizer.zero_grad(set_to_none=True); total=0.0
            epoch,chosen=epoch_batch(train,step,args.batch_size,args.seed)
            for offset in range(0,len(chosen),args.micro_batch_size):
                batch=chosen[offset:offset+args.micro_batch_size]; data=[torch.load(p/'latents.pt',map_location='cpu',weights_only=True) for p in batch]
                condition_keys=([random.choice(['face_hair','face_bare']) for _ in data]
                                if args.displacement_only else ['face_hair']*len(data))
                condition=torch.stack([d[key] for d,key in zip(data,condition_keys)]).to('cuda',dtype=torch.bfloat16)
                if args.displacement_only:
                    target=torch.stack([torch.load(args.displacement_latents/(p.name+'.pt'),map_location='cpu',weights_only=True) for p in batch]).to('cuda',dtype=torch.bfloat16)
                else:
                    target=torch.stack([d['maps'] for d in data]).to('cuda',dtype=torch.bfloat16)
                loss=flow_loss(model,target,condition,embeds,train_timesteps); (loss*len(batch)/len(chosen)).backward()
                total+=float(loss.detach())*len(batch)/len(chosen)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            optimizer.step()
            metrics={'train/global_step':step,'train/loss':total,'train/epoch':epoch+1,'train/step_seconds':time.monotonic()-started}
            if step==start+1 or step%10==0: print('STEP',step,'EPOCH',epoch+1,'LOSS',total,'SECONDS',time.monotonic()-started,flush=True)
            if step%args.log_every==0:
                optimizer.zero_grad(set_to_none=True)
                comparison,caption=evaluate(model,vae,embeds,holdout,args,step)
                metrics['holdout/comparison']=wandb.Image(comparison,caption=caption)
            wandb.log(metrics)
            if step%args.checkpoint_every==0 or step%args.log_every==0: save_checkpoint(model,optimizer,step,args)
        if total_steps>start and total_steps%args.checkpoint_every: save_checkpoint(model,optimizer,total_steps,args)
        print('TRAINING_COMPLETE',total_steps,flush=True)
    finally:
        run.finish()


if __name__=='__main__': main()
