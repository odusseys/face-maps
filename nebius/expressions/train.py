"""Train three 256px UV targets from an expressive portrait; launched explicitly."""
import argparse,json,math,os,random,time,uuid
from pathlib import Path
from facekit.paths import DATA,MODEL,ROOT
for key,path in [('HF_HOME',DATA/'cache/huggingface'),('XDG_CACHE_HOME',DATA/'cache'),('WANDB_DIR',DATA/'wandb'),('WANDB_CACHE_DIR',DATA/'cache/wandb'),('WANDB_CONFIG_DIR',DATA/'wandb_config')]:os.environ.setdefault(key,str(path))
from train_face_maps import add_lora,epoch_batch,list_samples
from nebius.expressions.prompt import PROMPT,load as load_prompt

TASK='face-albedo-canonical-head-expression-refined-256-v3'

def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DATA/'datasets/faces1000-canonical-head-256')
    p.add_argument('--output',type=Path,default=DATA/'training/flux2_canonical_head_refined_256_10k')
    p.add_argument('--wandb-project',default='flux2-face-uv-lora-20261031');p.add_argument('--wandb-name',default='nebius-canonical-head-refined-10k')
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--micro-batch-size',type=int,default=1)
    p.add_argument('--epochs',type=int,default=45);p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--inference-steps',type=int,default=28)
    p.add_argument('--log-every',type=int,default=200)
    p.add_argument('--seed',type=int,default=20261031);p.add_argument('--resume',type=Path);p.add_argument('--resume-latest',action='store_true')
    return p.parse_args()

def restore(model,optimizer,file,args,config):
    import torch
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    state=torch.load(file,map_location='cpu',weights_only=False)
    if state.get('task')!=TASK or state.get('dataset_config')!=config or state.get('inference_steps')!=args.inference_steps:raise ValueError('Incompatible checkpoint')
    expected=get_peft_model_state_dict(model);saved=state['lora']
    if set(saved)!=set(expected) or any(saved[k].shape!=v.shape for k,v in expected.items()):raise ValueError('Incompatible LoRA weights')
    if set_peft_model_state_dict(model,saved).unexpected_keys:raise ValueError('Unexpected LoRA keys')
    optimizer.load_state_dict(state['optimizer']);random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
    return state['step']

def save_final_adapter(model,args,config):
    """Export trained weights once; no optimizer state or resumable checkpoint."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    state={k:v.detach().cpu().contiguous() for k,v in get_peft_model_state_dict(model).items()}
    file=args.output/'final-adapter.safetensors';tmp=args.output/'final-adapter.safetensors.tmp'
    save_file(state,str(tmp),metadata={'task':TASK});tmp.replace(file)
    (args.output/'final-adapter.json').write_text(json.dumps({'task':TASK,'dataset':str(args.dataset),'dataset_config':config,
        'base_model':str(MODEL),'lora_rank':16,'lora_alpha':16,'state_dict_format':'peft'},indent=2))
    print('FINAL_ADAPTER_SAVED',file,flush=True)

def infer(model,condition,embeds,steps,seed):
    import torch
    from facekit.train_helpers import predict,inference_schedule
    shape=(condition.shape[0],condition.shape[1],condition.shape[2],condition.shape[3]*3)
    z=torch.randn(shape,device=condition.device,dtype=condition.dtype,generator=torch.Generator(device=condition.device).manual_seed(seed))
    scheduler=inference_schedule(MODEL,steps,shape,condition.device)
    with torch.no_grad():
        for t in scheduler.timesteps:
            prediction=predict(model,z,condition,t.expand(z.shape[0]).to(z.dtype)/1000,embeds)
            z=scheduler.step(prediction,t,z,return_dict=False)[0].to(condition.dtype)
    return z

def evaluate(model,vae,embeds,holdout,args,step,config):
    import torch
    from PIL import Image
    from facekit.latent import vae_decode
    from facekit.bootstrap import assets
    from nebius.expressions.debug import render_pair
    from nebius.expressions.external_eval import external_row,grid_with_external
    picks=random.Random(args.seed+step).sample(holdout,2); blender,_,_=assets();rows=[]
    model.eval();vae.to('cuda')
    try:
        for i,p in enumerate(picks):
            data=torch.load(p/'latents.pt',map_location='cpu',weights_only=True)
            condition=data['condition'].unsqueeze(0).to('cuda',dtype=torch.bfloat16)
            prediction=vae_decode(vae,infer(model,condition,embeds,args.inference_steps,args.seed+step+i))[0]
            work=args.output/'debug'/f'{step:07d}'/p.name;work.mkdir(parents=True,exist_ok=True)
            prediction.save(work/'predicted_maps.png')
            faces=render_pair(args.dataset,p,prediction,blender,work,config)
            target=Image.open(p/'maps.png').convert('RGB');panels=[]
            for j in range(3):panels.extend([target.crop((j*256,0,(j+1)*256,256)),prediction.crop((j*256,0,(j+1)*256,256))])
            rows.append(faces+panels)
        rows.append(external_row(args.dataset,vae,model,embeds,args,config,blender,infer,step))
        grid=grid_with_external(rows);grid.save(args.output/'debug'/f'{step:07d}'/'comparison.png')
        return grid,'Dataset sample IDs: '+', '.join(p.name for p in picks)+'; external portrait: aligned input, ground truth unavailable'
    finally:
        vae.to('cpu');torch.cuda.empty_cache();model.train()

def main():
    args=parse()
    if not args.dataset.resolve().is_relative_to(ROOT.resolve()) or not args.output.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Paths must be on FACE_SHARED_ROOT')
    if args.micro_batch_size<1 or args.batch_size<1 or args.batch_size%args.micro_batch_size or min(args.epochs,args.steps,args.log_every,args.inference_steps)<1:raise ValueError('Invalid training arguments')
    import torch,wandb
    from diffusers import AutoencoderKLFlux2,Flux2Transformer2DModel
    from facekit.train_helpers import flow_loss,inference_schedule
    from facekit.bootstrap import checkpoint
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    config=json.loads((args.dataset/'dataset.json').read_text());manifest=json.loads((args.dataset/'dataset_complete.json').read_text())
    validation=json.loads((args.dataset/'validation.json').read_text())
    if (config['panel_order']!=['albedo','neutral_shape','expression'] or config.get('shape_encoding')!='signed_log1p_q97_range4'
        or config.get('expression_encoding')!='signed_log1p_q97_range24' or manifest.get('refinement_steps')!=80
        or manifest['count']!=1000 or manifest['target_shape']!=[128,16,48] or validation['count']!=1000):
        raise ValueError('Incompatible or unvalidated refined canonical head dataset')
    train,holdout=list_samples(args.dataset)
    if len(train)!=900 or len(holdout)!=100:raise ValueError('Invalid dataset split')
    args.output.mkdir(parents=True,exist_ok=True)
    for key in ('WANDB_DIR','WANDB_CACHE_DIR','WANDB_CONFIG_DIR'):Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    schedule=inference_schedule(MODEL,args.inference_steps,(128,16,48),'cuda')
    total_steps=min(args.steps,args.epochs*math.ceil(len(train)/args.batch_size))
    checkpoint(vae_only=False,include_text=False);embeds=load_prompt()
    model=Flux2Transformer2DModel.from_pretrained(str(MODEL),subfolder='transformer',torch_dtype=torch.bfloat16).to('cuda')
    targets=add_lora(model);model.train();optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.learning_rate)
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).eval().requires_grad_(False)
    if not args.resume and args.resume_latest:
        files=sorted(args.output.glob('checkpoint-*.pt'));args.resume=files[-1] if files else None
    start=restore(model,optimizer,args.resume,args,config) if args.resume else 0
    if start>total_steps:raise ValueError('Checkpoint past requested training length')
    id_file=args.output/'wandb-run-id.txt'
    run_id=id_file.read_text().strip() if args.resume and id_file.exists() else (os.environ.get('WANDB_RUN_ID') or uuid.uuid4().hex[:16])
    run=wandb.init(project=args.wandb_project,name=args.wandb_name,id=run_id,resume='allow',dir=os.environ['WANDB_DIR'],config={
      'task':TASK,'dataset':str(args.dataset),'batch_size':args.batch_size,'micro_batch_size':args.micro_batch_size,'lora_targets':targets,
      'steps':total_steps,'learning_rate':args.learning_rate,'prompt':PROMPT,'condition':'expressive hair portrait',
      'target_order':config['panel_order'],'shape_encoding':config['shape_encoding'],'expression_encoding':config['expression_encoding'],
      'latent_refinement':config['latent_refinement'],'shape_bound_mm':config['shape_bound_mm'],'expression_bound_mm':config['expression_bound_mm'],
      'inference_steps':args.inference_steps,'inference_log_interval':args.log_every})
    id_file.write_text(run_id);wandb.define_metric('train/global_step');wandb.define_metric('*',step_metric='train/global_step')
    try:
        for step in range(start+1,total_steps+1):
            started=time.monotonic();optimizer.zero_grad(set_to_none=True);total=0.;epoch,chosen=epoch_batch(train,step,args.batch_size,args.seed)
            for offset in range(0,len(chosen),args.micro_batch_size):
                batch=chosen[offset:offset+args.micro_batch_size];data=[torch.load(p/'latents.pt',map_location='cpu',weights_only=True) for p in batch]
                condition=torch.stack([x['condition'] for x in data]).to('cuda',dtype=torch.bfloat16)
                target=torch.stack([x['maps'] for x in data]).to('cuda',dtype=torch.bfloat16)
                if condition.shape[1:]!=(128,16,16) or target.shape[1:]!=(128,16,48):raise ValueError('Invalid latent dimensions')
                loss=flow_loss(model,target,condition,embeds,schedule.timesteps);(loss*len(batch)/len(chosen)).backward();total+=float(loss.detach())*len(batch)/len(chosen)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optimizer.step()
            metrics={'train/global_step':step,'train/loss':total,'train/epoch':epoch+1,'train/step_seconds':time.monotonic()-started}
            if step%args.log_every==0:
                optimizer.zero_grad(set_to_none=True);comparison,caption=evaluate(model,vae,embeds,holdout,args,step,config)
                metrics['holdout/comparison']=wandb.Image(comparison,caption=caption)
            wandb.log(metrics)
        if total_steps>start:save_final_adapter(model,args,config)
    finally:run.finish()

if __name__=='__main__':main()
