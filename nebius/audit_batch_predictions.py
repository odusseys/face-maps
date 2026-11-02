#!/usr/bin/env python3
"""Check batched transformer predictions preserve individual sample identity."""
import json
import torch
from diffusers import Flux2Transformer2DModel
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from facekit.paths import DATA,MODEL
from facekit.train_helpers import predict,prompt_embedding
from train_face_maps import add_lora,list_samples


def stats(a,b):
    d=(a.float()-b.float()).abs()
    return {'max_abs':float(d.max()),'mean_abs':float(d.mean()),'rmse':float(d.square().mean().sqrt())}


def main():
    ds=DATA/'datasets/faces500-moderate-uv-fixed-texture';run=DATA/'training/flux2_moderate_fixed_texture'
    train,holdout=list_samples(ds);paths=[train[0],train[1],holdout[0],holdout[1]]
    condition=torch.stack([torch.load(p/'latents.pt',map_location='cpu',weights_only=True)['face_bare'] for p in paths]).to('cuda',torch.bfloat16)
    target=torch.stack([torch.load(ds/'latents_displacement'/(p.name+'.pt'),map_location='cpu',weights_only=True) for p in paths]).to('cuda',torch.bfloat16)
    generator=torch.Generator(device='cuda').manual_seed(56789)
    noise=torch.randn(target.shape,device='cuda',dtype=target.dtype,generator=generator)
    times=torch.tensor([.20,.50,.80,.95],device='cuda',dtype=target.dtype)
    noisy=(1-times[:,None,None,None])*target+times[:,None,None,None]*noise
    checkpoint=sorted(run.glob('checkpoint-*.pt'))[-1]
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);step=saved['step']
    model=Flux2Transformer2DModel.from_pretrained(str(MODEL),subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    add_lora(model);model.eval()
    if set(saved['lora'])!=set(get_peft_model_state_dict(model)):raise ValueError('Checkpoint mismatch')
    if set_peft_model_state_dict(model,saved['lora']).unexpected_keys:raise ValueError('Unexpected weights')
    del saved
    embeds=prompt_embedding(MODEL,True)
    with torch.inference_mode():
        batched=predict(model,noisy,condition,times,embeds)
        separate=torch.cat([predict(model,noisy[i:i+1],condition[i:i+1],times[i:i+1],embeds) for i in range(4)])
        permutation=torch.tensor([2,0,3,1],device='cuda')
        permuted=predict(model,noisy[permutation],condition[permutation],times[permutation],embeds)
        inverse=torch.argsort(permutation)
        changed_condition=condition.clone();changed_condition[0]=condition[1]
        changed_noisy=noisy.clone();changed_noisy[0]=noisy[1]
        perturbed=predict(model,changed_noisy,changed_condition,times,embeds)
    batch_vs_single=stats(batched,separate);permuted_result=stats(batched,permuted[inverse]);cross_item=stats(batched[1:],perturbed[1:])
    goal=(noise-target).float()
    loss_batch=(batched.float()-goal).square().mean()
    loss_micro=torch.stack([(separate[i].float()-goal[i]).square().mean() for i in range(4)]).mean()
    result={'checkpoint_step':step,'ids':[p.name for p in paths],'times':[float(v) for v in times],
            'batch_vs_individual':batch_vs_single,'permutation_reordered':permuted_result,
            'unmodified_items_when_first_changes':cross_item,'loss_batch':float(loss_batch),
            'loss_weighted_individual':float(loss_micro),'loss_abs_difference':float((loss_batch-loss_micro).abs()),
            'active_dropout_modules':[name for name,m in model.named_modules() if isinstance(m,torch.nn.Dropout) and m.p>0]}
    if batch_vs_single['rmse']>.02 or permuted_result['rmse']>.02 or cross_item['max_abs']>.02:
        raise ValueError(f'Batch interaction failed: {json.dumps(result)}')
    output=DATA/'audits/nebius-batch-predictions.json';output.write_text(json.dumps(result,indent=2))
    print('BATCH_PREDICTIONS_PASSED',json.dumps(result),flush=True)


if __name__=='__main__':main()
