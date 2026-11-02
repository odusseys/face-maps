"""Check identity pairing, canonical geometry, ranges and refined latents."""
import collections,json
import numpy as np,torch
from PIL import Image
from facekit.paths import DATA
from nebius.expressions.refine_latents import STEPS

def main():
    source=DATA/'datasets/faces1000-canonical-expression-256';ds=DATA/'datasets/faces1000-canonical-head-256'
    cfg=json.loads((ds/'dataset.json').read_text());manifest=json.loads((ds/'dataset_complete.json').read_text())
    if manifest['count']!=1000 or manifest['refinement_steps']!=STEPS:raise ValueError('Incomplete latent manifest')
    ref=np.load(ds/'reference.npz');uv=np.load(ds/'uv_mapping.npz');original=np.load(source/'reference.npz')
    ids=np.unique(ref['triangle_vertices']);oldids=np.unique(original['triangle_vertices'])
    if len(np.intersect1d(ids,oldids))!=len(oldids) or len(ids)<=len(oldids) or 'color_mask' not in uv:raise ValueError('Invalid head coverage')
    split=collections.Counter();condition_ratios=[];target_ratios=[];condition_shift=[];target_shift=[]
    samples=sorted((ds/'samples').glob('*/complete.json'))
    if len(samples)!=1000:raise ValueError('Incorrect dataset count')
    for i,meta_path in enumerate(samples,1):
        p=meta_path.parent;src=source/'samples'/p.name;meta=json.loads(meta_path.read_text());old=json.loads((src/'complete.json').read_text());split[meta['split']]+=1
        if any(meta[k]!=old[k] for k in ('index','split','texture','skin','hair','expression')):raise ValueError(f'Identity changed: {p.name}')
        with np.load(p/'geometry.npz') as a,np.load(src/'geometry.npz') as b:
            if not np.array_equal(a['shape_m'],b['shape_m']) or not np.array_equal(a['expression_m'],b['expression_m']):raise ValueError(f'Geometry changed: {p.name}')
        if meta['shape_clipped_fraction'] or meta['expression_clipped_fraction']:raise ValueError(f'UV clipping: {p.name}')
        for name,size in [('portrait_expression_hair.png',(256,256)),('portrait_expression_bare.png',(256,256)),('albedo.png',(256,256)),('shape.png',(256,256)),('expression.png',(256,256)),('maps.png',(768,256))]:
            with Image.open(p/name) as img:
                if img.size!=size:raise ValueError(f'Invalid dimensions: {p/name}')
        with Image.open(p/'maps.png') as image:
            for j,name in enumerate(('albedo.png','shape.png','expression.png')):
                with Image.open(p/name) as im:
                    if not np.array_equal(np.asarray(image.crop((j*256,0,(j+1)*256,256))),np.asarray(im)):raise ValueError(f'Mismatched panel: {p.name}')
        latent=torch.load(p/'latents.pt',map_location='cpu',weights_only=True)
        if tuple(latent['condition'].shape)!=(128,16,16) or tuple(latent['maps'].shape)!=(128,16,48) or not torch.isfinite(latent['condition']).all() or not torch.isfinite(latent['maps']).all():raise ValueError(f'Invalid latent: {p.name}')
        stats=json.loads((p/'refinement.json').read_text())
        if stats['steps']!=STEPS or any(not np.isfinite(stats[k]) for k in ('condition_before','condition_after','maps_before','maps_after','condition_delta_rms','maps_delta_rms')):raise ValueError(f'Invalid refinement: {p.name}')
        if stats['condition_after']>stats['condition_before']*1.001+1e-9 or stats['maps_after']>stats['maps_before']*1.001+1e-9:raise ValueError(f'Reconstruction worsened: {p.name}')
        condition_ratios.append(stats['condition_after']/max(stats['condition_before'],1e-20));target_ratios.append(stats['maps_after']/max(stats['maps_before'],1e-20))
        condition_shift.append(stats['condition_delta_rms']);target_shift.append(stats['maps_delta_rms'])
        if i%200==0:print('HEAD_VERIFIED',i,flush=True)
    if split!={'train':900,'holdout':100}:raise ValueError(f'Incorrect split: {split}')
    report={'count':len(samples),'split':dict(split),'head_uv_vertices':len(ids),'original_face_vertices':len(oldids),
      'shape_bound_mm':cfg['shape_bound_mm'],'expression_bound_mm':cfg['expression_bound_mm'],'shape_clipping':0,'expression_clipping':0,
      'mean_condition_loss_ratio':float(np.mean(condition_ratios)),'mean_map_loss_ratio':float(np.mean(target_ratios)),
      'worst_condition_loss_ratio':float(max(condition_ratios)),'worst_map_loss_ratio':float(max(target_ratios)),
      'mean_condition_latent_shift_rms':float(np.mean(condition_shift)),'mean_map_latent_shift_rms':float(np.mean(target_shift))}
    temp=ds/'validation.json.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(ds/'validation.json');print('HEAD_VALIDATED',json.dumps(report),flush=True)

if __name__=='__main__':main()
