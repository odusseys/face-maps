"""Audit paired geometry, images, latents, and expression statistics without training."""
import collections,json
from functools import lru_cache
import numpy as np,torch
from PIL import Image
from facekit.paths import DATA

def main():
    root=DATA/'datasets';raw=root/'faces1000-visibleface-centered';old=root/'faces1000-visibleface-uv-paired';out=root/'faces1000-canonical-expression-256'
    config=json.loads((out/'dataset.json').read_text());manifest=json.loads((out/'dataset_complete.json').read_text())
    if manifest['count']!=1000:raise ValueError('Incomplete dataset')
    samples=sorted((out/'samples').glob('*/complete.json'))
    if len(samples)!=1000:raise ValueError('Wrong sample count')
    kinds=collections.Counter();split=collections.Counter();shape_clip=[];expr_clip=[];errors=[];target_errors=[];magnitude=[]
    units=DATA/'assets/faceunits01/targets/faceunits'
    @lru_cache(maxsize=52)
    def target(name):
        data=np.loadtxt(units/(name+'.target'));data=data[data[:,0]<13380]
        result=np.zeros((13380,3),np.float32)
        np.add.at(result,data[:,0].astype(int),data[:,[1,3,2]]*np.array([1,-1,1])*.1)
        return result
    for i,meta_path in enumerate(samples):
        p=meta_path.parent;meta=json.loads(meta_path.read_text());source=raw/'samples'/p.name;neutral=old/'samples'/p.name
        oldmeta=json.loads((neutral/'complete.json').read_text());split[meta['split']]+=1;kinds[meta['expression']['expression_kind']]+=1
        if meta['split']!=oldmeta['split'] or meta['texture']!=oldmeta['texture']:raise ValueError(f'Identity/split changed: {p.name}')
        with np.load(p/'geometry.npz') as g, np.load(p/'raw_expression.npz') as e, np.load(source/'geometry.npz') as a,np.load(neutral/'geometry.npz') as b:
            if not np.array_equal(g['shape_m'],b['displacement_m']):raise ValueError(f'Shape changed with expression: {p.name}')
            summed=np.zeros_like(e['expression_delta_m'])
            for name,weight in meta['expression']['expression_weights'].items():summed+=target(name)*weight
            target_error=float(np.max(np.abs(summed-g['expression_m'])));target_errors.append(target_error)
            if target_error>6e-7:raise ValueError(f'Expression targets do not reproduce mesh: {p.name}: {target_error}')
            expected=e['expression_delta_m']@a['rotation']@b['rotation']*float(b['similarity_scale'])
            err=float(np.max(np.abs(expected-g['expression_m'])));errors.append(err)
            if err>1e-6:raise ValueError(f'Expression transform mismatch: {p.name}')
            magnitude.append(float(np.max(np.linalg.norm(g['expression_m'],axis=1))*1000))
        if not meta['expression']['expression_weights'] or magnitude[-1]<.001:raise ValueError(f'Missing expression: {p.name}')
        for name,size in [('portrait_expression_hair.png',(256,256)),('portrait_expression_bare.png',(256,256)),('portrait_neutral_bare.png',(256,256)),('albedo.png',(256,256)),('shape.png',(256,256)),('expression.png',(256,256)),('maps.png',(768,256))]:
            with Image.open(p/name) as im:
                if im.size!=size:raise ValueError(f'Bad dimensions: {p/name}')
        with Image.open(p/'maps.png') as im:
            for j,name in enumerate(('albedo.png','shape.png','expression.png')):
                if not np.array_equal(np.asarray(im.crop((j*256,0,(j+1)*256,256))),np.asarray(Image.open(p/name))):raise ValueError(f'Panel mismatch: {p.name} {name}')
        data=torch.load(p/'latents.pt',map_location='cpu',weights_only=True)
        if tuple(data['condition'].shape)!=(128,16,16) or tuple(data['maps'].shape)!=(128,16,48) or not torch.isfinite(data['condition']).all() or not torch.isfinite(data['maps']).all():raise ValueError(f'Invalid latents: {p.name}')
        shape_clip.append(meta['shape_clipped_fraction']);expr_clip.append(meta['expression_clipped_fraction'])
        if shape_clip[-1]>0 or expr_clip[-1]>0:raise ValueError(f'Clipped UV coordinates: {p.name}')
        if (i+1)%200==0:print('VERIFIED',i+1,flush=True)
    if split!={'train':900,'holdout':100}:raise ValueError(f'Invalid splits: {split}')
    report={'count':len(samples),'split':dict(split),'expression_kinds':dict(kinds),'max_expression_transform_error_m':max(errors),
      'max_expression_target_reconstruction_error_m':max(target_errors),
      'expression_max_vertex_motion_mm_median':float(np.median(magnitude)),
      'shape_clip_mean':float(np.mean(shape_clip)),'shape_clip_max':max(shape_clip),
      'expression_clip_mean':float(np.mean(expr_clip)),'expression_clip_max':max(expr_clip),
      'shape_bound_mm':config['shape_bound_mm'],'expression_bound_mm':config['expression_bound_mm']}
    tmp=out/'validation.json.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(out/'validation.json')
    print('VALIDATED',json.dumps(report),flush=True)

if __name__=='__main__':main()
