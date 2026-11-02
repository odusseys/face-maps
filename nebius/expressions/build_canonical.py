"""Build separately normalized 256px albedo, neutral shape, and expression maps."""
import json,shutil
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from facekit.paths import DATA
from facekit.uv import ATLAS,panel
from nebius.expressions.canonical_codec import ENCODING,RANGE_MULTIPLIER,make_panel

def atomic(path,data):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def main():
    root=DATA/'datasets';raw=root/'faces1000-visibleface-centered';neutral=root/'faces1000-visibleface-uv-paired'
    out=root/'faces1000-canonical-expression-256'; paths=[out/'samples'/f'{i:05d}' for i in range(1000)]
    if any(not (p/'render_complete.json').exists() for p in paths):raise ValueError('All 1000 expression renders required')
    ref=np.load(raw/'reference.npz')['vertices_m']; uv=np.load(raw/'uv_mapping.npz')
    mask,bbox=uv['mask'],uv['bbox']; indices,weights=uv['indices'][mask],uv['weights'][mask]
    records=[]
    for p in paths:
        src=raw/'samples'/p.name; old=neutral/'samples'/p.name
        metadata=json.loads((old/'complete.json').read_text());render=json.loads((p/'render_complete.json').read_text())
        with np.load(src/'geometry.npz') as r, np.load(old/'geometry.npz') as s, np.load(p/'raw_expression.npz') as e,np.load(p/'canonical_expression.npz') as c:
            shape=s['displacement_m'].copy()
            expression=c['expression_delta_m'].copy()
            aligned=e['expression_delta_m']@r['rotation']@s['rotation']*float(s['similarity_scale'])
            if np.max(np.abs(aligned-expression))>1e-6:raise ValueError(f'Expression reconstruction mismatch: {p}')
        if shape.shape!=ref.shape or expression.shape!=ref.shape or not np.isfinite(expression).all():raise ValueError(p)
        records.append((p,metadata,render,shape,expression))
    train=[r for r in records if r[1]['split']=='train']; holdout=[r for r in records if r[1]['split']=='holdout']
    if (len(train),len(holdout))!=(900,100):raise ValueError('Unexpected split')
    width=len(indices)*3;count=len(train)*width
    def bound(column):
        values=np.empty(count,dtype=np.float32)
        for i,r in enumerate(train):values[i*width:(i+1)*width]=np.abs(np.einsum('pi,pic->pc',weights,r[column][indices])*1000).reshape(-1)
        pos=.97*(count-1);lo,hi=int(np.floor(pos)),int(np.ceil(pos));values.partition((lo,hi))
        value=float(values[lo]+(pos-lo)*(values[hi]-values[lo]));del values
        if not np.isfinite(value) or value<=0:raise ValueError('Invalid normalization bound')
        return value
    shape_bound=bound(3);expr_bound=bound(4)
    previous=json.loads((neutral/'dataset.json').read_text())['bound_mm']
    if not np.isclose(shape_bound,previous,rtol=1e-6):raise ValueError(f'Neutral bounds differ: {shape_bound}, {previous}')
    print('NORMALIZATION',json.dumps({'shape_bound_mm':shape_bound,'expression_bound_mm':expr_bound,'components_each':count}),flush=True)
    config={'version':1,'count':1000,'train':900,'holdout':100,'panel_size':[256,256],'map_size':[768,256],
      'portrait_size':[256,256],'panel_order':['albedo','neutral_shape','expression'],
      'source':str(raw),'neutral_source':str(neutral),'geometry':'reference + neutral shape + expression',
      'expression_transform':'expression units in common canonical coordinates; inverse neutral alignment applied during rendering',
      'encoding':ENCODING,'encoding_range_scale_multiple':RANGE_MULTIPLIER,'normalization_quantile':.97,'normalization_population':'train UV pixel XYZ components',
      'normalization_components':count,'shape_bound_mm':shape_bound,'expression_bound_mm':expr_bound,
      'expression_pack':'faceunits01','expression_pack_sha256':'d113107bd7eb59f3af4df6fc0ec29bfcc593f496d0b336aec14f086a80ce7146','expression_seed':20261102,'expressions_per_identity':1,
      'model':json.loads((neutral/'dataset.json').read_text())['model'],
      'model_revision':json.loads((neutral/'dataset.json').read_text())['model_revision']}
    if (out/'dataset.json').exists() and json.loads((out/'dataset.json').read_text())!=config:raise ValueError('Dataset configuration changed')
    atomic(out/'dataset.json',config)
    for name in ('reference.npz','reference.blend','reference_skin.txt','uv_mapping.npz'):
        if not (out/name).exists():shutil.copy2(raw/name,out/name)
    @lru_cache(maxsize=32)
    def texture(path):
        image=Image.open(path).convert('RGB').resize((ATLAS,ATLAS),Image.Resampling.LANCZOS)
        return panel(image,mask,bbox).resize((256,256),Image.Resampling.LANCZOS)
    mapping=tuple(uv[k] for k in ('indices','weights','mask','bbox'))
    def rgb(image):
        rgba=image.convert('RGBA');background=Image.new('RGBA',rgba.size,(0,0,0,255));background.alpha_composite(rgba)
        return background.convert('RGB')
    for i,(p,meta,render,shape,expr) in enumerate(records,1):
        if (p/'complete.json').exists():continue
        for name in ('portrait_expression_hair.png','portrait_expression_bare.png'):
            im=rgb(Image.open(p/name))
            if im.size!=(256,256):raise ValueError(f'Invalid render: {p/name}')
            im.save(p/name)
        rgb(Image.open(raw/'samples'/p.name/'portrait_bare.png')).resize((256,256),Image.Resampling.LANCZOS).save(p/'portrait_neutral_bare.png')
        albedo=texture(meta['texture']);shape_img,shape_clip=make_panel(shape,mapping,shape_bound);expr_img,expr_clip=make_panel(expr,mapping,expr_bound)
        if shape_clip>0 or expr_clip>0:raise ValueError(f'Wider map encoding still clips: {p.name}')
        albedo.save(p/'albedo.png');shape_img.save(p/'shape.png');expr_img.save(p/'expression.png')
        joined=Image.new('RGB',(768,256))
        for j,im in enumerate((albedo,shape_img,expr_img)):joined.paste(im,(j*256,0))
        joined.save(p/'maps.png')
        np.savez_compressed(p/'geometry.npz',shape_m=shape,expression_m=expr)
        identity={k:meta[k] for k in ('seed','gender','age','race_weights','targets','camera')}
        atomic(p/'complete.json',{'index':meta['index'],'split':meta['split'],'texture':meta['texture'],
            'skin':meta['skin'],'hair':meta['hair'],'identity':identity,'expression':render,'shape_clipped_fraction':shape_clip,
            'expression_clipped_fraction':expr_clip,'shape_bound_mm':shape_bound,'expression_bound_mm':expr_bound})
        if i%50==0:print('BUILT',i,flush=True)
    atomic(out/'maps_complete.json',{'count':len(records),'train':len(train),'holdout':len(holdout)})
    print('MAPS_READY',out,flush=True)

if __name__=='__main__':main()
