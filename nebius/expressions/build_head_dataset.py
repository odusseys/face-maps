"""Build a separate 256px canonical dataset with wider head UV coverage."""
import json,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from facekit.paths import DATA
from facekit.uv import ATLAS,panel
from nebius.expressions.canonical_codec import ENCODING_HEAD,ENCODING_HEAD_SHAPE,make_panel

SOURCE=DATA/'datasets/faces1000-canonical-expression-256'
OUTPUT=DATA/'datasets/faces1000-canonical-head-256'
AUDIT=DATA/'runs/canonical-uv-coverage-audit'
_mapping=None
_config=None

def atomic(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def initialize():
    global _mapping,_config
    uv=np.load(OUTPUT/'uv_mapping.npz');_mapping=tuple(uv[k] for k in ('indices','weights','mask','bbox'))
    _config=json.loads((OUTPUT/'dataset.json').read_text())

@lru_cache(maxsize=32)
def albedo(texture):
    image=Image.open(texture).convert('RGB').resize((ATLAS,ATLAS),Image.Resampling.LANCZOS)
    return panel(image,_mapping[2],_mapping[3]).resize((256,256),Image.Resampling.LANCZOS)

def build(ident):
    src=SOURCE/'samples'/ident;dst=OUTPUT/'samples'/ident;dst.mkdir(parents=True,exist_ok=True)
    if (dst/'complete.json').exists():return ident
    meta=json.loads((src/'complete.json').read_text())
    with np.load(src/'geometry.npz') as g:
        shape,cs=make_panel(g['shape_m'],_mapping,_config['shape_bound_mm'],ENCODING_HEAD_SHAPE)
        expression,ce=make_panel(g['expression_m'],_mapping,_config['expression_bound_mm'],ENCODING_HEAD)
    if cs>0 or ce>0:raise ValueError(f'Clipped targets for {ident}: {cs}, {ce}')
    tex=albedo(meta['texture']);maps=Image.new('RGB',(768,256))
    for i,(name,img) in enumerate((('albedo.png',tex),('shape.png',shape),('expression.png',expression))):
        img.save(dst/name);maps.paste(img,(i*256,0))
    maps.save(dst/'maps.png')
    for name in ('portrait_expression_hair.png','portrait_expression_bare.png','portrait_neutral_bare.png','geometry.npz','render_complete.json'):
        shutil.copy2(src/name,dst/name)
    meta.update({'shape_clipped_fraction':cs,'expression_clipped_fraction':ce,'shape_bound_mm':_config['shape_bound_mm'],
      'expression_bound_mm':_config['expression_bound_mm'],'source_dataset':str(SOURCE)})
    atomic(dst/'complete.json',meta);return ident

def main():
    if not (SOURCE/'validation.json').is_file():raise ValueError('Canonical source dataset is not validated')
    scales=json.loads((AUDIT/'head-scales.json').read_text());source_cfg=json.loads((SOURCE/'dataset.json').read_text())
    if scales['train']!=900:raise ValueError('Wrong normalization population')
    cfg=dict(source_cfg);cfg.update({'source':str(SOURCE),'encoding':ENCODING_HEAD,'shape_encoding':ENCODING_HEAD_SHAPE,'expression_encoding':ENCODING_HEAD,
      'shape_bound_mm':scales['shape_m']['q97_mm'],'expression_bound_mm':scales['expression_m']['q97_mm'],
      'shape_range_multiple':4,'expression_range_multiple':24,'uv_coverage':'full reference head UV island',
      'normalization_components':scales['component_count'],'normalization_population':'training split head UV pixel XYZ components',
      'latent_refinement':{'steps':80,'learning_rate':.04,'regularization':.0001,'foreground_weight':1.,'background_weight':.1}})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'dataset.json').exists() and json.loads((OUTPUT/'dataset.json').read_text())!=cfg:raise ValueError('Output dataset configuration changed')
    atomic(OUTPUT/'dataset.json',cfg)
    with np.load(SOURCE/'reference.npz') as original,np.load(AUDIT/'head_island.npz') as candidate:
        arrays={k:original[k] for k in original.files};arrays['triangle_vertices']=candidate['triangle_vertices'];arrays['triangle_uv']=candidate['triangle_uv']
        arrays['eye_fit_vertex_ids']=np.unique(original['triangle_vertices']);np.savez_compressed(OUTPUT/'reference.npz',**arrays)
    for src,dst in [(AUDIT/'head_island-uv.npz',OUTPUT/'uv_mapping.npz'),(SOURCE/'reference.blend',OUTPUT/'reference.blend'),
                    (SOURCE/'reference_skin.txt',OUTPUT/'reference_skin.txt'),(DATA/'runs/canonical-eye-audit/reference_eyes.npy',OUTPUT/'reference_eyes.npy')]:
        if not dst.exists():shutil.copy2(src,dst)
    ids=[f'{i:05d}' for i in range(1000)]
    with ProcessPoolExecutor(max_workers=4,initializer=initialize) as pool:
        futures=[pool.submit(build,ident) for ident in ids]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%50==0:print('HEAD_MAPS',i,flush=True)
    atomic(OUTPUT/'maps_complete.json',{'count':1000,'source':str(SOURCE)})
    print('HEAD_MAPS_READY',OUTPUT,flush=True)

if __name__=='__main__':main()
