"""Create additional head-map examples for a small reconstruction study."""
import json
import numpy as np
from PIL import Image
from facekit.paths import DATA
from facekit.uv import ATLAS,panel
from nebius.expressions.canonical_codec import ENCODING_HEAD,ENCODING_HEAD_SHAPE,make_panel

def main():
    old=DATA/'datasets/faces1000-canonical-expression-256';base=DATA/'runs/canonical-uv-coverage-audit';pilot=base/'head-mixed-pilot';pilot.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((old/'dataset.json').read_text());scales=json.loads((base/'head-scales.json').read_text())
    cfg.update({'reference_blend':str(old/'reference.blend'),'encoding':ENCODING_HEAD,'shape_encoding':ENCODING_HEAD_SHAPE,'expression_encoding':ENCODING_HEAD,
                'shape_bound_mm':scales['shape_m']['q97_mm'],'expression_bound_mm':scales['expression_m']['q97_mm']})
    (pilot/'dataset.json').write_text(json.dumps(cfg,indent=2))
    for name,target in [('reference.npz',base/'head_island.npz'),('uv_mapping.npz',base/'head_island-uv.npz'),('reference.blend',old/'reference.blend'),('reference_skin.txt',old/'reference_skin.txt')]:
        link=pilot/name
        if not link.exists():link.symlink_to(target)
    uv=np.load(pilot/'uv_mapping.npz');mapping=tuple(uv[k] for k in ('indices','weights','mask','bbox'))
    for ident in ('00000','00001','00002','00003','00114','00390','00420','00831'):
        src=old/'samples'/ident;dst=pilot/'samples'/ident;dst.mkdir(parents=True,exist_ok=True)
        meta=json.loads((src/'complete.json').read_text());(dst/'complete.json').write_text(json.dumps(meta));geom=np.load(src/'geometry.npz')
        np.savez_compressed(dst/'geometry.npz',shape_m=geom['shape_m'],expression_m=geom['expression_m'])
        texture=Image.open(meta['texture']).convert('RGB').resize((ATLAS,ATLAS),Image.Resampling.LANCZOS)
        albedo=panel(texture,mapping[2],mapping[3]).resize((256,256),Image.Resampling.LANCZOS)
        shape,cs=make_panel(geom['shape_m'],mapping,cfg['shape_bound_mm'],ENCODING_HEAD_SHAPE);expr,ce=make_panel(geom['expression_m'],mapping,cfg['expression_bound_mm'],ENCODING_HEAD)
        if cs>0 or ce>0:raise ValueError(f'Clipped {ident}')
        image=Image.new('RGB',(768,256))
        for j,im in enumerate((albedo,shape,expr)):image.paste(im,(j*256,0))
        image.save(dst/'maps.png')
    print('HEAD_PILOT_READY',pilot,flush=True)

if __name__=='__main__':main()
