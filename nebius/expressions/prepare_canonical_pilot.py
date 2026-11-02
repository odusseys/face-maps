"""Build six temporary canonical targets to audit before full regeneration."""
import json
import numpy as np
from PIL import Image
from facekit.paths import DATA
from nebius.expressions.canonical_codec import ENCODING,make_panel

def main():
    old=DATA/'datasets/faces1000-shape-expression-256';pilot=DATA/'runs/expression-canonical-pilot'
    config=json.loads((old/'dataset.json').read_text());config['encoding']=ENCODING
    (pilot/'dataset.json').write_text(json.dumps(config,indent=2))
    for name in ('reference.npz','reference.blend','reference_skin.txt','uv_mapping.npz'):
        link=pilot/name
        if not link.exists():link.symlink_to(old/name)
    uv=np.load(old/'uv_mapping.npz');mapping=tuple(uv[k] for k in ('indices','weights','mask','bbox'))
    for i in range(6):
        ident=f'{i:05d}';p=pilot/'samples'/ident;src=old/'samples'/ident
        meta=json.loads((src/'complete.json').read_text());render=json.loads((p/'render_complete.json').read_text())
        with np.load(src/'geometry.npz') as geom,np.load(p/'canonical_expression.npz') as expr:
            shape=geom['shape_m'].copy();expression=expr['expression_delta_m'].copy()
        np.savez_compressed(p/'geometry.npz',shape_m=shape,expression_m=expression)
        shape_img,shape_clip=make_panel(shape,mapping,config['shape_bound_mm']);expr_img,expr_clip=make_panel(expression,mapping,config['expression_bound_mm'])
        texture=Image.open(src/'albedo.png').convert('RGB');maps=Image.new('RGB',(768,256))
        for j,img in enumerate((texture,shape_img,expr_img)):maps.paste(img,(j*256,0))
        maps.save(p/'maps.png')
        meta.update({'expression':render,'shape_clipped_fraction':shape_clip,'expression_clipped_fraction':expr_clip})
        (p/'complete.json').write_text(json.dumps(meta,indent=2))
    print('CANONICAL_PILOT_TARGETS_READY',pilot,flush=True)

if __name__=='__main__':main()
