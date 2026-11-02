"""Verify the rebuilt head maps before full latent refinement finishes."""
import json,random
import numpy as np
from PIL import Image
from facekit.paths import DATA
from nebius.expressions.canonical_codec import make_panel

def main():
    ds=DATA/'datasets/faces1000-canonical-head-256';source=DATA/'datasets/faces1000-canonical-expression-256'
    if json.loads((ds/'maps_complete.json').read_text())['count']!=1000:raise ValueError('Head map build incomplete')
    cfg=json.loads((ds/'dataset.json').read_text());uv=np.load(ds/'uv_mapping.npz');mapping=tuple(uv[k] for k in ('indices','weights','mask','bbox'))
    ids=random.Random(20261102).sample(range(1000),25);results=[]
    for i in ids:
        name=f'{i:05d}';p=ds/'samples'/name;old=source/'samples'/name
        m=json.loads((p/'complete.json').read_text());n=json.loads((old/'complete.json').read_text())
        if any(m[k]!=n[k] for k in ('index','split','texture','skin','hair','expression')):raise ValueError(f'Identity mismatch: {name}')
        with np.load(p/'geometry.npz') as a,np.load(old/'geometry.npz') as b:
            for key in ('shape_m','expression_m'):
                if not np.array_equal(a[key],b[key]):raise ValueError(f'Geometry mismatch: {name} {key}')
            shape,cs=make_panel(a['shape_m'],mapping,cfg['shape_bound_mm'],cfg['shape_encoding'])
            expr,ce=make_panel(a['expression_m'],mapping,cfg['expression_bound_mm'],cfg['expression_encoding'])
        if cs or ce or not np.array_equal(np.asarray(shape),np.asarray(Image.open(p/'shape.png'))) or not np.array_equal(np.asarray(expr),np.asarray(Image.open(p/'expression.png'))):raise ValueError(f'Map mismatch: {name}')
        results.append(name)
    report={'samples_checked':results,'all_geometry_and_maps_match':True,'clipping':0}
    out=DATA/'runs/head-map-preflight.json';out.write_text(json.dumps(report,indent=2));print('HEAD_MAP_PREFLIGHT_PASSED',len(results),flush=True)

if __name__=='__main__':main()
