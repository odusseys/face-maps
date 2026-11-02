"""Exercise the 256px debug renderer with real geometry, without model training."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from facekit.paths import DATA
from facekit.bootstrap import assets
from facekit.uv import ATLAS,encode_displacement,panel
from nebius.expressions.debug import render_pair,make_grid

def main():
    raw=DATA/'datasets/faces1000-visibleface-centered';neutral=DATA/'datasets/faces1000-visibleface-uv-paired';expression=DATA/'datasets/faces1000-shape-expression-256'
    temp=DATA/'runs/expression-debug-preflight';sample=temp/'sample';sample.mkdir(parents=True,exist_ok=True)
    for name in ('reference.npz','reference.blend','reference_skin.txt','uv_mapping.npz'):
        link=temp/name
        if not link.exists():link.symlink_to(raw/name)
    ident='00000';src=raw/'samples'/ident;old=neutral/'samples'/ident;new=expression/'samples'/ident
    meta=json.loads((old/'complete.json').read_text());(sample/'complete.json').write_text(json.dumps({'texture':meta['texture']}))
    with np.load(src/'geometry.npz') as a,np.load(old/'geometry.npz') as b,np.load(new/'raw_expression.npz') as e:
        shape=b['displacement_m'].copy();delta=(e['expression_delta_m']@a['rotation']@b['rotation']*float(b['similarity_scale'])).astype(np.float32)
    np.savez_compressed(sample/'geometry.npz',shape_m=shape,expression_m=delta)
    uv=np.load(raw/'uv_mapping.npz');mask=uv['mask'];field=np.zeros((ATLAS,ATLAS,3),np.float32)
    field[mask]=np.einsum('pi,pic->pc',uv['weights'][mask],delta[uv['indices'][mask]])*1000
    expression_panel=panel(Image.fromarray(encode_displacement(field,5.,'signed_linear')),mask,uv['bbox']).resize((256,256),Image.Resampling.LANCZOS)
    target=Image.new('RGB',(768,256));albedo=Image.open(old/'texture.png').resize((256,256),Image.Resampling.LANCZOS)
    shape_img=Image.open(old/'displacement.png').resize((256,256),Image.Resampling.LANCZOS)
    for j,img in enumerate((albedo,shape_img,expression_panel)):target.paste(img,(j*256,0))
    config={'shape_bound_mm':json.loads((neutral/'dataset.json').read_text())['bound_mm'],'expression_bound_mm':5.}
    blender,_,_=assets();faces=render_pair(temp,sample,target,blender,temp/'render',config)
    panels=[]
    for j in range(3):
        image=target.crop((j*256,0,(j+1)*256,256));panels.extend([image,image])
    grid=make_grid([faces+panels]);grid.save(temp/'preflight.png')
    for img in faces:
        if img.size!=(256,256):raise ValueError('Invalid rendered resolution')
    if not np.any(np.asarray(faces[0])!=np.asarray(faces[2])):raise ValueError('Expression did not affect GT render')
    print('DEBUG_PREFLIGHT_READY',temp/'preflight.png',flush=True)

if __name__=='__main__':main()
