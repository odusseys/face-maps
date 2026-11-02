"""Compare GT and predicted shape/expression UVs and hairless 256px renders."""
import os,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from facekit.uv import ATLAS,panel,inverse_panel,decode_displacement

def render_pair(dataset,sample,predicted,blender,work,config):
    dataset=Path(dataset);sample=Path(sample);work=Path(work);work.mkdir(parents=True,exist_ok=True)
    if predicted.size!=(768,256):raise ValueError('Expected three 256px panels')
    ref=np.load(dataset/'reference.npz');uv=np.load(dataset/'uv_mapping.npz');mask=uv['mask'];bbox=uv['bbox']
    def invert(im):return inverse_panel(im.resize((512,512),Image.Resampling.BILINEAR),bbox)
    white=Image.new('RGB',(ATLAS,ATLAS),'white')
    coverage=np.asarray(invert(panel(white,mask,bbox).resize((256,256),Image.Resampling.LANCZOS)),np.float32)/255
    def uncomposite(im):return np.clip(np.asarray(invert(im),np.float32)/np.maximum(coverage,1/255),0,255)
    nearest=distance_transform_edt(~mask,return_distances=False,return_indices=True)
    def vertices(panel_image,bound,encoding):
        if encoding in ('signed_log1p_q97_range4','signed_log1p_q97_range12','signed_log1p_q97_range24'):
            from nebius.expressions.canonical_codec import decode
            field=decode(uncomposite(panel_image),bound,encoding)
        else:field=decode_displacement(uncomposite(panel_image),bound,encoding)
        delta=np.zeros_like(ref['vertices_m']);count=np.zeros(len(delta),np.float32)
        for tri,coords in zip(ref['triangle_vertices'],ref['triangle_uv']):
            x=np.clip(np.rint(coords[:,0]*ATLAS-.5).astype(int),0,ATLAS-1)
            y=np.clip(np.rint((1-coords[:,1])*ATLAS-.5).astype(int),0,ATLAS-1)
            np.add.at(delta,tri,field[nearest[0,y,x],nearest[1,y,x]]);np.add.at(count,tri,1)
        good=count>0;delta[good]/=count[good,None]
        return delta
    sections=[predicted.crop((j*256,0,(j+1)*256,256)) for j in range(3)]
    pred_shape=vertices(sections[1],config['shape_bound_mm'],config.get('shape_encoding',config['encoding']))
    pred_expr=vertices(sections[2],config['expression_bound_mm'],config.get('expression_encoding',config['encoding']))
    geom=np.load(sample/'geometry.npz');gt_shape=geom['shape_m'];gt_expr=geom['expression_m']
    base=ref['vertices_m']
    for name,coords in [('gt_neutral',base+gt_shape),('pred_neutral',base+pred_shape),('gt_expression',base+gt_shape+gt_expr),('pred_expression',base+pred_shape+pred_expr)]:np.save(work/(name+'.npy'),coords)
    import json
    meta=json.loads((sample/'complete.json').read_text())
    Image.open(meta['texture']).convert('RGB').resize((ATLAS,ATLAS),Image.Resampling.LANCZOS).save(work/'gt_atlas.png')
    base_texture=Image.open((dataset/'reference_skin.txt').read_text().strip()).convert('RGB').resize((ATLAS,ATLAS),Image.Resampling.LANCZOS)
    raw=np.asarray(base_texture,np.float32);inferred=uncomposite(sections[0]);color_mask=uv['color_mask'] if 'color_mask' in uv else mask
    valid=mask&color_mask&(np.min(inferred,axis=2)>8)
    if valid.any():
        scale=np.clip(np.median(inferred[valid],axis=0)/np.maximum(np.median(raw[valid],axis=0),1),.1,3)
        base_texture=Image.fromarray(np.uint8(np.clip(raw*scale,0,255)))
    base_texture.paste(Image.fromarray(np.uint8(np.rint(inferred))),(0,0),Image.fromarray(mask.astype('uint8')*255));base_texture.save(work/'pred_atlas.png')
    env=os.environ.copy();env.update({'BLENDER_USER_RESOURCES':str(work/'blender_user'),'BLENDER_USER_EXTENSIONS':str(work/'blender_extensions'),'BLENDER_USER_CONFIG':str(work/'blender_config')})
    blend=Path(config.get('reference_blend',dataset/'reference.blend'))
    subprocess.run([str(blender),'-b',str(blend),'-t','6','--python-exit-code','1','--python',str(Path(__file__).with_name('blender_debug.py')),'--',str(work)],env=env,check=True,stdout=subprocess.DEVNULL)
    return [Image.open(work/(name+'.png')).convert('RGB') for name in ('gt_neutral','pred_neutral','gt_expression','pred_expression')]

def make_grid(rows):
    labels=['Neutral face','Expressive face','Albedo','Shape displacement','Expression displacement']
    size=256;header=28;grid=Image.new('RGB',(size*2*len(rows),(size+header)*len(labels)),'black');draw=ImageDraw.Draw(grid)
    for sample,row in enumerate(rows):
        if len(row)!=2*len(labels):raise ValueError('Expected five GT/prediction pairs')
        for r,label in enumerate(labels):
            for c,kind in enumerate(('GT','Pred')):
                x=(sample*2+c)*size;y=r*(size+header);draw.text((x+4,y+5),f'{kind} {label}',fill='white')
                grid.paste(row[2*r+c].convert('RGB').resize((size,size),Image.Resampling.LANCZOS),(x,y+header))
    return grid
