"""Evaluate a fixed real portrait without claiming unknown ground-truth maps."""
import os,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from facekit.paths import DATA
from facekit.uv import ATLAS,panel,inverse_panel,decode_displacement

EXTERNAL=DATA/'eval/reference-portrait/aligned.png'

def render_predictions(dataset,prediction,blender,work,config):
    dataset=Path(dataset);work=Path(work);work.mkdir(parents=True,exist_ok=True)
    if prediction.size!=(768,256):raise ValueError('Expected three 256px panels')
    ref=np.load(dataset/'reference.npz');uv=np.load(dataset/'uv_mapping.npz');mask=uv['mask'];bbox=uv['bbox']
    def invert(im):return inverse_panel(im.resize((512,512),Image.Resampling.BILINEAR),bbox)
    coverage=np.asarray(invert(panel(Image.new('RGB',(ATLAS,ATLAS),'white'),mask,bbox).resize((256,256),Image.Resampling.LANCZOS)),np.float32)/255
    def uncomposite(im):return np.clip(np.asarray(invert(im),np.float32)/np.maximum(coverage,1/255),0,255)
    nearest=distance_transform_edt(~mask,return_distances=False,return_indices=True)
    def vertices(im,bound,encoding):
        if encoding in ('signed_log1p_q97_range4','signed_log1p_q97_range12','signed_log1p_q97_range24'):
            from nebius.expressions.canonical_codec import decode
            field=decode(uncomposite(im),bound,encoding)
        else:field=decode_displacement(uncomposite(im),bound,encoding)
        delta=np.zeros_like(ref['vertices_m']);count=np.zeros(len(delta),np.float32)
        for tri,coords in zip(ref['triangle_vertices'],ref['triangle_uv']):
            x=np.clip(np.rint(coords[:,0]*ATLAS-.5).astype(int),0,ATLAS-1);y=np.clip(np.rint((1-coords[:,1])*ATLAS-.5).astype(int),0,ATLAS-1)
            np.add.at(delta,tri,field[nearest[0,y,x],nearest[1,y,x]]);np.add.at(count,tri,1)
        good=count>0;delta[good]/=count[good,None];return delta
    sections=[prediction.crop((i*256,0,(i+1)*256,256)) for i in range(3)]
    shape=vertices(sections[1],config['shape_bound_mm'],config.get('shape_encoding',config['encoding']))
    expression=vertices(sections[2],config['expression_bound_mm'],config.get('expression_encoding',config['encoding']));base=ref['vertices_m']
    np.save(work/'pred_neutral.npy',base+shape);np.save(work/'pred_expression.npy',base+shape+expression)
    texture=Image.open((dataset/'reference_skin.txt').read_text().strip()).convert('RGB').resize((ATLAS,ATLAS),Image.Resampling.LANCZOS)
    raw=np.asarray(texture,np.float32);inferred=uncomposite(sections[0]);color_mask=uv['color_mask'] if 'color_mask' in uv else mask
    valid=mask&color_mask&(np.min(inferred,axis=2)>8)
    if valid.any():
        scale=np.clip(np.median(inferred[valid],axis=0)/np.maximum(np.median(raw[valid],axis=0),1),.1,3)
        texture=Image.fromarray(np.uint8(np.clip(raw*scale,0,255)))
    texture.paste(Image.fromarray(np.uint8(np.rint(inferred))),(0,0),Image.fromarray(mask.astype('uint8')*255));texture.save(work/'pred_atlas.png')
    env=os.environ.copy();env.update({'BLENDER_USER_RESOURCES':str(work/'blender_user'),'BLENDER_USER_EXTENSIONS':str(work/'blender_extensions'),'BLENDER_USER_CONFIG':str(work/'blender_config')})
    script=Path(__file__).with_name('blender_external_debug.py')
    blend=Path(config.get('reference_blend',dataset/'reference.blend'))
    subprocess.run([str(blender),'-b',str(blend),'-t','6','--python-exit-code','1','--python',str(script),'--',str(work)],env=env,check=True,stdout=subprocess.DEVNULL)
    return [Image.open(work/(name+'.png')).convert('RGB') for name in ('pred_neutral','pred_expression')]

def external_row(dataset,vae,model,embeds,args,config,blender,infer,step):
    import torch
    from facekit.latent import vae_encode,vae_decode
    if not EXTERNAL.is_file():raise FileNotFoundError(f'Aligned external evaluation image missing: {EXTERNAL}')
    portrait=Image.open(EXTERNAL).convert('RGB')
    if portrait.size!=(256,256):raise ValueError('External portrait must be 256 x 256')
    if config.get('latent_refinement'):
        from nebius.expressions.refine_latents import refine,portrait_weight
        refined,_=refine(vae,[portrait],portrait_weight([portrait]),steps=config['latent_refinement']['steps'])
        condition=refined.to('cuda',dtype=torch.bfloat16)
    else:condition=vae_encode(vae,portrait).unsqueeze(0).to('cuda',dtype=torch.bfloat16)
    prediction=vae_decode(vae,infer(model,condition,embeds,args.inference_steps,args.seed+2))[0]
    work=args.output/'debug'/f'{step:07d}'/'external-portrait';work.mkdir(parents=True,exist_ok=True)
    prediction.save(work/'predicted_maps.png');portrait.save(work/'input.png')
    neutral,expressive=render_predictions(dataset,prediction,blender,work,config)
    panels=[prediction.crop((j*256,0,(j+1)*256,256)) for j in range(3)]
    return [None,neutral,portrait,expressive,None,panels[0],None,panels[1],None,panels[2]]

def grid_with_external(rows):
    labels=['Neutral face','Expressive face','Albedo','Shape displacement','Expression displacement']
    if len(rows)!=3:raise ValueError('Expected two dataset examples and one external example')
    size=256;header=28;grid=Image.new('RGB',(size*6,(size+header)*5),'black');draw=ImageDraw.Draw(grid)
    for sample,row in enumerate(rows):
        if len(row)!=10:raise ValueError('Expected five pairs')
        for r,label in enumerate(labels):
            for c in range(2):
                x=(sample*2+c)*size;y=r*(size+header);img=row[r*2+c]
                kind=('Input portrait' if sample==2 and r==1 and c==0 else 'GT unavailable' if sample==2 and c==0 else ('GT '+label if c==0 else 'Pred '+label))
                draw.text((x+4,y+5),kind,fill='white')
                if img is not None:grid.paste(img.convert('RGB').resize((size,size),Image.Resampling.LANCZOS),(x,y+header))
    return grid
