"""Render an inferred UV map on the standard face with no hair."""
import json, os, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from .uv import panel, inverse_panel, decode_displacement, ATLAS


def render_debug(dataset, sample, predicted, blender, work, bound_mm, encoding='signed_log1p_9', texture_path=None):
    dataset=Path(dataset); work=Path(work); work.mkdir(parents=True,exist_ok=True)
    ref=np.load(dataset/'reference.npz'); uvmap=np.load(dataset/'uv_mapping.npz')
    bbox=uvmap['bbox']; mask=uvmap['mask']
    # The panel was composited over black after resizing the UV mask. Undo that
    # coverage before decoding: black encodes -bound, not zero displacement.
    white=Image.new('RGB',(ATLAS,ATLAS),'white')
    coverage=np.asarray(inverse_panel(panel(white,mask,bbox),bbox),dtype=np.float32)/255
    def uncomposite(image):
        pixels=np.asarray(inverse_panel(image,bbox),dtype=np.float32)
        return np.clip(pixels/np.maximum(coverage,1/255),0,255)
    displacement = predicted if texture_path is not None else predicted.crop((512,0,1024,512))
    pred_disp=uncomposite(displacement)
    field=decode_displacement(pred_disp,bound_mm,encoding)
    # Vertex UVs lie on triangle edges; find the closest covered texel for stable sampling.
    nearest=distance_transform_edt(~mask,return_distances=False,return_indices=True)
    delta=np.zeros_like(ref['vertices_m']); count=np.zeros(len(delta),np.float32)
    for tri,uv in zip(ref['triangle_vertices'],ref['triangle_uv']):
        x=np.clip(np.rint(uv[:,0]*ATLAS-.5).astype(int),0,ATLAS-1)
        y=np.clip(np.rint((1-uv[:,1])*ATLAS-.5).astype(int),0,ATLAS-1)
        yy=nearest[0,y,x]; xx=nearest[1,y,x]
        np.add.at(delta,tri,field[yy,xx]); np.add.at(count,tri,1)
    good=count>0; delta[good]/=count[good,None]
    np.save(work/'vertices.npy',ref['vertices_m']+delta)
    if texture_path is not None:
        base=Image.open(texture_path).convert('RGB').resize((ATLAS,ATLAS),Image.Resampling.LANCZOS)
        base.save(work/'atlas.png')
    else:
        texture_pixels=uncomposite(predicted.crop((0,0,512,512)))
        texture=Image.fromarray(np.uint8(np.rint(texture_pixels)))
    # Recolor the neutral skin from the inferred face. This avoids leaking the
    # target texture into the portrait while reducing a visible face/neck seam.
        reference_skin=(dataset/'reference_skin.txt').read_text().strip()
        base=Image.open(reference_skin).convert('RGB').resize((ATLAS,ATLAS),Image.Resampling.LANCZOS)
        base_pixels=np.asarray(base,dtype=np.float32)
        inferred_pixels=np.asarray(texture,dtype=np.float32)
        valid=mask & (np.min(inferred_pixels,axis=2)>8)
        if valid.any():
            reference_color=np.median(base_pixels[valid],axis=0).clip(1,None)
            inferred_color=np.median(inferred_pixels[valid],axis=0)
            scale=np.clip(inferred_color/reference_color,.1,3)
            base=Image.fromarray(np.uint8(np.clip(base_pixels*scale,0,255)))
        base.paste(texture,(0,0),Image.fromarray(mask.astype('uint8')*255)); base.save(work/'atlas.png')
    script=Path(__file__).with_name('blender_debug.py')
    env=os.environ.copy(); env['BLENDER_USER_RESOURCES']=str(work/'blender_user')
    env['BLENDER_USER_EXTENSIONS']=str(work/'blender_extensions')
    env['BLENDER_USER_CONFIG']=str(work/'blender_config')
    subprocess.run([str(blender),'-b',str(dataset/'reference.blend'),'-t','6','--python-exit-code','1',
                    '--python',str(script),'--',str(work)],env=env,check=True)
    return Image.open(work/'predicted_face.png').convert('RGB')
