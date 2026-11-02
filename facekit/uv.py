"""Interpolate XYZ displacements on the fixed MakeHuman face UV island."""
import numpy as np
from PIL import Image

ATLAS = 1024


def raster_reference(reference):
    tris, uvs = reference['triangle_vertices'], reference['triangle_uv']
    index = np.full((ATLAS,ATLAS,3), -1, dtype=np.int32)
    weight = np.zeros((ATLAS,ATLAS,3),dtype=np.float32)
    for ids, uv in zip(tris,uvs):
        xy=np.stack([uv[:,0]*ATLAS-.5,(1-uv[:,1])*ATLAS-.5],axis=-1)
        (x0,y0),(x1,y1),(x2,y2)=xy
        den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(den)<1e-9: continue
        xmin=max(0,int(np.ceil(xy[:,0].min()))); xmax=min(ATLAS-1,int(np.floor(xy[:,0].max())))
        ymin=max(0,int(np.ceil(xy[:,1].min()))); ymax=min(ATLAS-1,int(np.floor(xy[:,1].max())))
        if xmin>xmax or ymin>ymax: continue
        y,x=np.mgrid[ymin:ymax+1,xmin:xmax+1]
        a=((y1-y2)*(x-x2)+(x2-x1)*(y-y2))/den
        b=((y2-y0)*(x-x2)+(x0-x2)*(y-y2))/den
        c=1-a-b; inside=(a>=-1e-8)&(b>=-1e-8)&(c>=-1e-8)
        index[y[inside],x[inside]]=ids
        weight[y[inside],x[inside]]=np.stack([a[inside],b[inside],c[inside]],axis=1)
    mask=index[:,:,0]>=0
    if mask.sum()<1000: raise RuntimeError('Face UV rasterization produced insufficient coverage')
    y,x=np.where(mask); bbox=(max(x.min()-14,0), max(y.min()-14,0), min(x.max()+15,ATLAS), min(y.max()+15,ATLAS))
    return index,weight,mask,bbox


def layout(bbox):
    x0,y0,x1,y1=map(int,bbox)
    w,h=y1-y0,x1-x0  # crop rotated clockwise
    scale=min(512/w,512/h)
    w,h=round(w*scale),round(h*scale)
    return w,h,(512-w)//2,(512-h)//2


def panel(image,mask,bbox):
    w,h,x,y=layout(bbox)
    rgba=image.convert('RGBA'); rgba.putalpha(Image.fromarray(mask.astype('uint8')*255))
    crop=rgba.crop(tuple(map(int,bbox))).transpose(Image.Transpose.ROTATE_270)
    crop=crop.resize((w,h),Image.Resampling.BILINEAR)
    result=Image.new('RGBA',(512,512),(0,0,0,255)); result.alpha_composite(crop,(x,y))
    return result.convert('RGB')


def inverse_panel(image,bbox):
    w,h,x,y=layout(bbox); x0,y0,x1,y1=map(int,bbox)
    crop=image.convert('RGB').crop((x,y,x+w,y+h)).resize((y1-y0,x1-x0),Image.Resampling.BILINEAR)
    crop=crop.transpose(Image.Transpose.ROTATE_90)
    atlas=Image.new('RGB',(ATLAS,ATLAS)); atlas.paste(crop,(x0,y0))
    return atlas


def encode_displacement(field, bound_mm, encoding):
    magnitude=np.minimum(np.abs(field)/bound_mm,1)
    if encoding=='signed_linear':
        encoded=.5+.5*np.sign(field)*magnitude
    elif encoding=='signed_log1p_9':
        encoded=.5+.5*np.sign(field)*np.log1p(9*magnitude)/np.log(10)
    else: raise ValueError(f'Unknown displacement encoding: {encoding}')
    return np.uint8(np.round(np.clip(encoded,0,1)*255))


def make_maps(texture, displacement_m, mapping, bound_mm, encoding='signed_log1p_9'):
    index, weights, mask, bbox=mapping
    field=np.zeros((ATLAS,ATLAS,3),dtype=np.float32)
    field[mask]=np.einsum('pi,pic->pc',weights[mask],displacement_m[index[mask]])*1000
    encoded=encode_displacement(field,bound_mm,encoding)
    tex=Image.open(texture).convert('RGB').resize((ATLAS,ATLAS),Image.Resampling.LANCZOS)
    texture_panel=panel(tex,mask,bbox); displacement_panel=panel(Image.fromarray(encoded),mask,bbox)
    maps=Image.new('RGB',(1024,512)); maps.paste(texture_panel,(0,0)); maps.paste(displacement_panel,(512,0))
    return texture_panel,displacement_panel,maps,float(np.mean(np.abs(field[mask])>bound_mm))


def decode_displacement(rgb,bound_mm,encoding='signed_log1p_9'):
    signed=np.asarray(rgb,dtype=np.float32)/127.5-1
    if encoding=='signed_linear': return signed*bound_mm/1000
    if encoding=='signed_log1p_9': return np.sign(signed)*np.expm1(np.abs(signed)*np.log(10))/9*bound_mm/1000
    raise ValueError(f'Unknown displacement encoding: {encoding}')
