"""Signed log map with a global q97 scale and twelve-scale finite range."""
import numpy as np
from facekit.uv import ATLAS,panel
from PIL import Image

RANGE_MULTIPLIER=12.0
ENCODING='signed_log1p_q97_range12'
ENCODING_HEAD='signed_log1p_q97_range24'
ENCODING_HEAD_SHAPE='signed_log1p_q97_range4'

def multiplier(encoding):
    if encoding==ENCODING:return 12.0
    if encoding==ENCODING_HEAD:return 24.0
    if encoding==ENCODING_HEAD_SHAPE:return 4.0
    raise ValueError(f'Unexpected encoding {encoding}')

def encode(field_mm,scale_mm,encoding=ENCODING):
    signed=np.sign(field_mm)*np.log1p(np.abs(field_mm)/scale_mm)/np.log1p(multiplier(encoding))
    return np.uint8(np.round(np.clip(.5+.5*signed,0,1)*255))

def decode(rgb,scale_mm,encoding=ENCODING):
    signed=np.clip(np.asarray(rgb,dtype=np.float32)/127.5-1,-1,1)
    return np.sign(signed)*np.expm1(np.abs(signed)*np.log1p(multiplier(encoding)))*scale_mm/1000

def make_panel(delta,mapping,scale_mm,encoding=ENCODING):
    indices,weights,mask,bbox=mapping;field=np.zeros((ATLAS,ATLAS,3),np.float32)
    field[mask]=np.einsum('pi,pic->pc',weights[mask],delta[indices[mask]])*1000
    clipped=float(np.mean(np.abs(field[mask])>multiplier(encoding)*scale_mm))
    return panel(Image.fromarray(encode(field,scale_mm,encoding)),mask,bbox).resize((256,256),Image.Resampling.LANCZOS),clipped
