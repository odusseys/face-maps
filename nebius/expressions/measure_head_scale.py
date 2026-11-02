"""Compute training-only q97 scales for the wider head UV mapping."""
import json,time
import numpy as np
from facekit.paths import DATA

def main():
    source=DATA/'datasets/faces1000-canonical-expression-256';uv=np.load(DATA/'runs/canonical-uv-coverage-audit/head_island-uv.npz')
    inds=uv['indices'][uv['mask']];w=uv['weights'][uv['mask']]
    paths=sorted((source/'samples').glob('*/complete.json'));train=[p.parent for p in paths if json.loads(p.read_text())['split']=='train']
    if len(train)!=900:raise ValueError('Expected 900 train identities')
    count=len(train)*len(inds)*3;width=len(inds)*3;result={}
    for key in ('shape_m','expression_m'):
        started=time.monotonic();vals=np.empty(count,np.float32)
        for i,p in enumerate(train):
            with np.load(p/'geometry.npz') as g:field=g[key]
            vals[i*width:(i+1)*width]=np.abs(np.einsum('pi,pic->pc',w,field[inds])*1000).ravel()
            if (i+1)%300==0:print('SCALE_PROGRESS',key,i+1,flush=True)
        maximum=float(vals.max());pos=.97*(count-1);lo=int(np.floor(pos));hi=int(np.ceil(pos));vals.partition((lo,hi))
        bound=float(vals[lo]+(pos-lo)*(vals[hi]-vals[lo]));del vals
        result[key]={'q97_mm':bound,'train_max_mm':maximum,'max_to_q97':maximum/bound,'seconds':time.monotonic()-started}
        print('HEAD_SCALE',key,json.dumps(result[key]),flush=True)
    result.update({'train':len(train),'uv_pixels':len(inds),'component_count':count})
    out=DATA/'runs/canonical-uv-coverage-audit/head-scales.json';out.write_text(json.dumps(result,indent=2));print('HEAD_SCALES_READY',out,flush=True)

if __name__=='__main__':main()
