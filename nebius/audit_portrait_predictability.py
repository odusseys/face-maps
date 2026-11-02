"""Test whether stored portrait latents preserve displacement information."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
root=Path(__import__('os').environ['FACE_SHARED_ROOT'])/'face_maps'
parser=argparse.ArgumentParser()
parser.add_argument('--dataset',type=Path,default=root/'datasets/faces500-centered-linear-q97')
parser.add_argument('--output',type=Path,default=root/'audits/portrait-latent-predictability-20261101.json')
args=parser.parse_args();ds=args.dataset
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from facekit.uv import panel,ATLAS
uv=np.load(ds/'uv_mapping.npz');white=panel(Image.new('RGB',(ATLAS,ATLAS),'white'),uv['mask'],uv['bbox'])
mask=np.asarray(white.resize((32,32),Image.Resampling.BILINEAR))[:,:,0]>=254
mask=np.repeat(mask[:,:,None],3,axis=2).reshape(-1)
records=[(p.parent,json.loads(p.read_text())['split']) for p in sorted((ds/'samples').glob('*/complete.json'))]
train=[p for p,s in records if s=='train'][::2];hold=[p for p,s in records if s=='holdout'];paths=train+hold
features=[];targets=[]
for path in paths:
    x=torch.load(path/'latents.pt',map_location='cpu',weights_only=True)['face_bare'].float()
    features.append(F.avg_pool2d(x.unsqueeze(0),4).reshape(-1).numpy())
    targets.append(np.asarray(Image.open(path/'displacement.png').convert('RGB').resize((32,32),Image.Resampling.BILINEAR),np.float32).reshape(-1)[mask])
x=np.stack(features);y=np.stack(targets);rng=np.random.default_rng(934)
train_i=rng.permutation(len(train));fit=train_i[:160];val=train_i[160:];test=np.arange(len(train),len(paths))
def predict(fit,query,alpha):
    xm=x[fit].mean(0);ym=y[fit].mean(0);a=(x[fit]-xm).astype(np.float64);b=(x[query]-xm).astype(np.float64)
    k=a@a.T/a.shape[1];cross=b@a.T/a.shape[1];lam=alpha*np.trace(k)/len(fit)
    return ym+cross@np.linalg.solve(k+lam*np.eye(len(fit)),y[fit]-ym)
scores={}
for alpha in (.0001,.001,.01,.1,1.,10.):
    scores[str(alpha)]=float(np.mean((predict(fit,val,alpha)-y[val])**2));print('VAL',alpha,scores[str(alpha)],flush=True)
best=min(scores,key=scores.get);pred=predict(train_i,test,float(best));mean=y[train_i].mean(0)
result={'train':len(train),'validation':len(val),'holdout':len(test),'features':'stored portrait latents average pooled to 8x8',
        'validation_mse':scores,'alpha':best,'holdout_mse':float(np.mean((pred-y[test])**2)),
        'holdout_mean_mse':float(np.mean((mean-y[test])**2))}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
print('LATENT_RESULT',json.dumps(result),flush=True)
