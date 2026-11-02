"""Align an external portrait to the neutral synthetic silhouette, without warping."""
import argparse,json,os,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage,optimize
from facekit.paths import DATA
from facekit.bootstrap import assets

def mask(image):
    pixels=np.asarray(image.convert('RGB'));foreground=np.max(pixels,axis=2)>8
    foreground=ndimage.binary_fill_holes(foreground)
    labels,count=ndimage.label(foreground)
    if not count:raise ValueError('No foreground in portrait')
    sizes=np.bincount(labels.ravel());sizes[0]=0
    return labels==np.argmax(sizes)

def transform(image,scale,dx,dy,size,resample):
    c=size/2
    coeff=(1/scale,0,c-(c+dx)/scale,0,1/scale,c-(c+dy)/scale)
    return image.transform((size,size),Image.Transform.AFFINE,coeff,resample=resample,fillcolor=0)

def align(source,reference):
    n=128
    a=Image.fromarray(np.uint8(mask(source.resize((n,n),Image.Resampling.LANCZOS)))*255)
    b=mask(reference.resize((n,n),Image.Resampling.LANCZOS))
    def score(params):
        x=np.asarray(transform(a,*params,n,Image.Resampling.NEAREST))>127
        def iou(x,y):
            union=np.logical_or(x,y).sum()
            return np.logical_and(x,y).sum()/max(union,1)
        return -(0.5*iou(x,b)+0.5*iou(x[:int(n*.75)],b[:int(n*.75)]))
    fit=optimize.differential_evolution(score,[(.35,1.8),(-45,45),(-45,45)],seed=20261102,popsize=12,maxiter=100,tol=.001,polish=True)
    scale,dx,dy=map(float,fit.x)
    source512=source.convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
    aligned=transform(source512,scale,dx*4,dy*4,512,Image.Resampling.BICUBIC).resize((256,256),Image.Resampling.LANCZOS)
    return aligned,{'scale':scale,'translation_256_px':[dx*2,dy*2],'objective_weighted_iou':float(-fit.fun),
                    'alignment':'uniform scale and translation; no face reshaping', 'output_size':[256,256]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=DATA/'eval/reference-portrait');args=parser.parse_args()
    folder=args.directory;source=Image.open(folder/'source.png').convert('RGB')
    dataset=DATA/'datasets/faces1000-shape-expression-256';blender,_,_=assets()
    env=os.environ.copy();env.update({'BLENDER_USER_RESOURCES':str(folder/'blender_user'),'BLENDER_USER_EXTENSIONS':str(folder/'blender_extensions'),'BLENDER_USER_CONFIG':str(folder/'blender_config')})
    ref=folder/'synthetic_white.png'
    subprocess.run([str(blender),'-b',str(dataset/'reference.blend'),'-t','3','--python-exit-code','1','--python',str(Path(__file__).with_name('reference_silhouette_blender.py')),'--',str(ref)],env=env,check=True,stdout=subprocess.DEVNULL)
    raw=Image.open(ref).convert('RGBA');black=Image.new('RGBA',raw.size,(0,0,0,255));black.alpha_composite(raw);reference=black.convert('RGB');reference.save(ref)
    aligned,metadata=align(source,reference);aligned.save(folder/'aligned.png')
    ref256=reference.resize((256,256),Image.Resampling.LANCZOS)
    overlay=Image.blend(ref256.convert('RGB'),aligned,.5)
    canvas=Image.new('RGB',(256*3,284),'black');draw=ImageDraw.Draw(canvas)
    for i,(label,image) in enumerate([('Synthetic silhouette',ref256),('Aligned portrait',aligned),('Overlay',overlay)]):
        draw.text((i*256+4,5),label,fill='white');canvas.paste(image,(i*256,28))
    canvas.save(folder/'alignment-preview.png')
    (folder/'alignment.json').write_text(json.dumps(metadata,indent=2))
    print('ALIGNMENT_READY',json.dumps(metadata),flush=True)

if __name__=='__main__':main()
