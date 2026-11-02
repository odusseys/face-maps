"""Compare default, fitted, and true eyes for several rendered faces."""
import json,os,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from facekit.paths import DATA
from facekit.bootstrap import assets
from nebius.expressions.eye_fit import fit_eyes

def main():
    ds=DATA/'datasets/faces1000-canonical-expression-256';raw=DATA/'datasets/faces1000-visibleface-centered'
    old=DATA/'datasets/faces1000-visibleface-uv-paired';out=DATA/'runs/canonical-eye-audit';source=DATA/'runs/head-vae-latent-optimization'
    ref=np.load(ds/'reference.npz');face_ids=np.unique(ref['triangle_vertices']);eyes=np.load(out/'reference_eyes.npy');blender,_,_=assets()
    grid=Image.new('RGB',(256*5,284*2),'black');draw=ImageDraw.Draw(grid);results=[]
    for row,ident in enumerate(('00000','00114')):
        p=source/ident/'step-080';work=out/ident;work.mkdir(parents=True,exist_ok=True)
        with np.load(raw/'samples'/ident/'geometry.npz') as a,np.load(old/'samples'/ident/'geometry.npz') as b:
            world=np.load(out/'samples'/ident/'eyes_world.npy')
            actual=(((world-a['moving_center'])@a['rotation']+a['reference_center']-b['moving_center'])@b['rotation'])*float(b['similarity_scale'])+b['reference_center']
        fitted=fit_eyes(ref['vertices_m'],np.load(p/'pred_neutral.npy'),eyes,face_ids)
        np.save(work/'actual.npy',actual);np.save(work/'fitted.npy',fitted)
        delta0=np.linalg.norm((eyes-actual)*1000,axis=1);delta1=np.linalg.norm((fitted-actual)*1000,axis=1)
        result={'sample':ident,'reference_eye_mean_error_mm':float(delta0.mean()),'fitted_eye_mean_error_mm':float(delta1.mean()),'fitted_eye_p95_error_mm':float(np.percentile(delta1,95))};results.append(result)
        path=lambda name:str((p/name).resolve());specs=[('gt-default',path('gt_expression.npy'),path('gt_atlas.png'),None),
          ('gt-actual',path('gt_expression.npy'),path('gt_atlas.png'),str(work/'actual.npy')),
          ('pred-default',path('pred_expression.npy'),path('pred_atlas.png'),None),
          ('pred-fitted',path('pred_expression.npy'),path('pred_atlas.png'),str(work/'fitted.npy'))]
        (work/'config.json').write_text(json.dumps([{'name':n,'mesh':m,'texture':t,'eyes':e} for n,m,t,e in specs]))
        env=os.environ.copy();env.update({'BLENDER_USER_RESOURCES':str(work/'blender_user'),'BLENDER_USER_EXTENSIONS':str(work/'blender_extensions'),'BLENDER_USER_CONFIG':str(work/'blender_config')})
        subprocess.run([str(blender),'-b',str(ds/'reference.blend'),'-t','6','--python-exit-code','1','--python',str(Path(__file__).with_name('ablation_blender.py')),'--',str(work)],env=env,check=True,stdout=subprocess.DEVNULL)
        images=[Image.open(ds/'samples'/ident/'portrait_expression_bare.png').convert('RGB')]+[Image.open(work/(n+'.png')).convert('RGB') for n,_,_,_ in specs]
        for col,(label,img) in enumerate(zip(['Original bare','GT default eyes','GT true eyes','Pred default eyes','Pred fitted eyes'],images)):
            draw.text((col*256+4,row*284+5),ident+' '+label,fill='white');grid.paste(img,(col*256,row*284+28))
        print('EYE_FIT_RESULT',json.dumps(result),flush=True)
    grid.save(out/'eye-fit-comparison.png');(out/'eye-fit-audit.json').write_text(json.dumps(results,indent=2));print('EYE_FIT_AUDIT_READY',out,flush=True)

if __name__=='__main__':main()
