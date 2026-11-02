"""Separate UV reconstruction, VAE geometry and VAE texture effects."""
import json,os,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from facekit.paths import DATA
from facekit.bootstrap import assets
from nebius.expressions.debug import render_pair

def main():
    ds=DATA/'datasets/faces1000-canonical-expression-256';out=DATA/'runs/canonical-render-factor-audit';out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ds/'dataset.json').read_text());blender,_,_=assets();results=[];grid=Image.new('RGB',(256*7,284*2),'black');draw=ImageDraw.Draw(grid)
    for row,ident in enumerate(('00000','00114')):
        p=ds/'samples'/ident;work=out/ident;ideal=work/'ideal';vae=DATA/'runs/canonical-vae-resolution-study'/ident/'1x'
        render_pair(ds,p,Image.open(p/'maps.png').convert('RGB'),blender,ideal,cfg)
        path=lambda root,name:str((root/name).resolve())
        gt=path(ideal,'gt_expression.npy');uv=path(ideal,'pred_expression.npy');v=path(vae,'pred_expression.npy')
        gttex=path(ideal,'gt_atlas.png');uvtex=path(ideal,'pred_atlas.png');vtex=path(vae,'pred_atlas.png')
        specs=[('gt',gt,gttex),('uv_geom',uv,gttex),('vae_geom',v,gttex),('uv_tex',gt,uvtex),('vae_tex',gt,vtex),('vae_both',v,vtex)]
        work.mkdir(parents=True,exist_ok=True);(work/'config.json').write_text(json.dumps([{'name':n,'mesh':m,'texture':t} for n,m,t in specs]))
        env=os.environ.copy();env.update({'BLENDER_USER_RESOURCES':str(work/'blender_user'),'BLENDER_USER_EXTENSIONS':str(work/'blender_extensions'),'BLENDER_USER_CONFIG':str(work/'blender_config')})
        subprocess.run([str(blender),'-b',str(ds/'reference.blend'),'-t','6','--python-exit-code','1','--python',str(Path(__file__).with_name('ablation_blender.py')),'--',str(work)],env=env,check=True,stdout=subprocess.DEVNULL)
        original=Image.open(p/'portrait_expression_bare.png').convert('RGB');images=[original]+[Image.open(work/(n+'.png')).convert('RGB') for n,_,_ in specs]
        labels=['Original bare','GT mesh+texture','UV geometry','VAE geometry','UV texture','VAE texture','VAE both']
        for col,(label,img) in enumerate(zip(labels,images)):
            x=col*256;y=row*284;draw.text((x+3,y+5),ident+' '+label,fill='white');grid.paste(img,(x,y+28))
        pixels=[np.asarray(x,np.float32) for x in images];gt_im=pixels[1];roi=np.s_[75:210,60:196,:]
        for label,img in zip(labels,pixels):
            if label=='GT mesh+texture':continue
            results.append({'sample':ident,'change':label,'face_roi_rgb_mae':float(np.mean(np.abs(img[roi]-gt_im[roi])))})
        print('ABLATION',ident,json.dumps(results[-6:]),flush=True)
    grid.save(out/'comparison.png');(out/'results.json').write_text(json.dumps(results,indent=2));print('ABLATION_READY',out,flush=True)

if __name__=='__main__':main()
