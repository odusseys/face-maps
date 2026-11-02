"""Measure and render VAE round trips at several internal resolutions."""
import json
import numpy as np,torch
from PIL import Image,ImageDraw
from diffusers import AutoencoderKLFlux2
from facekit.paths import DATA,MODEL
from facekit.bootstrap import assets
from facekit.latent import vae_encode,vae_decode
from nebius.expressions.debug import render_pair

def main():
    ds=DATA/'datasets/faces1000-canonical-expression-256';out=DATA/'runs/canonical-vae-resolution-study';out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ds/'dataset.json').read_text());ids=np.unique(np.load(ds/'reference.npz')['triangle_vertices'])
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval();blender,_,_=assets()
    samples=['00000','00114','00390'];factors=[1,1.5,2];results=[]
    grid=Image.new('RGB',(256*(2+len(factors)),284*len(samples)),'black');draw=ImageDraw.Draw(grid)
    for row,ident in enumerate(samples):
        sample=ds/'samples'/ident;target=Image.open(sample/'maps.png').convert('RGB');images=[Image.open(sample/'portrait_expression_bare.png').convert('RGB')]
        gt_face=None
        for factor in factors:
            size=(int(768*factor),int(256*factor))
            source=target if factor==1 else target.resize(size,Image.Resampling.LANCZOS)
            latent=vae_encode(vae,source);reconstruction=vae_decode(vae,latent.unsqueeze(0))[0]
            if factor!=1:reconstruction=reconstruction.resize((768,256),Image.Resampling.LANCZOS)
            work=out/ident/f'{factor:g}x';work.mkdir(parents=True,exist_ok=True);reconstruction.save(work/'decoded.png')
            faces=render_pair(ds,sample,reconstruction,blender,work,cfg)
            if gt_face is None:gt_face=faces[2]
            images.append(faces[3]);a=np.load(work/'gt_neutral.npy');b=np.load(work/'pred_neutral.npy');c=np.load(work/'gt_expression.npy');d=np.load(work/'pred_expression.npy')
            shape=np.linalg.norm(a[ids]-b[ids],axis=1)*1000;expr=np.linalg.norm((c-a)[ids]-(d-b)[ids],axis=1)*1000
            image_error=np.asarray(reconstruction,np.float32)-np.asarray(target,np.float32)
            result={'sample':ident,'factor':factor,'latent_shape':list(latent.shape),'shape_error_mm_mean':float(shape.mean()),
                'expression_error_mm_mean':float(expr.mean()),'expression_error_mm_p95':float(np.percentile(expr,95)),
                'map_pixel_mae':[float(np.abs(image_error[:,i*256:(i+1)*256]).mean()) for i in range(3)]}
            results.append(result);print('VAE_RESOLUTION',json.dumps(result),flush=True)
        images.insert(1,gt_face)
        for col,(label,img) in enumerate(zip(['Original bare','GT mesh','VAE 1x','VAE 1.5x','VAE 2x'],images)):
            x=col*256;y=row*284;draw.text((x+3,y+5),ident+' '+label,fill='white');grid.paste(img.convert('RGB'),(x,y+28))
    grid.save(out/'comparison.png');(out/'results.json').write_text(json.dumps(results,indent=2));print('STUDY_READY',out,flush=True)

if __name__=='__main__':main()
