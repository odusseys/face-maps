"""Measure wider head UV coverage at the same 256px panel size."""
import argparse,json
import numpy as np,torch
from PIL import Image,ImageDraw
from diffusers import AutoencoderKLFlux2
from facekit.paths import DATA,MODEL
from facekit.bootstrap import assets
from facekit.latent import vae_encode,vae_decode
from facekit.uv import ATLAS,panel
from nebius.expressions.canonical_codec import make_panel
from nebius.expressions.debug import render_pair

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',default='head_island');args=parser.parse_args()
    old=DATA/'datasets/faces1000-canonical-expression-256';out=DATA/'runs/canonical-uv-coverage-audit'/args.variant;out.mkdir(parents=True,exist_ok=True)
    candidate=out.parent;variant=out/'pilot';variant.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((old/'dataset.json').read_text());cfg['reference_blend']=str(old/'reference.blend');(variant/'dataset.json').write_text(json.dumps(cfg))
    for name,target in [('reference.npz',candidate/(args.variant+'.npz')),('uv_mapping.npz',candidate/(args.variant+'-uv.npz')),('reference.blend',old/'reference.blend'),('reference_skin.txt',old/'reference_skin.txt')]:
        path=variant/name
        if not path.exists():path.symlink_to(target)
    uv=np.load(variant/'uv_mapping.npz');mapping=tuple(uv[k] for k in ('indices','weights','mask','bbox'));ref=np.load(variant/'reference.npz');ids=np.unique(ref['triangle_vertices'])
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=torch.float32).to('cuda').eval();blender,_,_=assets()
    grid=Image.new('RGB',(256*5,284*2),'black');draw=ImageDraw.Draw(grid);results=[]
    for row,ident in enumerate(('00000','00114')):
        p=old/'samples'/ident;v=variant/'samples'/ident;v.mkdir(parents=True,exist_ok=True)
        meta=json.loads((p/'complete.json').read_text());(v/'complete.json').write_text(json.dumps(meta));geom=np.load(p/'geometry.npz')
        np.savez_compressed(v/'geometry.npz',shape_m=geom['shape_m'],expression_m=geom['expression_m'])
        texture=Image.open(meta['texture']).convert('RGB').resize((ATLAS,ATLAS),Image.Resampling.LANCZOS)
        albedo=panel(texture,mapping[2],mapping[3]).resize((256,256),Image.Resampling.LANCZOS)
        shape,cs=make_panel(geom['shape_m'],mapping,cfg['shape_bound_mm']);expr,ce=make_panel(geom['expression_m'],mapping,cfg['expression_bound_mm'])
        if cs>0 or ce>0:raise ValueError(f'Clipped candidate maps {ident}: {cs} {ce}')
        maps=Image.new('RGB',(768,256))
        for j,im in enumerate((albedo,shape,expr)):maps.paste(im,(j*256,0))
        maps.save(v/'maps.png');orig=Image.open(p/'maps.png').convert('RGB')
        cases=[('old PNG',old,p,orig),('wide PNG',variant,v,maps),('old VAE',old,p,vae_decode(vae,vae_encode(vae,orig).unsqueeze(0))[0]),('wide VAE',variant,v,vae_decode(vae,vae_encode(vae,maps).unsqueeze(0))[0])]
        images=[];gt_image=None
        for label,ds,sample,img in cases:
            work=out/ident/label.replace(' ','-');faces=render_pair(ds,sample,img,blender,work,cfg);gt_image=faces[2] if gt_image is None else gt_image;images.append(faces[3])
            gt=np.load(work/'gt_expression.npy');pred=np.load(work/'pred_expression.npy');err=np.linalg.norm(gt[ids]-pred[ids],axis=1)*1000
            roi=np.s_[75:210,60:196,:];rgb=np.mean(np.abs(np.asarray(gt_image,np.float32)[roi]-np.asarray(faces[3],np.float32)[roi]))
            result={'sample':ident,'case':label,'head_vertex_error_mm_mean':float(err.mean()),'head_vertex_error_mm_p95':float(np.percentile(err,95)),'face_roi_rgb_mae':float(rgb)}
            results.append(result);print('WIDE_UV_RESULT',json.dumps(result),flush=True)
        for col,(label,img) in enumerate(zip(['GT','old PNG','wide PNG','old VAE','wide VAE'],[gt_image]+images)):
            draw.text((col*256+4,row*284+5),ident+' '+label,fill='white');grid.paste(img,(col*256,row*284+28))
    grid.save(out/'comparison.png');(out/'results.json').write_text(json.dumps(results,indent=2))
    print('WIDE_UV_AUDIT_READY',out,flush=True)

if __name__=='__main__':main()
