"""Inspect actual map-to-mesh inversion without changing a dataset."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from facekit.paths import DATA
from facekit.bootstrap import assets
from nebius.expressions import debug

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,default=DATA/'datasets/faces1000-shape-expression-256')
    parser.add_argument('--output',type=Path,default=DATA/'runs/expression-inverse-audit');parser.add_argument('--samples',nargs='+',default=['00000','00001','00002','00003','00004','00005'])
    args=parser.parse_args();ds=args.dataset;out=args.output;out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ds/'dataset.json').read_text());ref=np.load(ds/'reference.npz');ids=np.unique(ref['triangle_vertices']);blender,_,_=assets()
    if config['encoding']=='signed_log1p_q97_range12':
        from nebius.expressions.canonical_codec import decode
        debug.decode_displacement=lambda rgb,scale,encoding:decode(rgb,scale)
    labels=['Input hair','Original bare','GT neutral mesh','Inverse neutral map','GT expression mesh','Inverse both maps']
    canvas=Image.new('RGB',(256*len(labels),284*len(args.samples)),'black');draw=ImageDraw.Draw(canvas);results=[]
    for row,ident in enumerate(args.samples):
        sample=ds/'samples'/ident;meta=json.loads((sample/'complete.json').read_text());maps=Image.open(sample/'maps.png').convert('RGB')
        work=out/ident;faces=debug.render_pair(ds,sample,maps,blender,work,config)
        images=[Image.open(sample/'portrait_expression_hair.png'),Image.open(sample/'portrait_expression_bare.png'),*faces]
        for col,(label,img) in enumerate(zip(labels,images)):
            x=col*256;y=row*284;draw.text((x+3,y+5),f'{ident} {label}',fill='white');canvas.paste(img.convert('RGB'),(x,y+28))
        a=np.load(work/'gt_neutral.npy');b=np.load(work/'pred_neutral.npy');c=np.load(work/'gt_expression.npy');d=np.load(work/'pred_expression.npy')
        neutral=np.linalg.norm(a[ids]-b[ids],axis=1)*1000;expressive=np.linalg.norm(c[ids]-d[ids],axis=1)*1000
        actual_delta=(c-a)[ids];recon_delta=(d-b)[ids];expression=np.linalg.norm(actual_delta-recon_delta,axis=1)*1000
        result={'sample':ident,'kind':meta['expression']['expression_kind'],'shape_clipped':meta['shape_clipped_fraction'],
          'expression_clipped':meta['expression_clipped_fraction'],'neutral_vertex_error_mm_mean':float(neutral.mean()),
          'expressive_vertex_error_mm_mean':float(expressive.mean()),'expression_vertex_error_mm_mean':float(expression.mean()),
          'expression_vertex_error_mm_p95':float(np.percentile(expression,95))}
        results.append(result);print('AUDIT_SAMPLE',json.dumps(result),flush=True)
    canvas.save(out/'inverse-comparison.png');(out/'audit.json').write_text(json.dumps(results,indent=2))
    print('INVERSE_AUDIT_READY',out,flush=True)

if __name__=='__main__':main()
