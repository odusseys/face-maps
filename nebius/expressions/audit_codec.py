"""Compare inverse renders with a q97-scaled, wider-range log encoding."""
import json
import numpy as np
from PIL import Image,ImageDraw
from facekit.paths import DATA
from facekit.bootstrap import assets
from nebius.expressions import debug
from nebius.expressions.canonical_codec import ENCODING,decode,make_panel

def main():
    ds=DATA/'datasets/faces1000-shape-expression-256';out=DATA/'runs/expression-codec-audit';out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ds/'dataset.json').read_text());ref=np.load(ds/'reference.npz');ids=np.unique(ref['triangle_vertices']);uv=np.load(ds/'uv_mapping.npz')
    mapping=tuple(uv[k] for k in ('indices','weights','mask','bbox'));blender,_,_=assets()
    debug.decode_displacement=lambda rgb,scale,encoding:decode(rgb,scale,ENCODING)
    canvas=Image.new('RGB',(256*4,284*6),'black');draw=ImageDraw.Draw(canvas);results=[]
    for row in range(6):
        ident=f'{row:05d}';p=ds/'samples'/ident;geom=np.load(p/'geometry.npz')
        shape,cs=make_panel(geom['shape_m'],mapping,config['shape_bound_mm']);expr,ce=make_panel(geom['expression_m'],mapping,config['expression_bound_mm'])
        texture=Image.open(p/'albedo.png').convert('RGB');maps=Image.new('RGB',(768,256))
        for i,img in enumerate((texture,shape,expr)):maps.paste(img,(i*256,0))
        work=out/ident;faces=debug.render_pair(ds,p,maps,blender,work,config)
        for col,(label,img) in enumerate(zip(['GT neutral','Inverse neutral','GT expression','Inverse expression'],faces)):
            x=col*256;y=row*284;draw.text((x+4,y+5),f'{ident} {label}',fill='white');canvas.paste(img,(x,y+28))
        a=np.load(work/'gt_neutral.npy');b=np.load(work/'pred_neutral.npy');c=np.load(work/'gt_expression.npy');d=np.load(work/'pred_expression.npy')
        shape_err=np.linalg.norm(a[ids]-b[ids],axis=1)*1000;expr_err=np.linalg.norm((c-a)[ids]-(d-b)[ids],axis=1)*1000
        result={'sample':ident,'shape_clip':cs,'expression_clip':ce,'neutral_vertex_error_mm_mean':float(shape_err.mean()),
          'expression_vertex_error_mm_mean':float(expr_err.mean()),'expression_vertex_error_mm_p95':float(np.percentile(expr_err,95))}
        results.append(result);print('CODEC_SAMPLE',json.dumps(result),flush=True)
    canvas.save(out/'comparison.png');(out/'audit.json').write_text(json.dumps(results,indent=2))
    print('CODEC_AUDIT_READY',out,flush=True)

if __name__=='__main__':main()
