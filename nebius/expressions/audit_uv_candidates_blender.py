"""Extract alternative head triangle selections for a UV coverage audit."""
import sys
from pathlib import Path
import bpy,numpy as np

out=Path(sys.argv[sys.argv.index('--')+1]);reference=np.load(out/'reference.npz');xyz=reference['vertices_m'];top=float(xyz[:,2].max())
h=bpy.data.objects['Human'];mesh=h.data;mesh.calc_loop_triangles();v=[];t=[]
for tri in mesh.loop_triangles:
    ids=np.asarray(tri.vertices)
    if np.any(ids>=len(xyz)):continue
    uv=np.asarray([mesh.uv_layers.active.data[i].uv[:] for i in tri.loops])
    v.append(ids);t.append(uv)
v=np.asarray(v);t=np.asarray(t);z=xyz[v,2];y=xyz[v,1];u=t.mean(axis=1)
for name,inside in [('front_wide',(z>top-.27).all(1)&(y<.04).all(1)),('all_head',(z>top-.27).all(1)),
                    ('front_island',(z>top-.27).all(1)&(y<.04).all(1)&(u[:,0]>.6)&(u[:,1]>.2)&(u[:,1]<.8)),
                    ('head_island',(z>top-.27).all(1)&(u[:,0]>.6)&(u[:,1]>.2)&(u[:,1]<.8))]:
    tv=v[inside];tu=t[inside];np.savez_compressed(out/(name+'.npz'),vertices_m=xyz,triangle_vertices=tv,triangle_uv=tu)
    print('UV_CANDIDATE',name,'triangles',len(tv),'vertices',len(np.unique(tv)),'uv_min',tu.reshape(-1,2).min(0),'uv_max',tu.reshape(-1,2).max(0),flush=True)
