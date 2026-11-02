"""Render the two predicted meshes for an external portrait at the standard view."""
import sys
from pathlib import Path
import bpy,numpy as np

work=Path(sys.argv[sys.argv.index('--')+1]);h=bpy.data.objects['Human'];matrix=np.asarray(h.matrix_world);inv=np.linalg.inv(matrix[:3,:3])
if h.data.shape_keys:h.shape_key_clear()
nodes=[node for mat in h.data.materials if mat and mat.use_nodes for node in mat.node_tree.nodes
       if node.type=='TEX_IMAGE' and node.image and 'diffuse' in node.image.name.lower()]
if not nodes:raise ValueError('No diffuse texture node')
atlas=bpy.data.images.load(str(work/'pred_atlas.png'),check_existing=True)
for node in nodes:node.image=atlas
scene=bpy.context.scene;scene.cycles.device='CPU';scene.cycles.samples=20
scene.render.resolution_x=scene.render.resolution_y=256;scene.render.resolution_percentage=100
for name in ('pred_neutral','pred_expression'):
    xyz=np.load(work/(name+'.npy'));local=(xyz-matrix[:3,3])@inv.T
    for i,v in enumerate(local):h.data.vertices[i].co=v.tolist()
    scene.render.filepath=str(work/(name+'.png'));bpy.ops.render.render(write_still=True)
