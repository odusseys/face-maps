"""Render GT and predicted neutral/expressive meshes with identical view and light."""
import sys
from pathlib import Path
import bpy,numpy as np

work=Path(sys.argv[sys.argv.index('--')+1]); h=bpy.data.objects['Human'];matrix=np.asarray(h.matrix_world);inv=np.linalg.inv(matrix[:3,:3])
if h.data.shape_keys:h.shape_key_clear()
nodes=[node for mat in h.data.materials if mat and mat.use_nodes for node in mat.node_tree.nodes
       if node.type=='TEX_IMAGE' and node.image and 'diffuse' in node.image.name.lower()]
if not nodes:raise ValueError('No diffuse texture node')
scene=bpy.context.scene;scene.cycles.device='CPU';scene.cycles.samples=20
scene.render.resolution_x=scene.render.resolution_y=256;scene.render.resolution_percentage=100
for label in ('gt_neutral','pred_neutral','gt_expression','pred_expression'):
    xyz=np.load(work/(label+'.npy')); local=(xyz-matrix[:3,3])@inv.T
    for i,v in enumerate(local):h.data.vertices[i].co=v.tolist()
    atlas='gt_atlas.png' if label.startswith('gt_') else 'pred_atlas.png'
    image=bpy.data.images.load(str(work/atlas),check_existing=True)
    for node in nodes:node.image=image
    scene.render.filepath=str(work/(label+'.png'));bpy.ops.render.render(write_still=True)
