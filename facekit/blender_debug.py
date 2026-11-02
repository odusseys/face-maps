"""Invoked by Blender for an inference debug portrait."""
import sys
from pathlib import Path
import bpy
import numpy as np

work=Path(sys.argv[sys.argv.index('--')+1]); xyz=np.load(work/'vertices.npy')
h=bpy.data.objects['Human']; matrix=np.asarray(h.matrix_world); inv=np.linalg.inv(matrix[:3,:3])
local=(xyz-matrix[:3,3])@inv.T
# MakeHuman uses active shape keys.  They override mesh vertex coordinates on
# evaluation, so discard them before applying the already-evaluated target mesh.
if h.data.shape_keys:
    h.shape_key_clear()
for i,v in enumerate(local): h.data.vertices[i].co=v.tolist()
replaced=0
for material in h.data.materials:
    if material and material.use_nodes:
        for node in material.node_tree.nodes:
            if node.type=='TEX_IMAGE' and node.image and 'diffuse' in node.image.name.lower():
                node.image=bpy.data.images.load(str(work/'atlas.png'),check_existing=False); replaced+=1
if not replaced: raise RuntimeError('Reference skin diffuse texture node not found')
scene=bpy.context.scene; scene.cycles.device='CPU'; scene.cycles.samples=20
scene.render.filepath=str(work/'predicted_face.png'); bpy.ops.render.render(write_still=True)
