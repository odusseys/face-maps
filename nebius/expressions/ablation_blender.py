"""Render controlled mesh/texture combinations in the same scene."""
import json,sys
from pathlib import Path
import bpy,numpy as np

work=Path(sys.argv[sys.argv.index('--')+1]);items=json.loads((work/'config.json').read_text())
h=bpy.data.objects['Human'];matrix=np.asarray(h.matrix_world);inv=np.linalg.inv(matrix[:3,:3])
if h.data.shape_keys:h.shape_key_clear()
eye=bpy.data.objects['Human.high-poly'];eye_matrix=np.asarray(eye.matrix_world);eye_inv=np.linalg.inv(eye_matrix[:3,:3]);default_eyes=np.empty(len(eye.data.vertices)*3,np.float32)
eye.data.vertices.foreach_get('co',default_eyes);default_eyes=default_eyes.reshape(-1,3)@eye_matrix[:3,:3].T+eye_matrix[:3,3]
nodes=[n for m in h.data.materials if m and m.use_nodes for n in m.node_tree.nodes if n.type=='TEX_IMAGE' and n.image and 'diffuse' in n.image.name.lower()]
if not nodes:raise ValueError('Missing diffuse nodes')
scene=bpy.context.scene;scene.cycles.device='CPU';scene.cycles.samples=20;scene.render.resolution_x=scene.render.resolution_y=256;scene.render.resolution_percentage=100
for item in items:
    xyz=np.load(item['mesh']);local=(xyz-matrix[:3,3])@inv.T
    for i,v in enumerate(local):h.data.vertices[i].co=v.tolist()
    eye_xyz=np.load(item['eyes']) if item.get('eyes') else default_eyes
    eye_local=(eye_xyz-eye_matrix[:3,3])@eye_inv.T
    for i,v in enumerate(eye_local):eye.data.vertices[i].co=v.tolist()
    img=bpy.data.images.load(item['texture'],check_existing=True)
    for node in nodes:node.image=img
    scene.render.filepath=str(work/(item['name']+'.png'));bpy.ops.render.render(write_still=True)
