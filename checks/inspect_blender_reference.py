"""Inspect evaluated reference vertices and Blender shape-key write behavior."""
import json
import sys
from pathlib import Path
import bpy
import numpy as np

args=sys.argv[sys.argv.index('--')+1:]
reference=np.load(args[0]) if args else None
h=bpy.data.objects['Human']
key_count=len(h.data.shape_keys.key_blocks) if h.data.shape_keys else 0
for mod in h.modifiers: mod.show_viewport=False
bpy.context.view_layer.update()


def evaluated():
    obj=h.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh=obj.to_mesh()
    data=np.empty(len(mesh.vertices)*3,np.float32);mesh.vertices.foreach_get('co',data)
    obj.to_mesh_clear();return data.reshape(-1,3)

before=evaluated(); matrix=np.asarray(h.matrix_world,dtype=np.float64)
world=before @ matrix[:3,:3].T+matrix[:3,3]
count=min(13380,len(before)); target=before.copy();target[:count,0]+=.02
for i in range(count): h.data.vertices[i].co=target[i]
h.data.update();bpy.context.view_layer.update()
after_data_write=evaluated()
while h.data.shape_keys and h.data.shape_keys.key_blocks:
    h.shape_key_remove(h.data.shape_keys.key_blocks[-1])
for i in range(count): h.data.vertices[i].co=target[i]
h.data.update();bpy.context.view_layer.update()
after_baked_write=evaluated()
images=[]
for mat in h.data.materials:
    if mat and mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type=='TEX_IMAGE' and node.image:
                images.append({'name':node.image.name,'colorspace':node.image.colorspace_settings.name})
result={'num_vertices':len(before),'shape_key_count':key_count,'ref_match_max_m':float(np.max(np.abs(world[:count]-reference['vertices_m']))) if reference is not None else None,
        'data_write_mean_shift_m':float(np.mean(after_data_write[:count,0]-before[:count,0])),
        'data_write_max_target_error_m':float(np.max(np.abs(after_data_write[:count]-target[:count]))),
        'baked_write_mean_shift_m':float(np.mean(after_baked_write[:count,0]-before[:count,0])),
        'baked_write_max_target_error_m':float(np.max(np.abs(after_baked_write[:count]-target[:count]))),
        'images':images}
print('BLENDER_AUDIT',json.dumps(result),flush=True)
