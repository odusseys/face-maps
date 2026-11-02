"""Extract the reference eye mesh for a controlled fitting experiment."""
import sys
from pathlib import Path
import bpy,numpy as np

out=Path(sys.argv[sys.argv.index('--')+1]);obj=bpy.data.objects['Human.high-poly']
mesh=obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh();a=np.empty(len(mesh.vertices)*3,np.float32);mesh.vertices.foreach_get('co',a)
matrix=np.asarray(obj.matrix_world);np.save(out/'reference_eyes.npy',a.reshape(-1,3)@matrix[:3,:3].T+matrix[:3,3])
print('REFERENCE_EYE_VERTICES',len(mesh.vertices),flush=True)
