"""Render the reference synthetic human as a white silhouette on black."""
import sys
from pathlib import Path
import bpy

output=Path(sys.argv[sys.argv.index('--')+1])
scene=bpy.context.scene
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=1
scene.render.resolution_x=scene.render.resolution_y=512;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
scene.render.film_transparent=True
white=bpy.data.materials.new('Silhouette white');white.use_nodes=True
nodes=white.node_tree.nodes;nodes.clear();out=nodes.new('ShaderNodeOutputMaterial');shader=nodes.new('ShaderNodeEmission')
shader.inputs['Color'].default_value=(1,1,1,1);shader.inputs['Strength'].default_value=1
white.node_tree.links.new(shader.outputs[0],out.inputs['Surface'])
scene.view_layers[0].material_override=white
scene.view_settings.view_transform='Standard';scene.render.filepath=str(output)
bpy.ops.render.render(write_still=True)
