"""Recreate an existing face, apply reproducible expression units, and render it."""
import sys, os, json, importlib, time
from pathlib import Path
import numpy as np
import bpy, addon_utils
from mathutils import Vector

arg=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text())
out, source, assets, mpfb, units=map(Path,(arg['output'],arg['source'],arg['assets'],arg['mpfb'],arg['units']))
ext_dir=mpfb.parent.parent.parent/'blender_extensions'
ext_dir.mkdir(exist_ok=True)
if not (ext_dir/'mpfb').exists(): (ext_dir/'mpfb').symlink_to(mpfb,target_is_directory=True)
bpy.ops.preferences.extension_repo_add(name='FaceDataset',custom_directory=str(ext_dir),use_custom_directory=True,type='LOCAL')
repo=next(r for r in bpy.context.preferences.extensions.repos if r.directory==str(ext_dir))
module=f'bl_ext.{repo.module}.mpfb'
addon_utils.enable(module,default_set=True,persistent=False)
Human=importlib.import_module(module+'.services.humanservice').HumanService
Target=importlib.import_module(module+'.services.targetservice').TargetService
Props=importlib.import_module(module+'.entities.objectproperties').HumanObjectProperties
scene=bpy.context.scene
scene.render.engine='CYCLES'; scene.cycles.samples=20; scene.cycles.use_denoising=True
prefs=bpy.context.preferences.addons['cycles'].preferences; prefs.compute_device_type='CUDA'; prefs.get_devices()
if not any(d.type=='CUDA' for d in prefs.devices): raise RuntimeError('No CUDA device')
for d in prefs.devices: d.use=d.type=='CUDA'
scene.cycles.device='GPU'; scene.render.resolution_x=scene.render.resolution_y=256
scene.render.resolution_percentage=100; scene.render.image_settings.file_format='PNG'
scene.render.image_settings.color_mode='RGBA'; scene.render.film_transparent=True
scene.world.color=(.18,.18,.18); scene.view_settings.view_transform='AgX'

def vertices(h):
    for mod in h.modifiers: mod.show_viewport=False
    bpy.context.view_layer.update(); obj=h.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh=obj.to_mesh()
    a=np.empty(len(mesh.vertices)*3,dtype=np.float32); mesh.vertices.foreach_get('co',a); obj.to_mesh_clear()
    mat=np.asarray(h.matrix_world)
    for mod in h.modifiers: mod.show_viewport=True
    return (a.reshape(-1,3)@mat[:3,:3].T+mat[:3,3])[:13380]

def asset(rel):
    matches=list(assets.rglob(rel))
    if len(matches)!=1: raise ValueError(f'Expected one asset {rel}, found {len(matches)}')
    return matches[0]

def expression(index):
    rng=np.random.default_rng(np.random.SeedSequence([20261102,index]))
    kind=['smile','frown','surprise','pucker','smirk','squint'][index%6]
    r=lambda a,b:float(rng.uniform(a,b))
    pair=lambda name,a,b:{name+'Left':(v:=r(a,b)),name+'Right':float(v*r(.85,1.0))}
    if kind=='smile': weights={**pair('mouthSmile',.2,.7),**pair('cheekSquint',.04,.20)}
    elif kind=='frown': weights={**pair('mouthFrown',.12,.45),**pair('browDown',.05,.27)}
    elif kind=='surprise': weights={'browInnerUp':r(.1,.45),**pair('browOuterUp',.1,.4),'jawOpen':r(.03,.12)}
    elif kind=='pucker': weights={'mouthPucker':r(.18,.55),**pair('browOuterUp',.02,.15)}
    elif kind=='smirk':
        a,b=r(.25,.6),r(.03,.18)
        if rng.random()<.5: a,b=b,a
        weights={'mouthSmileLeft':a,'mouthSmileRight':b}
    else: weights={**pair('eyeSquint',.12,.4),**pair('cheekSquint',.08,.3)}
    return kind,weights

def apply_canonical_expression(h,weights,source_sample,aligned_sample):
    """Apply a common reference-space expression before the neutral alignment."""
    count=len(h.data.vertices);canonical=np.zeros((count,3),dtype=np.float64)
    for name,weight in weights.items():
        data=np.loadtxt(units/(name+'.target'));data=data[data[:,0]<count]
        np.add.at(canonical,data[:,0].astype(int),data[:,[1,3,2]]*np.array([1,-1,1])*float(weight)*.1)
    with np.load(source_sample/'geometry.npz') as a,np.load(aligned_sample/'geometry.npz') as b:
        rotation=a['rotation']@b['rotation'];scale=float(b['similarity_scale'])
    world_delta=canonical@rotation.T/scale
    matrix=np.asarray(h.matrix_world);local_delta=world_delta@np.linalg.inv(matrix[:3,:3]).T
    key=h.shape_key_add(name='canonical_expression',from_mix=False);basis=key.relative_key
    coords=np.empty(count*3,dtype=np.float32);basis.data.foreach_get('co',coords)
    coords=coords.reshape(-1,3)+local_delta.astype(np.float32)
    key.data.foreach_set('co',coords.ravel());key.value=1.0
    return canonical[:13380].astype(np.float32),rotation,scale

def add_scene(top,target_z):
    target=(0,0,target_z)
    def point(obj): obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()
    data=bpy.data.cameras.new('Camera'); camera=bpy.data.objects.new('Camera',data); scene.collection.objects.link(camera)
    camera.location=(0,-2,target_z); point(camera); scene.camera=camera
    for name,loc,power,size in [('Key',(-.45,-.8,top+.20),18,1.2),('Fill',(.5,-.7,top+.05),8,1),('Rim',(.1,.5,top+.25),10,.8)]:
        d=bpy.data.lights.new(name,'AREA');d.energy=power;d.shape='DISK';d.size=size
        ob=bpy.data.objects.new(name,d);scene.collection.objects.link(ob);ob.location=loc;point(ob)

refdata=np.load(source/'reference.npz');ref=refdata['vertices_m']; mapped=np.unique(refdata['triangle_vertices'])
for index in arg['indices']:
    folder=out/'samples'/f'{index:05d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'render_complete.json').exists(): continue
    if arg.get('capture_eyes_only') and (folder/'eye_complete.json').exists():continue
    started=time.monotonic(); src=source/'samples'/folder.name;meta=json.loads((src/'complete.json').read_text())
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    h=Human.create_human()
    for k,v in zip(['gender','age','african','asian','caucasian'],[meta['gender'],meta['age'],*meta['race_weights']]):
        Props.set_value(k,float(v),entity_reference=h)
    Target.reapply_macro_details(h)
    for path,weight in meta['targets']: Target.load_target(h,str(mpfb/'data'/'targets'/path),weight=weight)
    Human.set_character_skin(str(asset(meta['skin']+'.mhmat')),h,skin_type='MAKESKIN',material_instances=False)
    neutral=vertices(h)
    with np.load(src/'geometry.npz') as old:
        reconstructed=(neutral-old['moving_center'])@old['rotation']+old['reference_center']-ref
        err=float(np.max(np.abs(reconstructed-old['displacement_m'])))
    if err>2e-6: raise ValueError(f'Neutral identity did not reproduce: {index}, {err}')
    if 'expression_override' in arg:
        kind='counterfactual';weights=arg['expression_override']
    else:
        kind,weights=expression(index)
    canonical_error=None
    if arg.get('canonical'):
        canonical,rotation,scale=apply_canonical_expression(h,weights,src,Path(arg['neutral_source'])/'samples'/folder.name)
    else:
        for name,weight in weights.items(): Target.load_target(h,str(units/(name+'.target')),weight=weight)
    expressive=vertices(h)
    if np.max(np.abs(expressive-neutral))<1e-6 and not arg.get('neutral_only'): raise ValueError(f'No expression applied: {index}')
    if arg.get('canonical'):
        canonical_error=float(np.max(np.abs((expressive-neutral)@rotation*scale-canonical)))
        if canonical_error>1e-6:raise ValueError(f'Canonical expression roundtrip failed: {index}, {canonical_error}')
    eyes=Human.add_mhclo_asset(str(asset('high-poly.mhclo')),h,asset_type='Eyes',subdiv_levels=0,material_type='MAKESKIN')
    if arg.get('capture_eyes'):
        bpy.context.view_layer.update();evaluated=eyes.evaluated_get(bpy.context.evaluated_depsgraph_get());eye_mesh=evaluated.to_mesh()
        coords=np.empty(len(eye_mesh.vertices)*3,dtype=np.float32);eye_mesh.vertices.foreach_get('co',coords);evaluated.to_mesh_clear()
        matrix=np.asarray(eyes.matrix_world);np.save(folder/'eyes_world.npy',coords.reshape(-1,3)@matrix[:3,:3].T+matrix[:3,3])
    if arg.get('capture_eyes_only'):
        if not arg.get('capture_eyes'):raise ValueError('capture_eyes_only requires capture_eyes')
        (folder/'eye_complete.json').write_text(json.dumps({'index':index,'canonical_reproduction_error_m':canonical_error}))
        print('EYES_CAPTURED',index,flush=True);continue
    hair=Human.add_mhclo_asset(str(asset(meta['hair']+'.mhclo')),h,asset_type='Hair',subdiv_levels=0,material_type='MAKESKIN')
    camera_meta=meta['camera'];face=expressive[mapped];target_z=camera_meta['hair_target_z_m'];focal=50/36
    required=max(np.max(focal*np.abs(face[:,0])/.455-face[:,1]),np.max(focal*np.abs(face[:,2]-target_z)/.455-face[:,1]),np.max(-face[:,1])+.05)
    distance=max(camera_meta['hair_distance_m'],float(required)+.025)
    xy=np.column_stack((.5+focal*face[:,0]/(face[:,1]+distance),.5+focal*(face[:,2]-target_z)/(face[:,1]+distance)))
    if xy.min()<.04 or xy.max()>.96:raise ValueError(f'Face is not fully visible: {index}')
    add_scene(float(neutral[:,2].max()),target_z);camera=scene.camera;camera.data.type='PERSP'
    camera.data.lens=50;camera.data.sensor_fit='HORIZONTAL';camera.data.sensor_width=36
    camera.location=(0,-distance,target_z)
    scene.render.filepath=str(folder/'portrait_expression_hair.png');bpy.ops.render.render(write_still=True)
    hair.hide_render=True; target_z=camera_meta['bare_target_z_m'];camera.data.type='ORTHO';camera.data.ortho_scale=.43
    camera.location=(0,-2,target_z);camera.rotation_euler=(Vector((0,0,target_z))-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(folder/'portrait_expression_bare.png');bpy.ops.render.render(write_still=True)
    np.savez_compressed(folder/'raw_expression.npz',expression_delta_m=(expressive-neutral).astype(np.float32))
    if arg.get('canonical'):np.savez_compressed(folder/'canonical_expression.npz',expression_delta_m=canonical)
    tmp=folder/'render_complete.json.tmp';tmp.write_text(json.dumps({'index':index,'expression_kind':kind,'expression_weights':weights,
         'camera_distance_m':distance,'face_bounds':xy.tolist() and [float(xy[:,0].min()),float(xy[:,1].min()),float(xy[:,0].max()),float(xy[:,1].max())],
         'canonical_expression':bool(arg.get('canonical')),'canonical_reproduction_error_m':canonical_error,
         'neutral_reproduction_error_m':err,'render_seconds':time.monotonic()-started},indent=2));tmp.replace(folder/'render_complete.json')
    print('EXPRESSION_RENDERED',index,time.monotonic()-started,flush=True)
