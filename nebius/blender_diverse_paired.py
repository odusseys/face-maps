"""Preview 20 faces with broader, independently sampled shape parameters."""
import sys, os, json, importlib, time
from pathlib import Path
import numpy as np
import bpy, addon_utils
from mathutils import Vector

arg = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text())
out, assets, mpfb = map(Path, (arg['output'], arg['assets'], arg['mpfb']))
ext_dir = mpfb.parent.parent.parent / 'blender_extensions'
ext_dir.mkdir(exist_ok=True)
link = ext_dir / 'mpfb'
if not link.exists():
    link.symlink_to(mpfb, target_is_directory=True)
bpy.ops.preferences.extension_repo_add(name='FaceDataset', custom_directory=str(ext_dir), use_custom_directory=True, type='LOCAL')
repo = next(r for r in bpy.context.preferences.extensions.repos if r.directory == str(ext_dir))
module = f'bl_ext.{repo.module}.mpfb'
addon_utils.enable(module, default_set=True, persistent=False)
Human = importlib.import_module(module+'.services.humanservice').HumanService
Target = importlib.import_module(module+'.services.targetservice').TargetService
Props = importlib.import_module(module+'.entities.objectproperties').HumanObjectProperties
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = arg['samples']
scene.cycles.use_denoising = True
if arg['device'] == 'CUDA':
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    devices = [d for d in prefs.devices if d.type == 'CUDA']
    if not devices:
        raise RuntimeError('CUDA rendering requested but Blender found no CUDA device')
    for d in prefs.devices: d.use = d.type == 'CUDA'
    scene.cycles.device = 'GPU'
    print('CYCLES_CUDA', [(d.name, d.use) for d in devices], flush=True)
else:
    scene.cycles.device = 'CPU'
scene.render.resolution_x = scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.film_transparent = True
scene.world.color = (.18,.18,.18)
scene.view_settings.view_transform = 'AgX'


def clear():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def vertices(h):
    for mod in h.modifiers: mod.show_viewport = False
    bpy.context.view_layer.update()
    obj = h.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = obj.to_mesh()
    a = np.empty(len(mesh.vertices)*3, dtype=np.float32)
    mesh.vertices.foreach_get('co', a)
    obj.to_mesh_clear()
    mat = np.asarray(h.matrix_world)
    for mod in h.modifiers: mod.show_viewport = True
    return (a.reshape(-1,3) @ mat[:3,:3].T + mat[:3,3])[:13380]


def add_scene(top, target_z=None, ortho_scale=.43):
    target = (0,0,top-.13 if target_z is None else target_z)
    def point(obj): obj.rotation_euler = (Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()
    data = bpy.data.cameras.new('Camera'); camera = bpy.data.objects.new('Camera', data)
    scene.collection.objects.link(camera); camera.location = (0,-2,target[2]); point(camera)
    data.type='ORTHO'; data.ortho_scale=ortho_scale; scene.camera=camera
    for name,loc,power,size in [('Key',(-.45,-.8,top+.20),18,1.2),('Fill',(.5,-.7,top+.05),8,1),('Rim',(.1,.5,top+.25),10,.8)]:
        d=bpy.data.lights.new(name,'AREA'); d.energy=power; d.shape='DISK'; d.size=size
        obj=bpy.data.objects.new(name,d); scene.collection.objects.link(obj); obj.location=loc; point(obj)


def asset(rel):
    matches = list(assets.rglob(rel))
    if len(matches) != 1: raise RuntimeError(f'Expected one asset {rel}, found {len(matches)}')
    return matches[0]


def prepare_reference():
    clear(); h=Human.create_human(); ref=vertices(h)
    assert ref.shape == (13380,3)
    top=float(ref[:,2].max()); face=(ref[:,2]>top-.235)&(ref[:,1]<-.085)
    mesh=h.data; mesh.calc_loop_triangles()
    tris=[]; uvs=[]
    for t in mesh.loop_triangles:
        ids=np.array(t.vertices)
        if np.all(ids < len(ref)) and np.all(face[ids]):
            tuv=np.array([mesh.uv_layers.active.data[i].uv[:] for i in t.loops])
            if tuv[:,0].mean()>.60 and .20<tuv[:,1].mean()<.80:
                tris.append(ids); uvs.append(tuv)
    np.savez_compressed(out/'reference.npz', vertices_m=ref, face_mask=face,
                        triangle_vertices=np.asarray(tris), triangle_uv=np.asarray(uvs),
                        matrix_world=np.asarray(h.matrix_world))
    skin=asset('young_caucasian_female.mhmat')
    diffuse=next(line.split(None,1)[1].strip() for line in skin.read_text().splitlines() if line.lower().startswith('diffusetexture '))
    (out/'reference_skin.txt').write_text(str(skin.parent/diffuse))
    Human.set_character_skin(str(skin),h,skin_type='MAKESKIN',material_instances=False)
    Human.add_mhclo_asset(str(asset('high-poly.mhclo')),h,asset_type='Eyes',subdiv_levels=0,material_type='MAKESKIN')
    add_scene(top); bpy.ops.wm.save_as_mainfile(filepath=str(out/'reference.blend'), compress=True)
    return ref,face,top


if arg.get('reference'):
    prepare_reference(); sys.exit(0)
z=np.load(out/'reference.npz'); ref=z['vertices_m']; mask=z['face_mask']; top=float(ref[:,2].max())
catalog=json.loads((assets.parent/'diverse-face-assets.json').read_text())
skins=catalog['skins']; hairs=[h for h in catalog['hairs'] if h not in ('bob01','bob02','short03','braid01')]
shapes=['oval','round','invertedtriangular','square','diamond']
for index in arg['indices']:
    folder=out/'samples'/f'{index:05d}'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'render_complete.json').exists(): continue
    start=time.monotonic(); rng=np.random.default_rng(np.random.SeedSequence([arg['seed'],index]))
    clear(); h=Human.create_human()
    race=rng.dirichlet([.7]*3); gender=float(rng.uniform(.02,.98)); age=float(rng.uniform(.28,.72))
    for k,v in zip(['gender','age','african','asian','caucasian'],[gender,age,*race]):
        Props.set_value(k,float(v),entity_reference=h)
    Target.reapply_macro_details(h)
    shape=str(rng.choice(shapes)); direction=lambda: str(rng.choice(['incr','decr']))
    targets=[(f'head/head-{shape}.target.gz',float(rng.uniform(.25,.58))),
        (f'head/head-scale-horiz-{direction()}.target.gz',float(rng.uniform(.10,.38))),
        (f'head/head-scale-vert-{direction()}.target.gz',float(rng.uniform(.05,.22))),
        (f'head/head-fat-{direction()}.target.gz',float(rng.uniform(.10,.48))),
        (f'nose/nose-scale-horiz-{direction()}.target.gz',float(rng.uniform(.12,.65))),
        (f'nose/nose-scale-vert-{direction()}.target.gz',float(rng.uniform(.10,.50))),
        (f'chin/chin-width-{direction()}.target.gz',float(rng.uniform(.12,.65))),
        (f'chin/chin-height-{direction()}.target.gz',float(rng.uniform(.10,.50))),
        (f'mouth/mouth-scale-horiz-{direction()}.target.gz',float(rng.uniform(.10,.55)))]
    for p,w in targets: Target.load_target(h,str(mpfb/'data'/'targets'/p),weight=w)
    skin=str(rng.choice(skins)); hair_name=str(rng.choice(hairs))
    mhmat=asset(skin+'.mhmat'); Human.set_character_skin(str(mhmat),h,skin_type='MAKESKIN',material_instances=False)
    Human.add_mhclo_asset(str(asset('high-poly.mhclo')),h,asset_type='Eyes',subdiv_levels=0,material_type='MAKESKIN')
    xyz=vertices(h); moving=xyz[mask]; fixed=ref[mask]; mc=moving.mean(0); fc=fixed.mean(0)
    u,_,vt=np.linalg.svd((moving-mc).T@(fixed-fc)); R=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
    aligned=(xyz-mc)@R+fc; delta=aligned-ref
    hair=Human.add_mhclo_asset(str(asset(hair_name+'.mhclo')),h,asset_type='Hair',subdiv_levels=0,material_type='MAKESKIN')
    bpy.context.view_layer.update()
    evaluated=hair.evaluated_get(bpy.context.evaluated_depsgraph_get()); hair_mesh=evaluated.to_mesh()
    try:
        hair_coords=np.empty(len(hair_mesh.vertices)*3,dtype=np.float32)
        hair_mesh.vertices.foreach_get('co',hair_coords)
        matrix=np.asarray(hair.matrix_world)
        hair_coords=hair_coords.reshape(-1,3)@matrix[:3,:3].T+matrix[:3,3]
    finally:
        evaluated.to_mesh_clear()
    sample_top=float(xyz[:,2].max())
    hair_top=float(hair_coords[:,2].max()); hair_width=float(np.ptp(hair_coords[:,0]))
    # A perspective hair portrait varies actual camera distance, while a
    # conservative fit check guarantees the full UV-mapped face is visible.
    mapped=np.unique(z['triangle_vertices']); face=xyz[mapped]
    target_z=float((face[:,2].min()+face[:,2].max())/2)
    focal_ratio=50.0/36.0; allowed=.455
    required=float(max(np.max(focal_ratio*np.abs(face[:,0])/allowed-face[:,1]),
                       np.max(focal_ratio*np.abs(face[:,2]-target_z)/allowed-face[:,1]),
                       np.max(-face[:,1])+.05))
    minimum=max(.48,required+.025); maximum=max(.95,minimum+.15)
    distance=float(rng.uniform(minimum,maximum))
    add_scene(sample_top,target_z,.43)
    camera=scene.camera; camera.data.type='PERSP'; camera.data.lens=50.0
    camera.data.sensor_fit='HORIZONTAL';camera.data.sensor_width=36.0
    camera.location=(0,-distance,target_z)
    camera.rotation_euler=(Vector((0,0,target_z))-camera.location).to_track_quat('-Z','Y').to_euler()
    bpy.context.view_layer.update()
    from bpy_extras.object_utils import world_to_camera_view
    depth=face[:,1]+distance
    xy=np.column_stack((.5+focal_ratio*face[:,0]/depth,.5+focal_ratio*(face[:,2]-target_z)/depth))
    if np.any(depth<=0) or xy.min()<.04 or xy.max()>.96:
        raise RuntimeError(f'Face outside requested framing: {index}')
    for j in {int(np.argmin(xy[:,0])),int(np.argmax(xy[:,0])),int(np.argmin(xy[:,1])),int(np.argmax(xy[:,1]))}:
        coord=world_to_camera_view(scene,camera,Vector(face[j]))
        if not (.035<=coord.x<=.965 and .035<=coord.y<=.965 and coord.z>0):
            raise RuntimeError(f'Blender camera visibility mismatch: {index}')
    scene.render.filepath=str(folder/'portrait_hair.png'); bpy.ops.render.render(write_still=True)
    hair.hide_render=True
    standard_target_z=sample_top-.13
    camera.data.type='ORTHO';camera.data.ortho_scale=.43
    camera.location=(0,-2,standard_target_z)
    camera.rotation_euler=(Vector((0,0,standard_target_z))-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(folder/'portrait_bare.png'); bpy.ops.render.render(write_still=True)
    diffuse=next(line.split(None,1)[1].strip() for line in mhmat.read_text().splitlines() if line.lower().startswith('diffusetexture '))
    tex=mhmat.parent/diffuse
    if not tex.is_file(): raise FileNotFoundError(tex)
    np.savez_compressed(folder/'geometry.npz', displacement_m=delta.astype(np.float32), rotation=R.astype(np.float32), moving_center=mc, reference_center=fc)
    meta={'index':index,'seed':arg['seed'],'gender':gender,'age':age,'race_weights':race.tolist(),
          'targets':targets,'skin':skin,'hair':hair_name,'texture':str(tex),
          'camera':{'hair_projection':'perspective','hair_distance_m':distance,'hair_target_z_m':target_z,
                    'hair_face_bounds':[float(xy[:,0].min()),float(xy[:,1].min()),float(xy[:,0].max()),float(xy[:,1].max())],
                    'bare_projection':'orthographic','bare_ortho_scale_m':.43,'bare_target_z_m':standard_target_z,
                    'head_top_m':sample_top,'hair_top_m':hair_top},
          'max_abs_coordinate_mm':float(np.max(np.abs(delta[mask]))*1000),'render_seconds':time.monotonic()-start}
    (folder/'render_complete.json').write_text(json.dumps(meta,indent=2))
    print('RENDERED',index,meta['render_seconds'],flush=True)
