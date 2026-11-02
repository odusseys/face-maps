"""Independent numerical checks for the saved face-map dataset; CPU only."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

p = argparse.ArgumentParser()
p.add_argument('dataset', type=Path)
p.add_argument('code', type=Path)
a = p.parse_args()
sys.path.insert(0, str(a.code))
from facekit.uv import ATLAS, panel, inverse_panel, encode_displacement, decode_displacement

ref = np.load(a.dataset/'reference.npz')
mapping = np.load(a.dataset/'uv_mapping.npz')
meta = json.loads((a.dataset/'dataset.json').read_text())
bound = float(meta['bound_mm'])
mask, bbox = mapping['mask'], mapping['bbox']
indices, weights = mapping['indices'], mapping['weights']
vertices, tris, uvs = ref['vertices_m'], ref['triangle_vertices'], ref['triangle_uv']
nearest = distance_transform_edt(~mask, return_distances=False, return_indices=True)
count = np.zeros(len(vertices), np.float32)
for tri in tris: np.add.at(count, tri, 1)
good = count > 0

white = Image.fromarray(np.full((ATLAS,ATLAS,3),255,np.uint8))
alpha = np.asarray(inverse_panel(panel(white,mask,bbox),bbox), dtype=np.float32)/255

def sample(img, corrected=False):
    atlas = np.asarray(inverse_panel(img,bbox),dtype=np.float32)
    if corrected:
        atlas = np.clip(atlas/np.maximum(alpha,1/255),0,255)
    field = decode_displacement(atlas,bound,'signed_linear')
    result = np.zeros_like(vertices)
    for tri, uv in zip(tris, uvs):
        x = np.clip(np.rint(uv[:,0]*ATLAS-.5).astype(int), 0, ATLAS-1)
        y = np.clip(np.rint((1-uv[:,1])*ATLAS-.5).astype(int), 0, ATLAS-1)
        np.add.at(result, tri, field[nearest[0,y,x], nearest[1,y,x]])
    result[good] /= count[good,None]
    return result

def errors(actual, desired):
    e = np.abs((actual[good]-desired[good])*1000).ravel()
    return {'mean_mm':float(e.mean()), 'p95_mm':float(np.quantile(e,.95)), 'p99_mm':float(np.quantile(e,.99)), 'max_mm':float(e.max()), 'over_0_5mm_fraction':float(np.mean(e>.5)), 'over_1mm_fraction':float(np.mean(e>1))}

# Verify independently that each stored barycentric assignment reaches its pixel center.
lookup = {tuple(t):uv for t,uv in zip(tris,uvs)}
yy,xx = np.where(mask)
rng = np.random.default_rng(514)
picks = rng.choice(len(xx), min(4000,len(xx)), replace=False)
err = []
for j in picks:
    y,x = yy[j],xx[j]
    uv = lookup[tuple(indices[y,x])]
    xyz = weights[y,x]@uv
    err.append(np.hypot(xyz[0]*ATLAS-.5-x,(1-xyz[1])*ATLAS-.5-y))

zero = np.full((ATLAS,ATLAS,3), 128, np.uint8)
zero_img = panel(Image.fromarray(zero),mask,bbox)
zero_est = sample(zero_img)
zeros = np.zeros_like(vertices)

samples = sorted((a.dataset/'samples').glob('*/geometry.npz'))
all_abs = []
results = []
for i, path in enumerate(samples):
    geom = np.load(path)
    delta = geom['displacement_m']
    all_abs.append(np.abs(delta[ref['face_mask']]).ravel())
    if i in {0,1,2,10,50,100,198,250,350,453,474,499}:
        est = sample(Image.open(path.parent/'displacement.png'))
        clipped = np.clip(delta,-bound/1000,bound/1000)
        results.append({'id':path.parent.name, 'component_clipped_fraction':float(np.mean(np.abs(delta[good])>bound/1000)), 'error_vs_original':errors(est,delta), 'error_vs_clipped':errors(est,clipped), 'corrected_error_vs_clipped':errors(sample(Image.open(path.parent/'displacement.png'),True),clipped), 'rotation_det':float(np.linalg.det(geom['rotation'])), 'rotation_orthogonal_max_error':float(np.max(np.abs(geom['rotation'].T@geom['rotation']-np.eye(3))))})
all_abs = np.concatenate(all_abs)*1000
print(json.dumps({'samples':len(samples), 'bound_mm':bound, 'independent_q97_mm':float(np.quantile(all_abs,.97)), 'all_components_above_bound':float(np.mean(all_abs>bound)), 'mask_pixels':int(mask.sum()), 'weights_sum_max_error':float(np.max(np.abs(weights[mask].sum(axis=1)-1))), 'barycentric_pixel_error_max':float(max(err)), 'face_vertices':int(ref['face_mask'].sum()), 'reconstructed_vertices':int(good.sum()), 'face_vertices_reconstructed':int(np.sum(good & ref['face_mask'])), 'zero_vector_roundtrip':errors(zero_est,zeros), 'zero_corrected_roundtrip':errors(sample(zero_img,True),zeros), 'samples_audit':results},indent=2))
