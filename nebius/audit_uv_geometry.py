#!/usr/bin/env python3
"""Independently recompute UV-only fits from the pre-alignment source transforms."""
import json
import numpy as np
from facekit.paths import DATA


def fit(reference, raw, ids):
    x0, y0 = raw[ids].mean(0), reference[ids].mean(0)
    x, y = raw[ids]-x0, reference[ids]-y0
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3); correction[-1,-1] = np.linalg.det(u @ vt)
    r = u @ correction @ vt
    scale = np.sum((x @ r)*y) / np.sum(x*x)
    return ((raw-x0) @ r * scale + y0 - reference)[ids]


def summary(v):
    return {k: float(x) for k, x in zip(('median','p95','max'),np.quantile(v,(.5,.95,1)))}


def main():
    datasets = DATA/'datasets'; source = datasets/'faces500-centered'
    old = datasets/'faces500-similarity-linear-q97'; new = datasets/'faces500-uv-aligned-uvq97'
    ref = np.load(source/'reference.npz'); v = ref['vertices_m']; ids = np.unique(ref['triangle_vertices'])
    agreement, changes, outside, invariant = [], [], [], []
    for i,p in enumerate(sorted((new/'samples').glob('*/geometry.npz'))):
        with np.load(source/'samples'/p.parent.name/'geometry.npz') as g:
            aligned = v + g['displacement_m']
            raw = (aligned-g['reference_center']) @ g['rotation'].T + g['moving_center']
        expected = fit(v, raw, ids)
        saved = np.load(p)['displacement_m'][ids]
        old_delta = np.load(old/'samples'/p.parent.name/'geometry.npz')['displacement_m'][ids]
        agreement.append(float(np.max(np.abs(expected-saved))*1000))
        changes.append(float(np.linalg.norm((saved-old_delta)*1000,axis=1).mean()))
        if i<10:
            select = np.ones(len(v),dtype=bool);select[ids] = False
            altered = raw.copy();altered[select] += [.7,-.5,.3]
            outside.append(float(np.max(np.abs(fit(v,altered,ids)-expected))*1000))
            altered = raw*1.7+np.array([.2,-.3,.1])
            invariant.append(float(np.max(np.abs(fit(v,altered,ids)-expected))*1000))
    if len(agreement)!=500 or max(agreement)>.001 or max(outside)>1e-8 or max(invariant)>1e-8:
        raise ValueError('UV-aligned geometry verification failed')
    result={'samples':len(agreement),'alignment_vertices':len(ids),
            'saved_vs_independent_fit_max_component_mm':summary(agreement),
            'change_vs_old_mean_vertex_mm':summary(changes),
            'outside_uv_perturbation_max_component_mm':max(outside),
            'scale_translation_invariance_max_component_mm':max(invariant)}
    path=DATA/'audits/nebius-uv-geometry-validation.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
