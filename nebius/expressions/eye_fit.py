"""Fit reference eye meshes to a predicted neutral face without GT leakage."""
import numpy as np

def fit_eyes(reference_vertices,neutral_vertices,reference_eyes,face_ids):
    predicted=np.asarray(reference_eyes,dtype=np.float64).copy();points=reference_vertices[face_ids]
    for side in (-1,1):
        eye_ids=reference_eyes[:,0]*side>0;center=reference_eyes[eye_ids].mean(axis=0)
        mask=((points[:,0]-center[0])**2+(points[:,2]-center[2])**2<.012**2)&(points[:,1]<-.1)&(points[:,0]*side>0)
        roi=face_ids[mask]
        if len(roi)<4:raise ValueError('Insufficient eye socket vertices')
        x=reference_vertices[roi];y=neutral_vertices[roi];xm=x.mean(0);ym=y.mean(0)
        u,_,vt=np.linalg.svd((x-xm).T@(y-ym));rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
        scale=float(np.clip(np.sum(((x-xm)@rotation)*(y-ym))/np.sum((x-xm)**2),.75,1.25))
        predicted[eye_ids]=(reference_eyes[eye_ids]-xm)@rotation*scale+ym
    # A small pull toward the reference limits over-correction on unusual faces.
    return (reference_eyes+.85*(predicted-reference_eyes)).astype(np.float32)
