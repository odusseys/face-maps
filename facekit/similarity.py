"""Scale-invariant face displacements for rigidly aligned fixed-topology meshes."""
import numpy as np


def normalize_aligned_displacement(reference, displacement, face_mask):
    """Fit one positive uniform scale after rigid alignment and recentering.

    The factor minimizes squared error to the reference over the face mask.
    It is applied to every vertex around the reference face centroid, so the
    returned coordinates remain in the reference frame and physical units.
    """
    ref=np.asarray(reference,dtype=np.float64)
    delta=np.asarray(displacement,dtype=np.float64)
    mask=np.asarray(face_mask,dtype=bool)
    if ref.ndim!=2 or ref.shape[1]!=3 or delta.shape!=ref.shape or mask.shape!=(len(ref),) or not mask.any():
        raise ValueError('Expected matching (vertices, 3) arrays and a nonempty vertex mask')
    target=ref[mask]; center=target.mean(axis=0)
    moving=ref+delta; moving_center=moving[mask].mean(axis=0)
    x=moving-moving_center; y=target-center
    denominator=np.sum(x[mask]*x[mask])
    if not np.isfinite(denominator) or denominator<=0:
        raise ValueError('Degenerate face geometry')
    scale=float(np.sum(x[mask]*y)/denominator)
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Invalid or non-positive similarity scale')
    return (center+scale*x-ref).astype(np.float32),scale
