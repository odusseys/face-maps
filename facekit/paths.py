"""Shared-storage defaults for MakeHuman/FLUX face experiments."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('FACE_SHARED_ROOT', str(Path.home() / 'datasets' / 'faces')))
DATA = ROOT / 'face_maps'
ASSETS = DATA / 'assets'
MODEL = ROOT / 'models' / 'FLUX.2-klein-base-4B'
MODEL_ID = 'black-forest-labs/FLUX.2-klein-base-4B'
MODEL_REVISION = 'a3b4f4849157f664bdbc776fd7453c2783562f4d'
BOUND_MM = 14.008914828504082
