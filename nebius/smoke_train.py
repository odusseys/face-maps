"""Exercise the unmodified trainer with a deliberately small smoke dataset."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_face_maps


def smoke_samples(dataset):
    records = []
    for marker in sorted((Path(dataset) / 'samples').glob('*/complete.json')):
        records.append((marker.parent, json.loads(marker.read_text())['split']))
    train = [path for path, split in records if split == 'train']
    holdout = [path for path, split in records if split == 'holdout']
    if len(train) < 1 or len(holdout) < 3:
        raise ValueError('Smoke dataset requires at least one training and three holdout faces')
    print(f'SMOKE_ONLY_SPLIT train={len(train)} holdout={len(holdout)}', flush=True)
    return train, holdout


if __name__ == '__main__':
    train_face_maps.list_samples = smoke_samples
    train_face_maps.main()
