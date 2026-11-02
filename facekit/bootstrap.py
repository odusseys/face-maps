"""Download immutable assets into the shared mount, never the home cache."""
import os
import io
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .paths import ASSETS, MODEL, MODEL_ID, MODEL_REVISION

BLENDER_URL = 'https://download.blender.org/release/Blender4.5/blender-4.5.3-linux-x64.tar.xz'
MPFB_URL = 'https://github.com/makehumancommunity/mpfb2/archive/refs/tags/v2.0.18.zip'
ASSET_URL = 'https://files.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip'


class RemoteZip(io.RawIOBase):
    """Seek through a ZIP over HTTP ranges so unused body assets are not fetched."""
    def __init__(self):
        self.pos = 0
        with urllib.request.urlopen(urllib.request.Request(ASSET_URL, method='HEAD'), timeout=60) as response:
            self.size = int(response.headers['Content-Length'])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos
    def read(self, size=-1):
        if size < 0: size = self.size - self.pos
        if not size: return b''
        end = min(self.size, self.pos + size) - 1
        req = urllib.request.Request(ASSET_URL, headers={'Range': f'bytes={self.pos}-{end}'})
        with urllib.request.urlopen(req, timeout=120) as response:
            if response.status != 206: raise RuntimeError('Asset server did not honor HTTP ranges')
            data = response.read()
        self.pos += len(data)
        return data


def extract_selected_assets(out):
    prefixes = [f'hair/{name}/' for name in ('short03','short01','bob02','afro01','ponytail01')]
    prefixes += [f'skins/{name}/' for name in ('young_caucasian_female','young_african_male',
                 'middleage_asian_female','old_caucasian_male','middleage_african_female')]
    prefixes += ['eyes/high-poly/', 'eyes/materials/']
    with zipfile.ZipFile(RemoteZip()) as archive:
        entries = [(x.filename,x.file_size) for x in archive.infolist() if not x.is_dir() and any(x.filename.startswith(p) for p in prefixes)]
    if not entries: raise RuntimeError('No expected MakeHuman assets found in archive')
    out.mkdir(parents=True, exist_ok=True)
    def fetch(entry):
        name,size = entry
        target = (out/name).resolve()
        if not target.is_relative_to(out.resolve()): raise ValueError('Unsafe asset archive path')
        if target.exists() and target.stat().st_size == size: return
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_name(target.name+'.partial')
        with zipfile.ZipFile(RemoteZip()) as archive, archive.open(name) as src, temp.open('wb') as dst:
            shutil.copyfileobj(src,dst)
        if temp.stat().st_size != size: raise ValueError(f'Incorrect asset size: {name}')
        temp.replace(target)
    print(f'Downloading {len(entries)} required MakeHuman asset files using HTTP ranges', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(fetch,entries))
    (out/'.selected-assets-complete').write_text('\n'.join(name for name,_ in entries))


def download(url, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.partial')
    print('Downloading', url, 'to', dest, flush=True)
    with urllib.request.urlopen(url, timeout=120) as src, part.open('wb') as out:
        shutil.copyfileobj(src, out, length=1024 * 1024)
    part.replace(dest)


def extract_zip(archive, out):
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (out / info.filename).resolve()
            if not target.is_relative_to(out.resolve()):
                raise ValueError('Unsafe archive path: ' + info.filename)
        z.extractall(out)


def assets():
    ASSETS.mkdir(parents=True, exist_ok=True)
    blender = ASSETS / 'blender-4.5.3-linux-x64' / 'blender'
    mpfb = ASSETS / 'mpfb2-2.0.18' / 'src' / 'mpfb'
    data = ASSETS / 'makehuman'
    if not blender.exists():
        archive = ASSETS / 'blender-4.5.3-linux-x64.tar.xz'
        download(BLENDER_URL, archive)
        with tarfile.open(archive) as t:
            t.extractall(ASSETS, filter='data')
    if not mpfb.exists():
        archive = ASSETS / 'mpfb2-v2.0.18.zip'
        download(MPFB_URL, archive)
        extract_zip(archive, ASSETS)
    marker = data/'.selected-assets-complete'
    if not marker.exists() or any(not (data/name).is_file() for name in marker.read_text().splitlines()):
        extract_selected_assets(data)
    candidates = list(data.rglob('short01.mhclo'))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one hair asset, got {len(candidates)}')
    # Assets are grouped under data/{hair,skins,eyes}.
    root = candidates[0].parent.parent.parent
    if not list((root / 'skins').rglob('*.mhmat')):
        raise RuntimeError(f'Cannot locate skins next to hair at {root}')
    return blender, mpfb, root


def checkpoint(vae_only=True,include_text=True):
    from huggingface_hub import snapshot_download
    MODEL.mkdir(parents=True, exist_ok=True)
    patterns = ['vae/*'] if vae_only else [
        'vae/*', 'transformer/*', 'scheduler/*', 'model_index.json', 'LICENSE.md']
    if not vae_only and include_text:
        patterns += ['text_encoder/*', 'tokenizer/*']
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(MODEL),
                      allow_patterns=patterns)
    return MODEL
