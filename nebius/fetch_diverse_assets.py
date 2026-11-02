#!/usr/bin/env python3
"""Fetch additional realistic MakeHuman skins and hair directly on Nebius."""
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from facekit.paths import ASSETS
from facekit.bootstrap import RemoteZip


def main():
    root=ASSETS/'makehuman';root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(RemoteZip()) as archive:
        entries=[]
        for info in archive.infolist():
            p=Path(info.filename)
            if info.is_dir() or len(p.parts)<3 or p.parts[0] not in ('skins','hair'):continue
            if p.parts[0]=='skins' and any(word in p.parts[1] for word in ('toon','special_suit')):continue
            target=(root/p).resolve()
            if not target.is_relative_to(root.resolve()):raise ValueError('Unsafe archive path')
            entries.append((info.filename,info.file_size))
    def fetch(entry):
        name,size=entry;target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.stat().st_size==size:return 'cached'
        temp=target.with_name(target.name+'.diverse.part')
        with zipfile.ZipFile(RemoteZip()) as archive,archive.open(name) as source,temp.open('wb') as out:
            while True:
                chunk=source.read(1024*1024)
                if not chunk:break
                out.write(chunk)
        if temp.stat().st_size!=size:raise ValueError(f'Incorrect size for {name}')
        temp.replace(target);return 'downloaded'
    with ThreadPoolExecutor(max_workers=8) as pool:states=list(pool.map(fetch,entries))
    skins=sorted(p.stem for p in (root/'skins').glob('*/*.mhmat') if 'special_suit' not in str(p) and 'toon' not in str(p))
    hairs=sorted(p.stem for p in (root/'hair').glob('*/*.mhclo'))
    if len(skins)<20 or len(hairs)<10:raise ValueError('Missing expected assets')
    result={'skins':skins,'hairs':hairs,'files':len(entries),'downloaded':states.count('downloaded')}
    path=ASSETS/'diverse-face-assets.json';path.write_text(json.dumps(result,indent=2))
    print('DIVERSE_ASSETS_READY',json.dumps(result),flush=True)


if __name__=='__main__':main()
