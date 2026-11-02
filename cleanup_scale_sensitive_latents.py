#!/usr/bin/env python3
"""Remove obsolete face target latents after verified shape-only replacements."""
import json
from pathlib import Path
import torch
from facekit.paths import DATA

DATASETS=('faces500','faces500-centered','faces500-centered-linear-q97',
          'smoke-3','linear-q97-smoke','reframe-parallel-check')


def atomic_json(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def verify_replacement():
    ds=DATA/'datasets'/'faces500-similarity-linear-q97'
    state=json.loads((ds/'dataset_complete.json').read_text())
    manifest=json.loads((ds/'latents_displacement'/'manifest.json').read_text())
    if not state.get('displacement_latents_ready') or not state.get('paired_map_latents_ready'):
        raise RuntimeError('Replacement dataset is not complete')
    if manifest.get('count')!=500 or not manifest.get('portrait_latents_verified_unchanged') or len(manifest.get('roundtrip_verification',[]))!=6:
        raise RuntimeError('Replacement dataset was not fully verified')
    records=sorted((ds/'samples').glob('*/complete.json'))
    if len(records)!=500 or any(not (ds/'latents_displacement'/(p.parent.name+'.pt')).is_file() for p in records):
        raise RuntimeError('Replacement target latent files are missing')
    return ds


def main():
    replacement=verify_replacement();rows=[]
    for name in DATASETS:
        ds=DATA/'datasets'/name;config_path=ds/'dataset.json'
        if not config_path.is_file():continue
        config=json.loads(config_path.read_text())
        if config.get('geometry_alignment') or 'rigidly aligned' not in config.get('displacement_axes',''):
            print('SKIP_UNRECOGNIZED',name,flush=True);continue
        count=0;bytes_before=0;bytes_after=0
        for path in sorted((ds/'samples').glob('*/latents.pt')):
            data=torch.load(path,map_location='cpu',weights_only=True)
            if 'maps' not in data:continue
            if not {'face_bare','face_hair'}.issubset(data):
                raise ValueError(f'Cannot safely preserve portrait latents: {path}')
            bytes_before+=path.stat().st_size;del data['maps']
            temp=path.with_suffix('.pt.tmp');torch.save(data,temp);temp.replace(path)
            bytes_after+=path.stat().st_size;count+=1
            meta_path=path.parent/'complete.json'
            if meta_path.exists():
                meta=json.loads(meta_path.read_text());meta['paired_map_latent_available']=False;atomic_json(meta_path,meta)
        removed=0
        target_dir=ds/'latents_displacement'
        for path in target_dir.glob('*.pt'):
            bytes_before+=path.stat().st_size;path.unlink();removed+=1
        manifest=target_dir/'manifest.json'
        if manifest.is_file():manifest.unlink()
        config.update({'latent_modes':['portrait_only'],'scale_sensitive_target_latents_removed':True,
                       'shape_only_replacement':str(replacement)})
        atomic_json(config_path,config)
        complete=ds/'dataset_complete.json'
        if complete.is_file():
            state=json.loads(complete.read_text());state.update({'displacement_latents_ready':False,'paired_map_latents_ready':False})
            atomic_json(complete,state)
        row={'dataset':name,'paired_latents_removed':count,'displacement_latents_removed':removed,
             'bytes_reclaimed':bytes_before-bytes_after}
        rows.append(row);print('CLEANED',json.dumps(row),flush=True)
    smoke=DATA/'audits'/'displacement-latents-smoke';smoke_count=0;smoke_bytes=0
    for path in smoke.glob('*.pt'):
        smoke_bytes+=path.stat().st_size;path.unlink();smoke_count+=1
    manifest=smoke/'manifest.json'
    if manifest.is_file():manifest.unlink()
    rows.append({'dataset':'audit displacement smoke','paired_latents_removed':0,
                 'displacement_latents_removed':smoke_count,'bytes_reclaimed':smoke_bytes})
    result={'replacement':str(replacement),'datasets':rows}
    atomic_json(DATA/'audits'/'scale-sensitive-latent-cleanup-20261101.json',result)
    print('CLEANUP_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
