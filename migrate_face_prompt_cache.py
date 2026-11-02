#!/usr/bin/env python3
"""Upgrade validated prompt caches without recomputing encoder outputs."""
import hashlib,json
import torch
from facekit.paths import MODEL,DATA,MODEL_ID,MODEL_REVISION
from facekit.prompt_cache import cache_identity,cache_path,encoder_provenance,load_prompt,prompt_for


def main():
    directory=DATA/'cache'/'prompt_embeddings'
    provenance=encoder_provenance(MODEL)
    for mode in (True,False):
        prompt=prompt_for(mode);identity=cache_identity(MODEL,prompt);dest=cache_path(identity)
        if dest.is_file():load_prompt(MODEL,mode,device='cpu');continue
        matches=[]
        for path in directory.glob('*.pt'):
            data=torch.load(path,map_location='cpu',weights_only=True)
            old=data.get('identity',{}) if isinstance(data,dict) else {}
            if old.get('prompt')!=prompt or old.get('model_id')!=MODEL_ID or old.get('model_revision')!=MODEL_REVISION or 'weights' not in old:continue
            weight_stats=[{'name':p['name'].split('/',1)[1],'size':p['size'],'mtime_ns':p['mtime_ns']} for p in provenance if p['name'].startswith('text_encoder/') and p['name'].endswith('.safetensors')]
            if old['weights']!=weight_stats:raise ValueError('Encoder weights changed since cache creation')
            for name,digest in old['configs'].items():
                if hashlib.sha256((MODEL/name).read_bytes()).hexdigest()!=digest:raise ValueError('Encoder or tokenizer configuration changed')
            if old.get('diffusers_version')!=identity['diffusers_version'] or old.get('transformers_version')!=identity['transformers_version']:
                raise ValueError('Library versions changed since cache creation')
            matches.append(data['embedding'])
        if len(matches)!=1:raise ValueError(f'Expected one valid earlier embedding for {prompt!r}, found {len(matches)}')
        temp=dest.with_suffix('.pt.tmp');torch.save({'identity':identity,'encoder_provenance':provenance,'embedding':matches[0]},temp);temp.replace(dest)
        load_prompt(MODEL,mode,device='cpu');print('MIGRATED',json.dumps({'mode':mode,'path':str(dest)}),flush=True)


if __name__=='__main__':main()
