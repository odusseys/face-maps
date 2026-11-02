"""Versioned FLUX prompt embeddings, with no text-encoder load at training time."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from .paths import DATA,MODEL_ID,MODEL_REVISION

PAIRED_PROMPT='Create a face texture UV map and signed XYZ displacement UV map from the portrait.'
DISPLACEMENT_PROMPT='Create a signed XYZ face displacement UV map from the portrait.'
MAX_LENGTH=128


def prompt_for(displacement_only):
    return DISPLACEMENT_PROMPT if displacement_only else PAIRED_PROMPT


def cache_identity(model,prompt):
    model=Path(model).resolve()
    config=model/'transformer'/'config.json'
    transformer_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()
    return {'version':1,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'model_path':str(model),
            'prompt':prompt,'max_sequence_length':MAX_LENGTH,'output_layers':[9,18,27],
            'dtype':'bfloat16','transformer_config_sha256':transformer_config_sha256,
            'diffusers_version':importlib.metadata.version('diffusers'),
            'transformers_version':importlib.metadata.version('transformers')}


def encoder_provenance(model):
    """Record source files when building the cache, not when loading it."""
    model=Path(model).resolve();files=[]
    for directory in ('text_encoder','tokenizer'):
        for p in sorted((model/directory).iterdir()):
            if p.is_file():
                stat=p.stat();files.append({'name':str(p.relative_to(model)),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})
    if not any(p['name'].startswith('text_encoder/') and p['name'].endswith('.safetensors') for p in files):
        raise FileNotFoundError('No text-encoder weights found for prompt cache provenance')
    return files


def cache_path(identity):
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return DATA/'cache'/'prompt_embeddings'/f'{digest}.pt'


def load_prompt(model,displacement_only,device='cuda'):
    identity=cache_identity(model,prompt_for(displacement_only));path=cache_path(identity)
    if not path.is_file():
        raise FileNotFoundError('Prompt embedding cache missing; run cache_face_prompts.py before training.')
    data=torch.load(path,map_location='cpu',weights_only=True)
    embedding=data.get('embedding') if isinstance(data,dict) else None
    config=json.loads((Path(model)/'transformer'/'config.json').read_text())
    expected=(1,MAX_LENGTH,config['joint_attention_dim'])
    if not isinstance(data,dict) or data.get('identity')!=identity or not isinstance(embedding,torch.Tensor) or tuple(embedding.shape)!=expected or embedding.dtype!=torch.bfloat16 or not bool(torch.isfinite(embedding).all()):
        raise ValueError(f'Invalid or incompatible prompt embedding cache: {path}')
    print('PROMPT_CACHE_HIT',path,flush=True)
    return embedding.to(device=device,dtype=torch.bfloat16)
