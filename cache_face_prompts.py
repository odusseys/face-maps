#!/usr/bin/env python3
"""Prepare the fixed face-training prompt embeddings once, on the CPU."""
import gc
import json
import os

os.environ['CUDA_VISIBLE_DEVICES']=''
from facekit.paths import MODEL
from facekit.prompt_cache import MAX_LENGTH,cache_identity,cache_path,encoder_provenance,prompt_for


def main():
    import torch
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer,Qwen3ForCausalLM
    todo=[]
    for mode in (True,False):
        prompt=prompt_for(mode);identity=cache_identity(MODEL,prompt);path=cache_path(identity)
        if path.exists():
            from facekit.prompt_cache import load_prompt
            load_prompt(MODEL,mode,device='cpu');continue
        todo.append((mode,prompt,identity,path))
    if not todo:
        print('ALL_PROMPTS_CACHED',flush=True);return
    provenance=encoder_provenance(MODEL)
    print('LOADING_TEXT_ENCODER_CPU',len(todo),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),subfolder='tokenizer',local_files_only=True)
    model=Qwen3ForCausalLM.from_pretrained(str(MODEL),subfolder='text_encoder',torch_dtype=torch.bfloat16,local_files_only=True).to('cpu').eval()
    for mode,prompt,identity,path in todo:
        with torch.inference_mode():
            embedding=Flux2KleinPipeline._get_qwen3_prompt_embeds(model,tokenizer,prompt,device=torch.device('cpu'),max_sequence_length=MAX_LENGTH).detach().to('cpu',torch.bfloat16).contiguous()
        if embedding.ndim!=3 or embedding.shape[:2]!=(1,MAX_LENGTH) or not bool(torch.isfinite(embedding).all()):
            raise ValueError('Invalid text embedding')
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.pt.tmp');torch.save({'identity':identity,'encoder_provenance':provenance,'embedding':embedding},temp);temp.replace(path)
        print('CACHED',json.dumps({'displacement_only':mode,'path':str(path),'shape':list(embedding.shape)}),flush=True)
    del model,tokenizer;gc.collect()
    from facekit.prompt_cache import load_prompt
    for mode in (True,False):load_prompt(MODEL,mode,device='cpu')
    print('ALL_PROMPTS_CACHED',flush=True)


if __name__=='__main__':main()
