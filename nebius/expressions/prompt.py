"""Cache the expression task's prompt embedding without training a model."""
import json
import torch
from diffusers import Flux2KleinPipeline
from transformers import AutoTokenizer,Qwen3ForCausalLM
from facekit.paths import MODEL
from facekit.prompt_cache import MAX_LENGTH,cache_identity,cache_path,encoder_provenance

PROMPT='Create a face albedo UV map, neutral face shape displacement UV map, and expression displacement UV map from the expressive portrait.'

def load(device='cuda'):
    identity=cache_identity(MODEL,PROMPT);path=cache_path(identity)
    data=torch.load(path,map_location='cpu',weights_only=True);embedding=data['embedding']
    dim=json.loads((MODEL/'transformer/config.json').read_text())['joint_attention_dim']
    if data['identity']!=identity or tuple(embedding.shape)!=(1,MAX_LENGTH,dim) or embedding.dtype!=torch.bfloat16 or not torch.isfinite(embedding).all():raise ValueError('Invalid expression prompt cache')
    return embedding.to(device)

def main():
    identity=cache_identity(MODEL,PROMPT);path=cache_path(identity)
    if path.exists():load('cpu');print('PROMPT_READY',path,flush=True);return
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),subfolder='tokenizer',local_files_only=True)
    model=Qwen3ForCausalLM.from_pretrained(str(MODEL),subfolder='text_encoder',torch_dtype=torch.bfloat16,local_files_only=True).to('cpu').eval()
    with torch.inference_mode():
        embedding=Flux2KleinPipeline._get_qwen3_prompt_embeds(model,tokenizer,PROMPT,device=torch.device('cpu'),max_sequence_length=MAX_LENGTH).detach().to('cpu',torch.bfloat16).contiguous()
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pt.tmp')
    torch.save({'identity':identity,'encoder_provenance':encoder_provenance(MODEL),'embedding':embedding},tmp);tmp.replace(path)
    load('cpu');print('PROMPT_READY',path,flush=True)

if __name__=='__main__':main()
