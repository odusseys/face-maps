#!/usr/bin/env python3
"""Matched-noise portrait swaps for both 600-step Nebius experiments."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import Flux2Transformer2DModel
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from facekit.paths import DATA, MODEL
from facekit.train_helpers import predict, prompt_embedding
from train_face_maps import add_lora


def chosen_records(dataset, split):
    paths = [p.parent for p in sorted((dataset/'samples').glob('*/complete.json'))
             if json.loads(p.read_text())['split'] == split]
    return paths[::max(1, len(paths)//12)][:12]


def experiment(model, embeds, dataset, checkpoint, step):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['step'] != step or set(state['lora']) != set(get_peft_model_state_dict(model)):
        raise ValueError('Unexpected checkpoint')
    if set_peft_model_state_dict(model, state['lora']).unexpected_keys:
        raise ValueError('Unexpected checkpoint weights')
    del state
    result = {}
    for split in ('train', 'holdout'):
        paths = chosen_records(dataset, split)
        result[split] = {}
        for sigma in (.5, .8, .95, 1.):
            good_losses, bad_losses, output_diffs = [], [], []
            for offset in range(0, len(paths), 4):
                batch = paths[offset:offset+4]
                cond = torch.stack([torch.load(p/'latents.pt', map_location='cpu', weights_only=True)['face_bare'] for p in batch]).to('cuda', torch.bfloat16)
                target = torch.stack([torch.load(dataset/'latents_displacement'/(p.name+'.pt'), map_location='cpu', weights_only=True) for p in batch]).to('cuda', torch.bfloat16)
                generator = torch.Generator(device='cuda').manual_seed(543 + offset + int(100*sigma))
                noise = torch.randn(target.shape, device='cuda', dtype=target.dtype, generator=generator)
                noisy = (1-sigma)*target + sigma*noise
                timestep = torch.full((len(batch),), sigma, device='cuda', dtype=target.dtype)
                with torch.inference_mode():
                    good = predict(model, noisy, cond, timestep, embeds).float()
                    bad = predict(model, noisy, cond.roll(1, 0), timestep, embeds).float()
                goal = (noise-target).float(); dims = (1, 2, 3)
                good_losses.extend(((good-goal)**2).mean(dims).cpu().tolist())
                bad_losses.extend(((bad-goal)**2).mean(dims).cpu().tolist())
                output_diffs.extend(((good-bad)**2).mean(dims).sqrt().cpu().tolist())
            good_mean, bad_mean = np.mean(good_losses), np.mean(bad_losses)
            result[split][str(sigma)] = {'ids': [p.name for p in paths],
                'correct_mse': good_losses, 'swapped_mse': bad_losses,
                'correct_mse_mean': float(good_mean), 'swapped_mse_mean': float(bad_mean),
                'relative_mse_increase_percent': float(100*(bad_mean-good_mean)/good_mean),
                'output_difference_rms_mean': float(np.mean(output_diffs))}
            print('SWAP', dataset.name, split, sigma, json.dumps(result[split][str(sigma)]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, default=600, choices=(200, 400, 600))
    args = parser.parse_args()
    datasets = DATA/'datasets'; training = DATA/'training'
    old = datasets/'faces500-similarity-linear-q97'
    new = datasets/'faces500-uv-aligned-uvq97'
    for split in ('train','holdout'):
        left, right = chosen_records(old, split), chosen_records(new, split)
        if [p.name for p in left] != [p.name for p in right]:
            raise ValueError('Comparison IDs differ')
        for a, b in zip(left, right):
            x = torch.load(a/'latents.pt', map_location='cpu', weights_only=True)['face_bare']
            y = torch.load(b/'latents.pt', map_location='cpu', weights_only=True)['face_bare']
            if not torch.equal(x, y): raise ValueError('Portrait latent changed')
    model = Flux2Transformer2DModel.from_pretrained(str(MODEL), subfolder='transformer',
             torch_dtype=torch.bfloat16, local_files_only=True).to('cuda')
    add_lora(model); model.eval()
    embeds = prompt_embedding(MODEL, True)
    report = {'method': 'Same 12 portraits per split, noise seed and sample order; the target and noisy latent stay fixed while portraits are cyclically swapped within each batch. MSE is over all latent positions. This is an intermediate diagnostic, not a final convergence test.',
              'baseline': experiment(model, embeds, old, training/f'nebius-uv-baseline-600/checkpoint-{args.step:07d}.pt', args.step),
              'corrected': experiment(model, embeds, new, training/f'nebius-uv-corrected-600/checkpoint-{args.step:07d}.pt', args.step)}
    path = DATA/f'audits/nebius-uv-conditioning-{args.step}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print('AUDIT_COMPLETE', path, flush=True)


if __name__ == '__main__': main()
