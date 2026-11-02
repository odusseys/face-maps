# Face-map replication on Nebius

The copied source and all new assets, models, data, logs, and checkpoints live
under `~/datasets/faces`. All generation and processing run on this VM.
The separate Conda environment is `faces`; it uses the existing CUDA 13 PyTorch
installation and pins the original experiment's `diffusers` version.

The preparation script generates 500 faces with 100 holdouts, produces the
similarity-aligned 97th-percentile displacement maps and FLUX.2 latents, and
caches both text prompts. It is resumable after interruption:

```bash
ssh nebius
bash ~/datasets/faces/code/face_pipeline/nebius/prepare_faces.sh
```

The original similarity dataset is at
`~/datasets/faces/face_maps/datasets/faces500-similarity-linear-q97`.
The unnormalized source is retained separately as `faces500-centered`.
Displacement training now uses the separate fixed-texture dataset described below.

Once preparation completes, launch the displacement experiment with:

```bash
bash ~/datasets/faces/code/face_pipeline/nebius/train_faces_uvq97.sh
```

The training command resumes its latest checkpoint when available. W&B uses
the VM's existing account on CoreWeave Forge, in the same project as the source
experiment: `flux2-face-displacement-only`. It does not upload checkpoints or
source code. Its project-local credential file is readable only by the user.
The VM is preemptible, so a stopped run must be restarted; its datasets and
checkpoints are stored on its persistent boot disk.

The separate `smoke-6` fixture and `smoke_train.py` deliberately use only three
training faces and three holdout faces to test the full pipeline. They are not
the 500-face training dataset. The source experiment's q97 normalization can
clip more than 3% of UV pixels because its percentile is computed over vertices.

## UV-aligned dataset

`faces500-uv-aligned-uvq97` is a separate dataset built from the same faces on
Nebius. Its similarity transform is fitted only to vertices represented by the
face UV triangles. Its signed-linear displacement bound is the 97th percentile
of absolute XYZ components at interpolated UV pixels from the training split;
holdout examples do not set the bound. Portrait renders and their latents are
reused, while displacement and paired-map latents are generated anew.

To regenerate or resume the corrected data and latents on this VM:

```bash
cd ~/datasets/faces/code/face_pipeline
export FACE_SHARED_ROOT="$HOME/datasets/faces" PYTHONNOUSERSITE=1 PYTHONPATH="$PWD"
export HF_HOME="$FACE_SHARED_ROOT/face_maps/cache/huggingface"
export XDG_CACHE_HOME="$FACE_SHARED_ROOT/face_maps/cache"
~/miniconda3/envs/faces/bin/python -u nebius/build_uv_aligned_dataset.py
~/miniconda3/envs/faces/bin/python -u encode_shape_latents.py --dataset "$FACE_SHARED_ROOT/face_maps/datasets/faces500-uv-aligned-uvq97"
~/miniconda3/envs/faces/bin/python -u nebius/audit_uv_aligned_dataset.py
```

To launch or resume the fixed-texture displacement experiment, run:

```bash
bash ~/datasets/faces/code/face_pipeline/nebius/train_faces_uvq97.sh
```

The clipping comparison is saved under
`~/datasets/faces/face_maps/audits/nebius-uv-alignment-comparison.json`.
The previous datasets and run remain available for controlled comparisons.
A successful smoke test establishes that training runs; it does not establish
that the model has learned to use portrait conditioning.

## Fixed-texture displacement training

`faces500-uv-aligned-fixed-texture` preserves all original shapes, splits,
UV-only alignment, displacement targets and target latents. Portraits are
re-rendered using the same skin material for every face and re-encoded.
The dataset does not include paired-map targets and is for displacement-only
training. The reference material used in ground-truth and predicted debug
renders is the same texture. The displacement training launcher verifies all 500
metadata records and the texture checksum before starting.

The experiment uses 10,000 steps, resumable checkpoints every 200 steps, and logs a three-face
inference comparison to CoreWeave Forge W&B every 200 steps. The launcher
rejects a different inference interval and uses a lock to prevent duplicate
launches. Its output is `face_maps/training/flux2_uv_fixed_texture`.

## Moderate shape variation

The standard displacement launcher targets the separate
`face_maps/datasets/faces500-moderate-uv-fixed-texture` dataset, with the
moderated shape distribution from the 20-face preview, one skin texture, and
a 400/100 train/holdout split. Its run and checkpoints are separate under
`face_maps/training/flux2_moderate_fixed_texture`. The earlier fixed-texture
run and checkpoint remain in `face_maps/training/flux2_uv_fixed_texture`.

Run `bash ~/datasets/faces/code/face_pipeline/nebius/train_faces_uvq97.sh` to launch
or resume the moderate-shape run when desired. It logs inference comparisons
at every 200 steps. Dataset preparation does not start training.

## Diverse paired texture and displacement training

`faces1000-visibleface-uv-paired` contains 1,000 freshly generated identities,
with 900 training and 100 holdout faces. It retains the moderated shape range,
UV-mapped-vertex similarity alignment, and a signed-linear displacement range
calibrated on training UV coordinates. It uses 20 skin materials and six
face-visible hairstyles. The hair portrait has a random perspective camera
distance, with UV-face coverage checked before every render; the hairless view
uses a standard orthographic camera.

Paired training conditions on the hair portrait and predicts a 1024x512 image:
texture on the left and displacement on the right. The debug comparison shows
the hair input, the hairless ground-truth portrait at standard view, the predicted
face at standard view, and ground-truth/predicted texture and displacement
panels. Each comparison and checkpoint is saved every 200 steps.

After data preparation, run or resume the 10,000-step experiment with:

```bash
bash ~/datasets/faces/code/face_pipeline/nebius/train_faces.sh
```

The launcher validates identity pairing, diversity, camera framing and latent
shapes before training, and writes to a separate checkpoint directory. The
preparation and generated data live entirely on the Nebius VM.
