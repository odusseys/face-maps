# Expressive 256px face dataset

The separate `faces1000-canonical-head-256` dataset reuses the earlier 1,000
identities and their split. Each identity has one independently sampled set of
MakeHuman Face Units, recorded in its metadata. The input is the expressive
hair portrait; its face remains fully in frame at the original randomized
camera distance or slightly farther if needed.

`geometry.npz` contains the aligned neutral shape offset from the reference and
the expression-only offset in shared canonical coordinates. The renderer
transforms each canonical expression back through the identity's neutral
alignment before producing the matching portrait; consequently the aligned
expressive mesh equals reference + shape + canonical expression. Identical
expression weights produce identical canonical offsets for any identity.

The maps are quantized, UV-rasterized approximations of those arrays, not a
lossless representation. The UV crop includes the full head island. Each
signed XYZ map has its own 97th-percentile scale computed from interpolated
training-set UV pixels. Signed logarithmic encoding spans four shape scales and
24 expression scales; the preview renderer applies the matching inverse.
Validation rejects clipping in either map.

Each training target is `maps.png`, a 768 x 256 row of albedo, neutral shape,
and expression, each 256 x 256. `latents.pt` contains the conditioning portrait
and the complete target row at matching resolution. Each latent is refined for
80 decoder optimization steps while VAE weights remain frozen; this improves
the pixel reconstruction without changing the image or map resolution. The
same portrait refinement is used for the external evaluation image. Debug
images have five rows: neutral face, expressive face, albedo, shape map, and
expression map, with ground truth and prediction adjacent in each row. The
hairless ground-truth and predicted faces are rendered using the same reference
view and lighting. Ground truth uses the full saved mesh; the predicted mesh
can vary only UV-covered vertices, since the network cannot infer vertices
outside these maps.

The preview samples two holdout identities and the fixed, aligned external
portrait at `face_maps/eval/reference-portrait/aligned.png`. Its unknown
ground-truth maps are marked unavailable. The external inference seed is fixed
across previews. The source image is fitted to a white synthetic silhouette
using a uniform scale and translation; uncovered areas are black.

The training entry point is `nebius/expressions/train.sh`, which uses the
existing CoreWeave Forge W&B configuration and defaults to 10,000 steps. It is
separate from data generation and must be invoked explicitly. It can resume an
existing checkpoint but does not write new checkpoints. It exports only the
final adapter weights when training completes; this is not resumable training
state. Preparation, rendering
checks, and forward-only model checks do not start training. Changes to the
training loop take effect when the Python process next starts.
