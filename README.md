# Face maps pipeline

Snapshot of the face-data generation, rendering, FLUX training, and evaluation code. The current expression-aware pipeline is documented in [nebius/expressions/README.md](nebius/expressions/README.md); earlier experiments are described in [nebius/README.md](nebius/README.md).

This is a code snapshot. Datasets, generated images, model weights, credentials, and runtime environments are not included. Provision those separately before running any scripts. Scripts under `nebius/` reflect the original VM environment and may require path and dependency adjustments elsewhere. The shared root defaults to `~/datasets/faces` and can be overridden with `FACE_SHARED_ROOT`.

The snapshot omits scheduler-specific launchers, workflow configuration, and diagnostics tied to the original cluster. The original face pipeline remains separate and unchanged.
