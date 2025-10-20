# TinyPhysics Training Pipeline

This package contains the tooling required to extract supervised examples from
openpilot logs and train a TinyPhysics surrogate model that is compatible with
`controls_challenge/tinyphysics.py`.

## Quick start

1. Install dependencies for evaluation (see the root `README.md`) and the
   training extras:
   ```bash
   uv pip install -r controls_challenge/training/requirements.txt
   ```
   To enable `--export-onnx`, install the optional `onnxscript` dependency:
    ```bash
    uv pip install onnxscript
    ```
2. Provide one or more routes/segments (either explicitly with `--routes` /
   `--routes-file` or by fetching every route for a dongle with `--dongle-id`).
3. Launch training:
   ```bash
   python -m controls_challenge.training.train \
     --dongle-id 3b58edf884ab4eaf \
     --routes-limit 50 \
     --samples-path controls_challenge/data/tinyphysics_segments \
     --write-samples \
     --extract-threads 4 \
     --export-onnx \
     --onnx-path controls_challenge/models/tinyphysics_trained.onnx \
     --onnx-opset 18
   ```

The command above will:

- download and cache the requested logs (respecting `FILEREADER_CACHE` if set),
- extract one-minute TinyPhysics segments that mirror the public dataset format,
- write each segment to `<NNNNN>.csv` (when `--write-samples` is set),
- split the data by route into train/validation sets,
- train a lightweight recurrent model that predicts the next lateral
  acceleration token, and
- optionally export the trained weights to ONNX so they can be evaluated by the
  existing simulator harness.

### Additional options

- Use `--git-commit <hash>` (repeatable) to keep only routes that match specific
  openpilot commit hashes.
- Provide multiple `--dongle-id` flags to merge data from several devices.
- Set `--routes-limit 0` to fetch every available route for each dongle.
- Increase `--extract-threads` to parallelize data extraction and writing.

The default artefacts are stored under `controls_challenge/training_runs/tinyphysics`.
You can point `--onnx-path` at `controls_challenge/models/` to swap in your newly
trained model for the evaluation scripts.

## Components

- `data.py` — streams logs, slices them into one-minute segments, and emits both
  dataset-style CSVs and matching label arrays for training.
- `dataset.py` — converts cached segments into autoregressive PyTorch datasets
  that mirror the simulator inputs and apply global normalization.
- `model.py` — defines a compact GRU-based classifier that emits logits over the
  TinyPhysics token vocabulary.
- `train.py` — orchestrates extraction, caching, training, validation, and ONNX export.

Each piece is modular. You can swap in different model architectures or add new
metrics without changing the extraction pipeline.
