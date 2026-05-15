# Sapiens2: Human Image Matting

Per-pixel alpha matting for human subjects. Predicts a soft foreground mask
(`alpha` in `[0, 1]`) plus a pre-multiplied foreground RGB output.

## Inference Guide

![Sapiens2-1B matting demo](assets/sapiens2_1b_matting_demo.gif)

Runs on demo set (`demo/data`, 100 frames) by default:

```bash
cd $SAPIENS_ROOT/sapiens/dense
./scripts/demo/matting.sh
```

Open the script and adjust:
- `INPUT` — path to your image directory (default: `../../demo/data`)
- `OUTPUT` — where to save visualizations
- `MODEL_NAME` — model size (default: `sapiens2_1b`)
- `JOBS_PER_GPU`, `GPU_IDS` — parallelism (defaults: 3 jobs/GPU on GPUs 0–7)

Outputs per image:
- Side-by-side visualization: `[input | alpha matte | foreground on chroma green]`
- `--save_pred` (on by default in the script) additionally writes `<name>_alpha.npy`
  with the raw alpha as `float32` in `[0, 1]`.

## Training Guide

In-the-wild matting on a mixture of relit / synthetic / captured datasets,
loaded via the `MattingBaseDataset` (annotation-driven JSON manifest) and
`MattingGSSDataset` (RGBA image manifest).

The dataset paths and `pretrained_checkpoint` in
`configs/matting/gss_p3m_metasim/sapiens2_1b_matting_gss_p3m_metasim-1024x768.py`
point to the internal locations used to train the released model — edit them
to point at your own data and backbone checkpoint before launching training.

Launch single-node multi-GPU training:

```bash
cd $SAPIENS_ROOT/sapiens/dense
./scripts/matting/train/sapiens2_1b/node.sh
```

Open the script and adjust:
- `DEVICES` — GPU IDs (default: `0,1,2,3,4,5,6,7`)
- `TRAIN_BATCH_SIZE_PER_GPU` — per-GPU batch size (default: 8)
- `mode` — set to `debug` for a single-process run

Outputs are written under `Outputs/matting/train/<MODEL>/node/<timestamp>/`.

## Dataset Format

`MattingBaseDataset` expects an annotation JSON of the form:

```json
[
  {"image": "relpath/to/img.png", "mask": "relpath/to/mask.png"},
  ...
]
```

Where `mask` is either:
- a single-channel alpha image, or
- an RGBA image whose alpha channel is the matte (RGB channels treated as the
  pre-multiplied foreground).

`MattingGSSDataset` expects a `.txt` manifest with one RGBA image path per line.
