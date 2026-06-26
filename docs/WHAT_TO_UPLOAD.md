# Upload Checklist

## GitHub Repository

Commit source code, documentation, model metadata, and the small `results/` tables.

The `results/` folder is small enough for Git, but it has more than 100 files. Use the command line instead of GitHub's browser uploader:

```bash
git status --short
git add results
git commit -m "Add computational defogging result tables"
git push origin main
```

## Kaggle Model Asset

Upload model weights and inference metadata to:

https://www.kaggle.com/models/alingold/fog-removal

Expected files:

- `fog_chamber_nafnet_model_state_20260615.pth`
- `synthetic_finetuned_nafnet_model_state_20260615.pt`
- `run_config_fog_chamber_nafnet.json`
- `run_config_synthetic_finetuned_nafnet.json`
- `SHA256SUMS.txt`
- `checkpoints_manifest.csv`

## Kaggle Datasets

Fog-chamber dataset:

- https://www.kaggle.com/datasets/alingold/fog-chamber

Synthetic fine-tuning source dataset:

- https://www.kaggle.com/datasets/kaggleprollc/mapillary-vistas-image-data-collection

Source image archive:

- https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings

## Do Not Upload To Git

- checkpoint binaries (`*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`)
- raw image datasets
- rendered prediction folders
- demo videos
- private CHPC scripts
- local backups
