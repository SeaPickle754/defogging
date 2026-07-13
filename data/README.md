# Data

Raw image datasets are not stored in Git. Download the large image assets from Kaggle and arrange them locally before running training or paired evaluation.

## Public Assets

- Fog-chamber dataset: https://www.kaggle.com/datasets/alingold/fog-chamber
- Mapillary Vistas clear images for synthetic fine-tuning: https://www.kaggle.com/datasets/kaggleprollc/mapillary-vistas-image-data-collection
- Source archive for display images: https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings

The model weights are separate from the image datasets:

- https://www.kaggle.com/models/alingold/fog-removal

## Expected Local Layout

For paired fog-chamber workflows, the important requirement is matching category and filename:

```text
data/fog_chamber/
  foggy/
    apparel/image0000.jpg
    cars/image0000.jpg
    ...
  ground_truth_matched/
    apparel/image0000.jpg
    cars/image0000.jpg
    ...
```

The matched clear targets contain 5,495 images across six categories. The paper split assigns every 10th sorted image within each category to the held-out test split, giving 552 test pairs.

## Qualitative Real-Fog Examples

The paper also used unpaired real-fog examples:

- `aircraft_window_fog/`: aircraft-window examples; qualitative only.
- `free_flowing_fog/`: free-flowing outdoor fog examples; qualitative only.

These examples do not have paired clear targets, so they are used for visual transfer tests rather than PSNR/SSIM evaluation.

## External Datasets

Some advanced workflows require external datasets that are not redistributed here:

- O-HAZE and NH-HAZE paired real-haze datasets.
- NTIRE 2026 nighttime haze training pairs.
- Third-party model source trees for the full 30-model benchmark.

Check each external dataset or model repository for its own license before downloading or redistributing it.
