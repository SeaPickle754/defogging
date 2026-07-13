# Code Overview

This folder keeps the paper workflows separated by role.

- `nafnet_finetuning/`: NAFNet training, evaluation, and inference utilities. Start here for released-model inference.
- `synthetic_finetuning/`: spatial synthetic-fog generator and Mapillary synthetic fine-tuning workflow.
- `fog_chamber_benchmark/`: paired fog-chamber benchmark wrapper. Full use requires third-party model source trees.
- `classification_semantic_preservation/`: ResNet-50 semantic-preservation check.
- `dark_channel_prior/`: classical dark-channel-prior baseline.
- `fog_statistics/`: fog proxy, PSD, and paired structure-loss analyses.

Beginner inference entrypoint:

```bash
python code/nafnet_finetuning/run_defogging_inference.py --help
```

Most training and evaluation scripts accept command-line paths. External datasets and checkpoints must be downloaded locally before running those workflows.
