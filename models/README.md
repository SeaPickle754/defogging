# Models

Trained checkpoint binaries are not tracked in Git. Download them from:

https://www.kaggle.com/models/alingold/fog-removal

Expected checkpoint files:

- `fog_chamber_nafnet_model_state_20260615.pth`
- `synthetic_finetuned_nafnet_model_state_20260615.pt`

Small metadata files in this folder are safe to track:

- `checkpoints_manifest.csv`: model roles, expected filenames, SHA256 hashes, and byte sizes.
- `SHA256SUMS.txt`: checksum file for local verification.
- `run_config_fog_chamber_nafnet.json`: architecture/config metadata for the fog-chamber checkpoint.
- `run_config_synthetic_finetuned_nafnet.json`: architecture/config metadata for the synthetic fine-tuned checkpoint.

Verify downloaded checkpoints from the folder containing the files:

```bash
sha256sum -c SHA256SUMS.txt
```

Task-specific O-HAZE/NH-HAZE and NTIRE checkpoints are not included in this public release.
