# Local Path Audit

Before upload, check for private workstation or CHPC paths:

```bash
rg -n "/home/|/mnt/|/media/|/scratch|@|u[0-9]{7}" .
```

Expected matches should be reviewed case by case. Public docs should use relative paths, Kaggle URLs, or placeholders such as `path/to/foggy_images`.

Also check for stale dataset folder names:

```bash
rg -n "archive_gt_matched" --glob '!docs/LOCAL_PATH_AUDIT.md' .
rg -n "general_code" --glob '!docs/LOCAL_PATH_AUDIT.md' .
```

The public paired-target folder name should be `ground_truth_matched/`, and synthetic fine-tuning imports should resolve to bundled code.
