# GPU PC workflow for the five-machine baseline

This workflow compares PIDNet-S, PP-LiteSeg-STDC1, and MobileNetV3-Large +
LR-ASPP on five entirely held-out machine folds. It is intended for the first
company-data baseline, before teacher models or knowledge distillation.

## 1. Clone and create the environment

Run these commands in PowerShell from a work-approved location:

```powershell
git clone https://github.com/tiny-ticop/FM2Edge.git
cd FM2Edge
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the CUDA-enabled PyTorch build appropriate for the PC by using the
official PyTorch installation selector. Install this project after PyTorch:

```powershell
python -m pip install --upgrade pip
pip install -e ".[dev,tensorboard]"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')"
pytest
```

Do not start a full run unless `torch.cuda.is_available()` prints `True`.

## 2. Prepare images and masks

Place private data under the ignored `data/` directory. Do not commit it.

```text
data/
├─ raw/
│  ├─ machine_01/delay/90/image_001.jpg
│  ├─ machine_01/delay/91/image_002.jpg
│  ├─ machine_02/delay/...
│  ├─ machine_03/delay/...
│  ├─ machine_04/delay/...
│  └─ machine_05/delay/...
└─ masks/
   ├─ machine_01/delay/90/image_001.png
   ├─ machine_01/delay/91/image_002.png
   ├─ machine_02/delay/...
   ├─ machine_03/delay/...
   ├─ machine_04/delay/...
   └─ machine_05/delay/...
```

The source image may be JPG while its mask is PNG, but their relative folders
and filename stems must match. Masks must be single-channel, have the same
pixel dimensions as their images, and use only:

- `0`: background
- `1`: target/foreground
- `255`: ignore/unlabeled

Do not use JPEG masks, anti-aliased edges, or RGB color masks.

Build and validate the manifest. This scans every pair and stops on a missing
mask, size mismatch, RGB mask, or unexpected pixel value:

```powershell
python scripts/prepare_dataset.py --dataset machine --root data --images-dir raw --masks-dir masks --phase-folder delay --mask-extension .png --allowed-mask-values 0 1 255 --required-mask-values 0 1 --expected-machines 5 --output data/company/manifest.csv
```

Create five deterministic machine-disjoint folds:

```powershell
python scripts/make_splits.py --manifest data/company/manifest.csv --output-dir configs/splits/company --folds 5 --seed 42
Get-Content configs/splits/company/fold_01.yaml
```

With folders named exactly `machine_01` through `machine_05`, seed 42 assigns:

| Fold | Train | Validation | Test |
|---|---|---|---|
| 1 | 01, 03, 05 | 02 | 04 |
| 2 | 01, 04, 05 | 03 | 02 |
| 3 | 01, 02, 04 | 05 | 03 |
| 4 | 02, 03, 04 | 01 | 05 |
| 5 | 02, 03, 05 | 04 | 01 |

Review all generated split files before training. Folder names different from
the table produce a different deterministic assignment.

## 3. Run the smoke check

The smoke command trains and evaluates all three models on fold 1 for two
epochs. It uses separate experiment directories and cannot overwrite the full
runs.

```powershell
python scripts/run_machine_baselines.py --mode smoke
```

If GPU memory is insufficient, retry all models with the same smaller batch:

```powershell
python scripts/run_machine_baselines.py --mode smoke --batch-size 2 --rerun
```

On Windows, use `--num-workers 0` if worker-process startup causes an error.
Before the full run, inspect prediction panels under:

```text
results/machine_smoke_<model>/fold_01/predictions/
```

Confirm that input, ground truth, and prediction orientation are correct and
that the ground-truth foreground is visible.

## 4. Run all 15 experiments

The full command runs three models over all five folds. The maximum is 50
epochs, with validation-mIoU early stopping after 10 non-improving epochs.
Best checkpoints are selected only from validation machines. Keep batch size
and worker settings identical across models for a fair comparison.

```powershell
python scripts/run_machine_baselines.py --mode full
```

Epochs and resource settings can be explicitly overridden:

```powershell
python scripts/run_machine_baselines.py --mode full --epochs 30 --batch-size 2 --num-workers 0
```

Completed fold results are skipped when the command is run again. Use
`--rerun` only when intentionally replacing results. To run selected folds:

```powershell
python scripts/run_machine_baselines.py --mode full --folds 3 4 5
```

## 5. Results and recovery

Each model/fold directory contains:

```text
results/machine_<model>/fold_<NN>/
├─ checkpoints/{best,last}.pt
├─ history/{epochs.csv,training_summary.json}
├─ metrics/{summary.json,per_image.csv,per_machine.csv,per_delay.csv}
├─ predictions/{best,worst,random}/
├─ config.yaml
├─ split.yaml
├─ manifest.csv
└─ environment.json
```

The saved summaries include best epoch, completed epochs, training time,
forward-pass milliseconds per image, FPS, training/evaluation peak allocated
GPU memory, parameter count, and checkpoint size. These values are useful for
comparison on the same PC; they are not a substitute for final target-device
benchmarks.

Cross-fold comparison files are updated after each completed run:

```text
results/machine_baseline_comparison/full_runs.csv
results/machine_baseline_comparison/full_summary.json
```

Rebuild them without retraining with:

```powershell
python scripts/aggregate_machine_baselines.py --mode full
```

Copy the complete `results/` directory to approved storage after the run. It is
ignored by Git because it contains data-derived artifacts and checkpoints.

## Important interpretation limits

- All three baselines start from scratch; no pretrained weights are downloaded.
- The default resize is 640 x 416 pixels (width x height), preserving the
  1000 x 650 source aspect ratio while using dimensions friendly to the models.
- Five machines provide a useful first comparison but still give an uncertain
  estimate of deployment performance.
- Do not tune settings against test-machine results. Use validation machines,
  then evaluate each held-out test machine once.
