# FM2Edge

Frozen DINOv2/v3 knowledge-distillation experiments are documented in
[the Japanese KD PC guide](docs/KNOWLEDGE_DISTILLATION_PC_JA.md).
The existing Baseline, generalization-study and foundation-probe workflows remain available.

PyTorch experiment foundation for evaluating lightweight semantic segmentation
on entirely unseen machine domains. Phase 1 provides a leakage-safe baseline:

```text
external layout -> canonical manifest -> machine-disjoint split
                -> PIDNet-S training -> validation checkpoint selection
                -> one-time unseen-machine evaluation and artifacts
```

## Phase 1 scope

- Cityscapes adapter (`city` is treated as `machine_id`)
- Configurable company-layout adapter
- Reproducible train/validation/test machine folds
- Official-structure PIDNet-S with a common model output
- PyTorch PP-LiteSeg-STDC1 with the same model output contract
- Dependency-free PyTorch MobileNetV3-Large + LR-ASPP with the same contract
- Cross-entropy + Dice training objective
- IoU, Dice, precision, recall, Boundary F1, confidence and entropy
- Per-image, per-delay, per-machine CSV summaries
- Best/worst/random prediction panels
- Config, split, environment, checkpoints, history and metrics per run

Teacher models, knowledge distillation, domain augmentation, balanced
sampling, HTML reports and SBC benchmarks belong to later phases.

## Frozen DINO foundation probes

The foundation-probe phase evaluates whether frozen DINO patch features help
segmentation on held-out machines. It supports DINOv2 ViT-S/14 and DINOv3
ViT-S/16, linear and lightweight convolutional probes, validated feature
caching, and a DINOv2 random-initialization control.

See `docs/FOUNDATION_PROBE_PC_JA.md` for the company-PC workflow. Official DINO
repositories, weights, company data, cached features, and credentials remain
outside Git.

## Immediate Colab PoC with Oxford-IIIT Pet

The disposable Oxford-IIIT Pet adapter can download the public segmentation
dataset immediately and treat breed as a temporary domain proxy. It is isolated
from the core/company pipeline and can be deleted later without changing
training or evaluation code.

See [docs/OXFORD_PET_POC.md](docs/OXFORD_PET_POC.md) for commands, label mapping,
limitations, license notes, and the exact list of removable files. A runnable
Colab workflow is provided at
`notebooks/phase1_5_oxford_pet_colab.ipynb`. For this private repository, follow
[docs/COLAB_SETUP.md](docs/COLAB_SETUP.md) to grant read-only access through
Colab Secrets without embedding a token in the notebook.

## Setup

Python 3.10+ is supported. In a fresh virtual environment or Colab runtime:

```bash
pip install -e ".[dev,tensorboard]"
```

For CUDA, install the PyTorch build matching the runtime first, following the
official PyTorch selector, and then install this project.

## Five-minute synthetic smoke test

The generated images are deliberately tiny and test plumbing, not accuracy.

```bash
python scripts/create_synthetic_dataset.py --output data/synthetic
python scripts/prepare_dataset.py \
  --dataset machine \
  --root data/synthetic \
  --output data/synthetic/manifest.csv
python scripts/make_splits.py \
  --manifest data/synthetic/manifest.csv \
  --output-dir configs/splits/synthetic \
  --folds 5 \
  --seed 42
python scripts/train.py --config configs/experiments/pidnet_s_synthetic.yaml
python scripts/evaluate.py --config configs/experiments/pidnet_s_synthetic.yaml
```

The PP-LiteSeg-STDC1 plumbing can be checked against the same split with:

```bash
python scripts/train.py \
  --config configs/experiments/pp_liteseg_stdc1_synthetic.yaml
python scripts/evaluate.py \
  --config configs/experiments/pp_liteseg_stdc1_synthetic.yaml
```

MobileNetV3-Large + LR-ASPP can be checked without installing Torchvision:

```bash
python scripts/train.py \
  --config configs/experiments/mobilenet_v3_lraspp_synthetic.yaml
python scripts/evaluate.py \
  --config configs/experiments/mobilenet_v3_lraspp_synthetic.yaml
```

The test command is:

```bash
pytest
```

## Cityscapes PoC

Download and extract `leftImg8bit_trainvaltest` and `gtFine_trainvaltest`
according to the Cityscapes terms. Only labeled official train/val images are
indexed; the unlabeled official test set is not used.

```bash
python scripts/prepare_dataset.py \
  --dataset cityscapes \
  --root data/cityscapes \
  --output data/cityscapes/manifest.csv \
  --max-machines 12 \
  --max-samples-per-machine 200 \
  --seed 42

python scripts/make_splits.py \
  --manifest data/cityscapes/manifest.csv \
  --output-dir configs/splits/cityscapes \
  --folds 5 \
  --seed 42

python scripts/train.py --config configs/experiments/pidnet_s_cityscapes.yaml
python scripts/evaluate.py --config configs/experiments/pidnet_s_cityscapes.yaml
```

Do not use a Cityscapes-trained segmentation checkpoint as initialization for
this experiment: it may already contain information from held-out cities.

## Company-layout adapter

For the complete Windows GPU-PC workflow that validates JPG/PNG pairs, runs
all three Students over five folds, and aggregates results, see
[`docs/MACHINE_BASELINE_PC.md`](docs/MACHINE_BASELINE_PC.md).

After completing that baseline, the configurable machine-count, image-count,
augmentation, heatmap, input-domain, and feature-distribution study is
documented in
[`docs/MACHINE_GENERALIZATION_ANALYSIS_PC.md`](docs/MACHINE_GENERALIZATION_ANALYSIS_PC.md).
The Japanese quick-start is
[`docs/MACHINE_GENERALIZATION_ANALYSIS_PC_JA.md`](docs/MACHINE_GENERALIZATION_ANALYSIS_PC_JA.md).

The expected default layout is:

```text
root/
├── images/<machine_id>/delay/<delay>/<image>
└── masks/<machine_id>/delay/<delay>/<image>
```

If the phase folder is not named `delay`, pass it without editing code:

```bash
python scripts/prepare_dataset.py \
  --dataset machine \
  --root /path/to/root \
  --phase-folder acquisition_phase \
  --output data/company/manifest.csv
```

Image and mask paths in the manifest are relative to `data.root`, so the same
manifest remains usable after moving the dataset root. Mask value conversion is
configured through `mask_value_map`; for example `{0: 0, 255: 1}`.

## Split and leakage contract

Each split YAML stores `train_machines`, `val_machines`, `test_machines`, seed,
and fold number. Loading fails if the sets overlap. `train.py` only constructs
train and validation datasets. The test set is constructed only by
`evaluate.py`, after a validation-selected checkpoint exists.

Split files must be reviewed and committed before substantive experiments.
Changing a split creates a different experiment; do not overwrite a split used
for a reported result.

## Result layout

```text
results/<experiment>/fold_<NN>/
├── config.yaml
├── split.yaml
├── environment.json
├── manifest.csv
├── checkpoints/{best,last}.pt
├── history/epochs.csv
├── tensorboard/
├── metrics/{per_image,per_machine,per_delay}.csv
├── metrics/summary.json
└── predictions/{best,worst,random}/
```

`summary.json` includes machine-level IoU standard deviation and worst-machine
IoU. Average performance alone should not be used for conclusions.

## Current limitations

- Phase 1 uses deterministic resize/normalization only; domain augmentation is
  intentionally deferred.
- Boundary F1 uses a two-pixel tolerance at the resized evaluation resolution.
  Ignored bands are nearest-class-filled for boundary scoring only, while all
  region metrics continue to exclude ignored pixels.
- The PIDNet boundary head is exposed but boundary supervision is deferred.
- Pretrained weights are not downloaded automatically. This avoids hidden
  network access and forces weight provenance to be recorded.
- PIDNet-S uses BatchNorm; Phase 1 therefore requires a training batch size of
  at least two. Gradient accumulation does not change BatchNorm batch size.
- CPU measurements must eventually be repeated on the target SBC; Colab CPU
  numbers are not representative.

See [THIRD_PARTY.md](THIRD_PARTY.md) before using external data or weights.
