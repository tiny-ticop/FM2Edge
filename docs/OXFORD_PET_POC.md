# Oxford-IIIT Pet PoC

Oxford-IIIT Pet is a disposable public-data adapter for validating FM2Edge
before company images are available. Breed is used only as a proxy for
`machine_id`; results must not be interpreted as expected industrial-domain
performance.

## Why it is isolated

The core pipeline imports only the canonical manifest. Oxford-specific code is
limited to:

- `src/fm2edge/data/public/oxford_pet.py`
- `scripts/download_oxford_pet.py`
- `scripts/prepare_oxford_pet.py`
- `configs/datasets/oxford_pet.yaml`
- `configs/experiments/pidnet_s_oxford_pet_smoke.yaml`
- `configs/experiments/pp_liteseg_stdc1_oxford_pet_smoke.yaml`
- `notebooks/phase1_5_oxford_pet_colab.ipynb`
- `tests/test_oxford_pet.py`
- this document and its `THIRD_PARTY.md` row

Deleting those items does not change the company adapter, canonical Dataset,
split generation, models, trainer, evaluator, losses, or metrics.

## Domain and label mapping

The official `trainval.txt` and `test.txt` lists are combined into one manifest.
They are not used as the experimental split because they split images within
every breed. FM2Edge regenerates train/validation/test splits by breed.

```text
machine_id = breed parsed from <breed>_<number>
delay      = empty string
trimap 1   = pet foreground (class 1)
trimap 2   = background (class 0)
trimap 3   = uncertain border (ignore 255)
```

The original list membership remains in `split_source` for traceability.
Region metrics exclude trimap border pixels. Boundary F1 fills ignored regions
from their nearest semantic classes for boundary scoring only, so the ignored
trimap band does not erase the foreground/background transition.

## Local or Colab commands

```bash
python scripts/download_oxford_pet.py --root data/oxford_pet

python scripts/prepare_oxford_pet.py \
  --root data/oxford_pet \
  --output data/oxford_pet/manifest_smoke.csv \
  --max-breeds 12 \
  --max-samples-per-breed 40 \
  --seed 42

python scripts/make_splits.py \
  --manifest data/oxford_pet/manifest_smoke.csv \
  --output-dir configs/splits/oxford_pet_smoke \
  --folds 5 \
  --seed 42

python scripts/train.py \
  --config configs/experiments/pidnet_s_oxford_pet_smoke.yaml

python scripts/evaluate.py \
  --config configs/experiments/pidnet_s_oxford_pet_smoke.yaml
```

After accepting the PIDNet-S pipeline, PP-LiteSeg-STDC1 can use the exact same
manifest and split:

```bash
python scripts/train.py \
  --config configs/experiments/pp_liteseg_stdc1_oxford_pet_smoke.yaml

python scripts/evaluate.py \
  --config configs/experiments/pp_liteseg_stdc1_oxford_pet_smoke.yaml
```

This comparison configuration starts from scratch. Official PaddleSeg weights
use Paddle's checkpoint format and are not silently converted or loaded by the
PyTorch implementation.

The downloader stores verified archives under `data/oxford_pet/archives` and
reuses them on subsequent runs. `data/` and `results/` are ignored by Git.

## License boundary

The dataset is offered by Oxford VGG under CC BY-SA 4.0 for commercial and
research purposes, while copyright in individual images remains with their
original owners. This PoC does not establish that its data or trained artifacts
are suitable for a product. See `THIRD_PARTY.md` and the official dataset page.
