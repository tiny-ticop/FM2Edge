# GPU PC workflow for machine-generalization factor analysis

This workflow analyzes why lightweight segmentation models succeed or fail on
unseen machines. It is separate from, and does not replace, the original
three-model baseline workflow in `docs/MACHINE_BASELINE_PC.md`.

## 1. Install the analysis environment

From a fresh clone, create/activate the same CUDA-enabled Python environment as
the baseline, then install the analysis extra:

```powershell
pip install -e ".[dev,tensorboard,analysis]"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')"
pytest
```

Do not run the full study unless CUDA availability is `True`.

## 2. Reuse the data and folds

Use the same validated paths as the baseline:

```text
data/raw/<machine>/delay/<delay>/<image>.jpg
data/masks/<machine>/delay/<delay>/<image>.png
data/company/manifest.csv
configs/splits/company/fold_01.yaml ...
```

If this is a new clone, copy the private `data/` directory from approved
storage or rebuild the manifest and folds exactly as described in
`docs/MACHINE_BASELINE_PC.md`. Never add private data or results to Git.

Previous baseline `results/` may be copied into the clone for reference, but
the study remains self-contained and creates its own reference runs under
`results/generalization_analysis/`.

## 3. Configure the study

Edit only:

```text
configs/analyses/machine_generalization.yaml
```

The default five-machine study contains:

- three models and five folds;
- one reference condition;
- training-machine counts 1, 2, and 3 at a fixed total of 50 images;
- 10, 25, 50, and all images per training machine;
- brightness/contrast, color, noise, blur, and combined augmentation.

Common changes:

```yaml
folds: [1, 2, 3, 4, 5]
seeds: [42]

suites:
  machine_diversity:
    machine_counts: [1, 2, 3]
    total_images: 50
    max_combinations_per_count: null
  images_per_machine:
    counts: [10, 25, 50, all]
  augmentation:
    presets: [brightness_contrast, color, noise, blur, combined]
```

When more machines are added, regenerate machine-disjoint folds first. Every
requested `machine_counts` value must be no larger than the smallest train pool
among the selected folds. Use `max_combinations_per_count` to cap combinations
when the number of machines becomes large. The planner never silently reduces
machine counts or requested image counts.

Any change to sampling, augmentation strength, training settings, machines, or
image allocation changes the run fingerprint, so an old completed result is
not mistaken for the new condition.

## 4. Generate and review the immutable plan

```powershell
python scripts/plan_generalization_study.py
Get-Content results\generalization_analysis\plan\plan_summary.json
```

The default config generates 240 runs. Review both files before starting:

```text
results/generalization_analysis/plan/plan_summary.json
results/generalization_analysis/plan/experiment_plan.csv
```

Each row fixes the model, fold, train/validation/test machines, exact train
sample IDs, augmentation, seed, generated config, and result directory.

## 5. Run a smoke check

This executes one reference fold for each model for two epochs in separate
smoke directories:

```powershell
python scripts/run_generalization_study.py --smoke
```

If required:

```powershell
python scripts/run_generalization_study.py --smoke --smoke-epochs 2 --rerun
```

Inspect input, GT, prediction, and error panels below
`results/generalization_analysis/smoke/` before starting the full study.

## 6. Run suites independently

```powershell
python scripts/run_generalization_study.py --suite reference
python scripts/run_generalization_study.py --suite machine_diversity
python scripts/run_generalization_study.py --suite images_per_machine
python scripts/run_generalization_study.py --suite augmentation
```

Useful filters:

```powershell
python scripts/run_generalization_study.py --suite augmentation --models pidnet_s
python scripts/run_generalization_study.py --suite machine_diversity --folds 1 2
python scripts/run_generalization_study.py --suite images_per_machine --limit 3
```

Completed runs are skipped. If execution stops, run the same command again
without `--rerun`. The currently incomplete run restarts, while completed runs
remain untouched. Check progress and errors in:

```text
results/generalization_analysis/run_status.csv
results/generalization_analysis/runs/<run_id>/fold_<NN>/run.log
```

The analysis config stores a lightweight best-model checkpoint and omits the
optimizer-heavy last checkpoint by default. This substantially reduces disk
usage across hundreds of runs. Set `keep_last_checkpoint: true` only when last
checkpoints are required.

## 7. Generate tables, heatmaps, and factor curves

Analysis can be regenerated whenever more runs finish:

```powershell
python scripts/analyze_generalization_study.py
```

Outputs:

```text
results/generalization_analysis/analysis/
├─ report.md
├─ tables/all_runs.csv
├─ tables/aggregate_results.csv
├─ tables/paired_effects.csv
├─ figures/machine_count_iou.png
├─ figures/images_per_machine_iou.png
├─ figures/augmentation_iou.png
├─ figures/machine_heatmap_<model>.png
└─ domain_statistics/
```

Confidence intervals cluster by unknown test machine. With only five machines,
treat them as exploratory uncertainty estimates rather than definitive
significance tests.

## 8. Extract model-feature distributions

After reference runs finish:

```powershell
python scripts/extract_generalization_features.py
```

After combined-augmentation runs also finish:

```powershell
python scripts/extract_generalization_features.py --include-combined
```

For every completed model/fold, this saves global, foreground, and background
feature vectors, a PCA plot colored by machine and split role, machine
silhouette score, and train-to-test centroid distance under:

```text
results/generalization_analysis/analysis/features/
```

## 9. Add confirmation seeds only after screening

After selecting the useful strategy, change `seeds` to `[42, 43, 44]` and
disable unrelated suites in the study YAML. Regenerate the plan and run only
the reference and selected strategy. This avoids multiplying all exploratory
conditions by three.

## 10. Preserve and share results

Copy the entire `results/generalization_analysis/` directory to approved
storage. The most important review artifacts are:

```text
analysis/report.md
analysis/tables/aggregate_results.csv
analysis/tables/paired_effects.csv
analysis/figures/
analysis/features/feature_summary.csv
run_status.csv
```

GPU-PC inference timing is suitable only for comparisons on that PC. Final
deployment latency must still be measured on the target edge device.
