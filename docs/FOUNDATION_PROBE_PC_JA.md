# Frozen DINO 基盤モデル評価：業務PC実行手順

既存の会社データ、manifest、5-fold splitを変更せず、Frozen DINOの特徴が未知機台のセグメンテーションに有効かを評価します。DINO本体は更新せず、Linearまたは軽量Conv headだけを学習します。

会社データ、DINO重み、特徴cacheはGitHubへ追加されません。`data/`、`checkpoints/`、`external/`、`results/`はgitignore対象です。

## 1. 環境と会社データの確認

```powershell
cd C:\path\to\FM2Edge
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,tensorboard,analysis]"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
pytest -q
Test-Path data\company\manifest.csv
Get-ChildItem configs\splits\company\fold_*.yaml
```

## 2. DINOv2公式repositoryと重み

DINOコードや重みはFM2Edgeへ組み込まず、公式repositoryをgitignore対象の`external/`へcloneします。

```powershell
git clone https://github.com/facebookresearch/dinov2.git external/dinov2
New-Item -ItemType Directory -Force checkpoints
curl.exe -L "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth" -o "checkpoints/dinov2_vits14_pretrain.pth"
```

URLと利用条件は実行時点の[DINOv2公式repository](https://github.com/facebookresearch/dinov2)でも確認してください。標準DINOv2 ViT-S/14についてはDINOv3のようなアクセス申請はありません。

## 3. DINOv3の準備

[DINOv3公式repository](https://github.com/facebookresearch/dinov3)のweight accessから申請し、会社の利用条件と専用ライセンスを確認してください。承認メールの一時URL、token、認証情報は設定やGitへ記録しません。

```powershell
git clone https://github.com/facebookresearch/dinov3.git external/dinov3
curl.exe -L "承認メールに記載されたViT-S16のURL" -o "checkpoints/dinov3_vits16_pretrain_lvd1689m.pth"
```

DINOv3が未準備でもDINOv2実験は実行できます。ランナーはDINOv3だけを`skipped_prerequisite`として記録します。

## 4. 配置・再現情報・Git除外の確認

```powershell
python scripts/check_foundation_setup.py
python scripts/record_foundation_lock.py
git status --short
```

DINOv2が`READY`になることを確認します。lock作成コマンドはrepository commitとweight SHA-256を`configs/foundation_models.lock.yaml`へ記録します。秘密情報や重み本体は記録しません。

`git status`に会社データ、`.pth`、`external/`、cacheが出ていないことも確認してください。

## 5. 実験計画

標準条件は`configs/analyses/foundation_probe.yaml`で変更できます。

```powershell
python scripts/plan_foundation_probes.py
Get-Content results\foundation_probe\plan\plan_summary.json
```

標準では合計25 runsです。

- DINOv2 pretrained：2 heads × 5 folds = 10
- DINOv2 random-init：Linear × 5 folds = 5
- DINOv3 pretrained：2 heads × 5 folds = 10

## 6. 最初のスモークテスト

DINOv2 pretrained、Linear、fold 1、1 epochをonline modeで確認します。

```powershell
python scripts/run_foundation_probes.py --mode smoke --teachers dinov2_vits14_pretrained --heads linear --folds 1
Get-Content results\foundation_probe\run_status_smoke.csv
Get-ChildItem results\foundation_probe\smoke -Recurse -Filter summary.json
```

`predictions/best`、`worst`、`random`の画像でInput、Ground truth、Prediction、Error mapを確認し、前景と背景が逆転していないことも確認します。

## 7. 特徴cacheとDINOv2本実験

本実験ランナーは対応cacheがなければ最初のrun前に全画像分を一度だけ生成します。明示的に先に生成する場合は次を使います。

cacheはhead学習の高速化にだけ使います。最終評価はDINO backboneを含むonline推論で実行されるため、保存される推論時間はBaselineと同様にend-to-endです。

```powershell
python scripts/cache_foundation_features.py --config results\foundation_probe\generated\configs\dinov2_vits14_pretrained__linear__fold_01.yaml
```

DINOv2の全15 runsを実行します。

```powershell
python scripts/run_foundation_probes.py --mode full --teachers dinov2_vits14_pretrained dinov2_vits14_random
Get-Content results\foundation_probe\run_status_full.csv
```

中断後は同じコマンドを再実行してください。`summary.json`まで完成したrunは自動skipされます。やり直す場合だけ`--rerun`を付けます。

## 8. DINOv3追加実験

DINOv3のrepositoryと重みを配置して`READY`を確認後に実行します。

```powershell
python scripts/run_foundation_probes.py --mode smoke --teachers dinov3_vits16_pretrained --heads linear --folds 1
python scripts/run_foundation_probes.py --mode full --teachers dinov3_vits16_pretrained
```

## 9. 集計とBaseline比較

```powershell
python scripts/analyze_foundation_probes.py
```

既存Baselineのrun表が別の場所にある場合は指定します。Baseline結果がなくてもDINO単独集計は成功します。

```powershell
python scripts/analyze_foundation_probes.py --baseline-runs "D:\results\machine_baseline_comparison\full_runs.csv"
```

主な出力は`results/foundation_probe/analysis/`内の`runs.csv`、`summary_by_condition.csv`、`pretrained_vs_random.csv`、`paired_baseline_deltas.csv`、機台別IoUヒートマップです。

## 10. DINO特徴分布

cache生成後、対象teacherの生成済みconfigを指定します。

```powershell
python scripts/analyze_foundation_features.py --config results\foundation_probe\generated\configs\dinov2_vits14_pretrained__linear__fold_01.yaml
```

機台別PCA、machine silhouette、前景・背景特徴距離、foldごとのseen-to-unseen重心距離が`results/foundation_probe/analysis/features/`へ保存されます。

## 11. 条件変更と注意点

`configs/analyses/foundation_probe.yaml`でfold、head、epoch、batch size、repository、weight、cache出力先を変更できます。GPUメモリ不足時は`training.batch_size`を1へ下げ、必要なら`gradient_accumulation`を増やします。

このフェーズではaugmentationを必ず`none`にします。DINO fine-tuning、Knowledge Distillation、KD＋augmentationはprobe評価後の別フェーズです。
