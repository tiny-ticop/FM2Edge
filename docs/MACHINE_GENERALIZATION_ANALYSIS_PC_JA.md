# 実機汎化要因分析：業務GPU PC実行手順

この手順は、小型3モデルの5-fold baseline完了後に、未知機台性能へ影響する
要因を分析するためのものです。従来のbaselineコードと結果はそのまま残ります。

## 1. clone後の環境準備

PowerShellでリポジトリへ移動し、CUDA版PyTorchが入った仮想環境を有効化します。

```powershell
cd FM2Edge
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,tensorboard,analysis]"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')"
pytest
```

本実験前に`True`とGPU名が表示されることを確認してください。

## 2. データと分割の確認

前回と同じ以下のファイルを使用します。

```text
data/raw/<機台名>/delay/<条件>/<画像>.jpg
data/masks/<機台名>/delay/<条件>/<マスク>.png
data/company/manifest.csv
configs/splits/company/fold_01.yaml ～ fold_05.yaml
```

新しいcloneでは、承認済み保存場所から`data/`をコピーするか、
`docs/MACHINE_BASELINE_PC.md`に従ってmanifestとsplitを再生成してください。

## 3. 実験条件を変更する場所

次のYAMLだけを編集します。

```text
configs/analyses/machine_generalization.yaml
```

主な変更箇所：

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

- 機台を増減した場合は、先にsplitを再生成します。
- `machine_counts`は、各foldで利用可能なtrain候補数以下にします。
- 組み合わせが多すぎる場合は`max_combinations_per_count`へ上限を指定します。
- 枚数不足や不正な機台数は自動変更せず、エラーで停止します。
- 条件変更時はrun fingerprintも変わるため、古い結果を誤って再利用しません。

## 4. 実験計画を生成・確認

```powershell
python scripts/plan_generalization_study.py
Get-Content results\generalization_analysis\plan\plan_summary.json
```

標準設定は240 runsです。開始前に以下を確認してください。

```text
results/generalization_analysis/plan/plan_summary.json
results/generalization_analysis/plan/experiment_plan.csv
```

CSVには、各runのモデル、fold、train／validation／test機台、使用画像ID、
augmentation、seed、出力先が固定保存されます。

## 5. 3モデルのスモークテスト

```powershell
python scripts/run_generalization_study.py --smoke
```

各モデルをfold 1・2 epochで実行します。次の予測画像を確認してください。

```text
results/generalization_analysis/smoke/<run_id>/fold_01/predictions/
```

入力、GT、予測、誤差マップの位置・向き・ラベルが正しいことを確認します。

## 6. suite単位で本実験

一度に全部実行せず、次の順番を推奨します。

```powershell
python scripts/run_generalization_study.py --suite reference
python scripts/run_generalization_study.py --suite machine_diversity
python scripts/run_generalization_study.py --suite images_per_machine
python scripts/run_generalization_study.py --suite augmentation
```

途中停止した場合は、同じコマンドを再実行してください。完了済みrunはスキップされ、
未完了runだけ再実行されます。通常は`--rerun`を付けません。

進捗とログ：

```text
results/generalization_analysis/run_status.csv
results/generalization_analysis/runs/<run_id>/fold_<NN>/run.log
```

一部だけ実行する例：

```powershell
python scripts/run_generalization_study.py --suite augmentation --models pidnet_s
python scripts/run_generalization_study.py --suite machine_diversity --folds 1 2
python scripts/run_generalization_study.py --suite images_per_machine --limit 3
```

## 7. 集計・図表作成

完了しているrunだけで、いつでも再集計できます。

```powershell
python scripts/analyze_generalization_study.py
```

主な出力：

```text
results/generalization_analysis/analysis/report.md
results/generalization_analysis/analysis/tables/aggregate_results.csv
results/generalization_analysis/analysis/tables/paired_effects.csv
results/generalization_analysis/analysis/figures/
results/generalization_analysis/analysis/domain_statistics/
```

以下が自動生成されます。

- 学習機台数と未知機台IoUの関係
- 1機台あたり画像数とIoUの関係
- augmentationごとのbaseline差分
- 学習機台subset×未知機台のヒートマップ
- 機台ごとの輝度、色、コントラスト、鮮鋭度、前景率
- 入力ドメイン距離と未知機台IoUの関係
- 全背景／全前景へ崩壊した画像の割合
- 未知機台単位bootstrap 95%信頼区間

## 8. モデル特徴分布

基準モデルの特徴PCA：

```powershell
python scripts/extract_generalization_features.py
```

combined augmentationとの比較：

```powershell
python scripts/extract_generalization_features.py --include-combined
```

出力先：

```text
results/generalization_analysis/analysis/features/
```

機台色分けPCA、global／foreground／background特徴、machine silhouette、
train-test特徴重心距離、特徴距離とIoUの散布図が保存されます。

## 9. 追加seed確認

全条件を最初から3 seedsで回さず、seed 42で要因をスクリーニングします。
有効な戦略を選んだ後、YAMLを次のように変更します。

```yaml
seeds: [42, 43, 44]
```

不要なsuiteは`enabled: false`にして、基準条件と最終候補だけ再確認します。

## 10. 保存するもの

実験終了後、以下をフォルダごと承認済みストレージへコピーしてください。

```text
results/generalization_analysis/
```

特に共有するもの：

```text
analysis/report.md
analysis/tables/aggregate_results.csv
analysis/tables/paired_effects.csv
analysis/figures/
analysis/features/feature_summary.csv
run_status.csv
```
