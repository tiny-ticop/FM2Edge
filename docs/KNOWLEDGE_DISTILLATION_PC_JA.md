# DINO知識蒸留：業務PCでの実行手順

この手順は既存の会社データmanifest・機台単位splitを使用します。過去のBaseline、要因分析、Frozen DINO probeのコマンドは変更していません。

## 1. 更新・環境確認

```powershell
git pull --ff-only
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,tensorboard,analysis]"
python scripts/check_foundation_setup.py
pytest -q
```

DINOv2/v3のrepository・重みの準備は[既存の手順書](FOUNDATION_PROBE_PC_JA.md)を参照してください。DINOv3の未準備環境では該当runだけ`skipped_prerequisite`となります。業務PCでは両方がREADYであることを確認します。

## 2. データ・設定を確認

`configs/analyses/knowledge_distillation.yaml`を確認します。

- `data.root`、`manifest`、`splits_dir`：前回と同じもの。
- `image_size`：高さ416×幅640。
- `students`：初期値はPIDNet-SとMobileNetV3＋LR-ASPP。
- `teachers`：DINOv2/v3のローカルrepositoryとweight。
- `folds`、`seeds`：変更可能。既定は5-fold、seed42。
- `training`：既定50 epochs、early stopping 10、batch4。
- `methods`：`none`、`mgd`、`heteroakd`、`gkd_cnn_source_only`。

機台数を変える場合は既存split生成処理でtrain/val/testが分離されたsplitを準備し、この設定から参照します。画像数subsetはsplitの`train_sample_ids`を再利用します。Teacher probeにも同じsubsetが適用されます。

PP-LiteSegは`students`へ追加できますが、GKD型は未対応なのでその組合せは計画に生成しません。過去のPP-LiteSeg Baselineはそのまま実行できます。

## 3. 実験計画

```powershell
python scripts/plan_distillation_study.py
Get-Content results\knowledge_distillation\plan\summary.json
```

既定ではKD60 runs＋蒸留なし10 runs＝70 runsです。GKDは各run内に2段階があります。HeteroAKDのfold別Teacher probe学習は別費用で、同じTeacher/fold/seed間で共有します。

この時点では学習・test評価を実行しません。

## 4. まず1 fold・短時間で確認

```powershell
python scripts/run_distillation_study.py --mode smoke --teachers dinov2 --students mobilenet_v3_lraspp --methods mgd --folds 1
python scripts/run_distillation_study.py --mode smoke --teachers dinov3 --students mobilenet_v3_lraspp --methods mgd --folds 1
python scripts/run_distillation_study.py --mode smoke --teachers dinov2 dinov3 --students pidnet_s mobilenet_v3_lraspp --methods heteroakd gkd_cnn_source_only --folds 1
```

smokeはonline Teacher、1 epochです。GKDは表現1 epoch＋タスク1 epochになります。HeteroAKDのprobeがなければtrain/valだけで1 epoch学習して準備します。smokeではwarmupを0にして蒸留枝の動作を確認します。

smokeでも既定は最終test機台を評価するため、厳格にtestを温存する場合は`--train-only`を付けます。学習・val・exportまで確認し、test結果を使った条件調整を避けてください。

```powershell
python scripts/run_distillation_study.py --mode smoke --teachers dinov2 --methods mgd --folds 1 --train-only
```

## 5. 本学習

まず蒸留なしを同じ新ランナーで実行すると、学習条件が揃う比較結果を用意できます。

```powershell
python scripts/run_distillation_study.py --mode full --methods none
python scripts/run_distillation_study.py --mode full --teachers dinov2 --methods mgd heteroakd gkd_cnn_source_only
python scripts/run_distillation_study.py --mode full --teachers dinov3 --methods mgd heteroakd gkd_cnn_source_only
```

fullはcached Teacher特徴を利用します。KDはtrain sampleのみのcacheを必要に応じて追加生成します。HeteroAKD probe準備ではtrain/valのcacheを生成します。testは蒸留に使用しません。

GPUメモリ不足時はbatchを2まで下げられます。ただし厳密な比較には蒸留なし条件も同じbatchに揃えてください。勾配蓄積はBatchNormのbatchサイズを代替しません。

## 6. 過去のDINO probeを再利用

Teacher設定に以下のようなpatternを追加できます。

```yaml
probe_config_pattern: results/foundation_probe/generated/configs/dinov2_vits14_pretrained__linear__fold_{fold:02d}.yaml
probe_checkpoint_pattern: results/foundation_probe/runs/dinov2_vits14_pretrained__linear__fold_{fold:02d}/fold_{fold:02d}/checkpoints/best.pt
```

probeの保存済みconfig・split・manifest、前処理、Teacher識別情報を検査します。不一致時は停止します。別subsetのprobeや全機台で学習したprobeは使えません。

過去probeの画像・maskが学習後に変更されていないことも確認してください。旧probeはファイル内容の学習時hashを保存していないため、snapshotだけではその変更を証明できません。不明なら今回の専用probeを再学習します。

## 7. 中断・再開

同じコマンドを再実行すると、完成済みrunをskipし、未完了runは最後のepoch境界から再開します。

復元対象はStudent、蒸留adapter、optimizer、scheduler、AMP、乱数、DataLoader順序、early stopping、GKD段階です。epoch途中の中断はそのepochをやり直します。Teacher probeの準備途中では既存probe trainerを使うため、probe学習は最初から再実行します。

同条件を最初から再実行する場合だけ`--rerun`を付けます。これは当該KD結果を更新するため、旧結果を残したい場合は先にバックアップするか、新しい`output_dir`を指定してください。

条件・データ・重みを変更した場合、既存runのidentity不一致で停止します。別`output_dir`を指定して新しい実験として実行してください。

## 8. 結果確認・Baseline比較

```powershell
python scripts/analyze_distillation_study.py
python scripts/analyze_distillation_study.py --baseline-root "D:\FM2Edge_baseline_results"
```

`--baseline-root`は各foldの`config.yaml`・`metrics/summary.json`を含む結果ディレクトリです。学習時に参照したmanifest・split・初期重みも読み取り可能である必要があります。古いPCの絶対パスのままなら移設して指定を整えてください。

集計先：`results/knowledge_distillation/analysis/`

- `runs.csv`、`summary_by_condition.csv`
- `Foreground_IoU`、`Foreground_Recall`も保存。foregroundが存在しない機台などで指標が未定義の場合はnullとなります。
- `paired_baseline_deltas.csv`：条件一致時のIoU差分。GKDは予算差を明示。
- `per_machine.csv`、`machine_heatmap.csv`、`machine_heatmap.png`

一致するBaselineがなければ`no_matching_baseline`として記録し、KD単独集計は継続します。古い結果のmask内容変更は元の保存情報だけでは検証できないため、データを不変にして比較してください。

各run内には以下を保存します。

```text
identity.json / config.yaml / split.yaml / manifest.csv / environment.json
checkpoints/last.pt    学習再開用（蒸留枝を含む。Teacher本体は含まない）
checkpoints/best.pt    val選択済みStudentのみ
student.pt            配布・推論用Studentのみ
history/epochs.csv / training_summary.json
metrics/summary.json / per_image.csv / per_machine.csv / per_delay.csv
predictions/best / worst / random
completed.json
```

学習時パラメータと推論時Studentパラメータを分けて記録します。Student-only評価を新しいモデルinstanceで行い、Teacher/adapterをGPUメモリ計測に含めません。CPU-only製品の実際のlatency/RAMは対象PC/SBCで別途測定してください。

## 9. 手法の解釈・注意

[実装差分](KD_METHODS.md)と[文献調査](KNOWLEDGE_DISTILLATION_SURVEY_JA.md)を参照してください。HeteroAKDはDINO probe適用版、GKDはproxy/CLS/masked-image損失なしのCNN source-only版で、原論文の完全再現ではありません。

GKDは既定で表現25 epochs＋タスク25 epochsです。PIDNetは融合までの特徴抽出部を凍結し、既存segmentation headsを学習します。decoderへの適応制限で悪化する可能性もあり、結果を見て有効性を判断します。

会社データ、DINO重み、cache、結果、認証情報はGitへ追加しません。過去フェーズは[Baseline手順](MACHINE_BASELINE_PC.md)、[要因分析手順](MACHINE_GENERALIZATION_ANALYSIS_PC_JA.md)、[DINO probe手順](FOUNDATION_PROBE_PC_JA.md)から従来どおり実行できます。
