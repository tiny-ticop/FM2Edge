# Phase5.5：ImageNet事前学習＋DINO蒸留（業務PC手順）

## 目的と最初に確認すること

同じ業務データ・機台別foldで「事前学習の効果」と「事前学習後のKD追加効果」を切り分けます。
対象はMobileNetV3-Large＋LR-ASPP、PIDNet-Sです。PP-LiteSegは対象外です。

| 初期化 | GTのみ | GT＋KD（MGD / HeteroAKD応用 / GKD-CNN応用） |
| --- | --- | --- |
| Random | 既存Phase5の保存結果 | 既存Phase5の保存結果 |
| ImageNet | 今回の新規実験 | 今回の新規実験（DINOv2 / v3） |

標準は2生徒×（GTのみ1条件＋KD6条件）×5fold×1seed＝**70run**。
Teacher probeが必要な場合、その学習はこのStudent run数には含みません。
既存モデル・KD損失・ランナー・過去設定は変更していません。結果は別ディレクトリへ保存します。

## 1. pull・環境確認

リポジトリのルートで実行してください。既存のGPU用PyTorch環境をそのまま使います。

```powershell
git pull --ff-only
python -m pip install -e ".[analysis,dev]"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

torchvisionは実験実行に不要です。公式重みのキー変換はcore PyTorchだけで行います。
会社データ・重み・変換済み重み・cache・結果は、それぞれ既存のgitignore対象の
`data/`、`checkpoints/`、`results/`へ置いてください。
`git status --short`で会社画像や重みが表示されないことを確認します。

## 2. 公式ImageNet重みの取得

### MobileNetV3-Large

[Torchvision公式モデル説明](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html)
の **IMAGENET1K_V2** を使用します。LR-ASPPの領域学習済み重みではありません。

```powershell
New-Item -ItemType Directory -Force checkpoints/imagenet
Invoke-WebRequest -Uri https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth -OutFile checkpoints/imagenet/mobilenet_v3_large-5c1a4163.pth
```

### PIDNet-S

[公式README](https://github.com/XuJiacong/PIDNet)の現行配布フォルダを開き、
**ImageNet / PIDNet-S**を手動ダウンロードします。旧個別リンクの失効に注意してください。
Cityscapes/CamVid重み、PIDNet-M/L重みは今回使いません。
取得ファイルを `checkpoints/imagenet/PIDNet_S_ImageNet.pth.tar` として配置してください。
公式ファイル名が異なる場合は、以降の`--source`に実際の名前を指定してください。
使用条件・ライセンスは公式配布案内を確認してください。

## 3. 変換と読込監査

```powershell
python scripts/convert_student_imagenet.py --student mobilenet_v3_lraspp --source checkpoints/imagenet/mobilenet_v3_large-5c1a4163.pth --output checkpoints/imagenet/mobilenet_v3_large_fm2edge.pt
python scripts/convert_student_imagenet.py --student pidnet_s --source checkpoints/imagenet/PIDNet_S_ImageNet.pth.tar --output checkpoints/imagenet/pidnet_s_fm2edge.pt
```

それぞれ`.pt`と同名の`.json`監査レポートが生成されます。既存ファイルは上書きしません。
再変換する場合は別名を指定し、実験設定の`student_weights`も変更してください。
元重みを事前に固定したSHA-256で検証する場合は `--source-sha256 <64桁のハッシュ>` を付けます。
MobileNetは公式V2重みのhashも確認します。PIDNetのSHAは取得した公式ファイルを記録するもので、
ファイル名だけで配布元の真正性を保証するものではありません。

監査の確認項目：

- `source_id`：意図したモデルとImageNet重みであること。
- `feature_parameter_coverage`：MobileNetの特徴抽出部は1.0を要求します。
- `loaded_keys` / `mapping`：Conv、BN、SEも読み込まれていること。
- `head_keys_not_loaded`：既存領域ヘッドは全てランダム初期化のままです。
- `missing_feature_modules`：PIDNetは分類事前学習で省かれたタスク専用モジュールがあり得ます。
  省略されたモジュール全体を明示し、Shared I-branch（conv1、layer1〜5）の欠落、
  取得したモジュール内の読込漏れ、未知のキー、形状・dtype不一致はエラーにします。
  これは「モデル全体がImageNet学習済み」を意味しません。読込率も併記して評価してください。

旧ローダーの「ヘッドなどN tensors remain initialized」警告は意図したものです。
新しい監査を通過していれば、単にランダム初期化で学習してしまう事態とは区別できます。

## 4. 業務Phase5の条件を引き継ぐ（推奨）

編集元は `configs/analyses/pretrained_distillation.yaml` です。変更をGitへpushする必要はありません。
ローカル設定を`results/phase5_5_local.yaml`へコピーして使っても構いません。

Phase5でepoch、学習率、KD係数などを変更していた場合、**過去のexperiment_plan.csvを指定**してください。
過去planが指す各run設定をコピーし、Student初期重みと出力先だけを変更します。
データ、split、正規化、学習率、epoch、KD係数、Teacher設定、probe指定も維持します。
Teacher probeは既存のものを再利用でき、provenance確認は既存trainerが行います。
元run設定へのパスが存在することを確認してください。

```powershell
python scripts/run_pretrained_distillation.py --mode plan --phase5-plan results/knowledge_distillation/plan/experiment_plan.csv
```

標準設定と同じ場合だけ、`--phase5-plan`を省略できます。

```powershell
python scripts/run_pretrained_distillation.py --mode plan
```

`results/phase5_5_pretrained_distillation/plan/summary.json`のrunsを確認します。
全条件が揃っていれば70です。過去planを絞っていた場合は、その条件の分だけ生成します。
今回の設定の`students / folds / seeds / methods / teachers`が選択範囲となります。
`--phase5-plan`を指定した場合、今回の`training`値で過去の条件を上書きしません。
Phase5.5自体で条件を変更したい場合は、別出力先の設定で実行し、条件差を伴う比較とします。

標準は416×640、50epoch、early stopping 10、augmentationなし、cached feature。
現在のPhase5 trainer自体がaugmentationなし専用なので、非対応augmentationは拒否します。
未知test機台はTeacher probe学習・Student学習・best epoch選択・調整に使いません。

## 5. 1foldのonlineスモークテスト

まずDINOv2で、両生徒のGTのみ＋3手法を確認します。

```powershell
python scripts/run_pretrained_distillation.py --mode smoke --phase5-plan results/knowledge_distillation/plan/experiment_plan.csv --teachers dinov2 --folds 1
```

smokeは1epoch、GKDだけ各段階1epoch、featureはonlineです。`smoke/`へ別保存します。
`--teachers dinov2`でもGTのみ条件は実行されます。
必要なら `--students mobilenet_v3_lraspp` で、生徒1種類だけ確認できます。
HeteroAKD smokeではwarmupを0にして蒸留経路を確認します。
既存Teacher probeを指定した場合、そのprobeの学習済み重みを利用します。
DINOv3も同じ操作で確認できます。

```powershell
python scripts/run_pretrained_distillation.py --mode smoke --phase5-plan results/knowledge_distillation/plan/experiment_plan.csv --teachers dinov3 --methods mgd heteroakd gkd_cnn_source_only --folds 1
```

重み・repository不足は`run_status.csv`へ`skipped_prerequisite`として記録します。
エラーを無視して成功扱いにはしません。

## 6. 5foldの本実験

```powershell
python scripts/run_pretrained_distillation.py --mode full --phase5-plan results/knowledge_distillation/plan/experiment_plan.csv
```

標準は70runです。Teacher cache・probeがなければ必要な分を生成します。
条件が一致する既存cacheは再利用します。GTのみrunはDINOなしで実行できます。
モデル・教師・手法・fold・seedはフィルタ可能です。

```powershell
python scripts/run_pretrained_distillation.py --mode full --phase5-plan results/knowledge_distillation/plan/experiment_plan.csv --students mobilenet_v3_lraspp --teachers dinov3 --methods mgd --folds 1 2 3 4 5
```

## 7. 中断・再開

同じコマンドを再実行すると、完了済みrunをスキップし、途中runをepoch境界から再開します。
重み・GT・設定が変わった既存runへの再開は拒否されます。条件変更時は別output_dirを使ってください。
`--rerun`は同じ条件の再学習で、そのrunの成果物を上書きします。必要な結果を先にコピーしてください。
`--train-only`は学習まで実行し、test評価は後の通常再実行に回します。
Teacher probeの途中学習は旧仕様のままで、Studentのようなepoch再開はありません。

## 8. 統合分析

Phase5のrootは、**同一の実験セットだけ**を指定してください。
複数の候補が同条件で見つかった場合、都合のよい結果を選ばず`ambiguous_control`で除外します。

```powershell
python scripts/analyze_pretrained_distillation.py --plan results/phase5_5_pretrained_distillation/plan/experiment_plan.csv --phase5-root results/knowledge_distillation --prediction-samples 3
```

過去結果がなくても単独集計できます。

```powershell
python scripts/analyze_pretrained_distillation.py --plan results/phase5_5_pretrained_distillation/plan/experiment_plan.csv
```

出力は `phase5_5_analysis/` です。

| 保存物 | 内容 |
| --- | --- |
| runs.csv / summary_by_condition.csv | 初期化・生徒・教師・手法別の値、平均、標準偏差、実際のrun数 |
| paired_deltas.csv / summary_paired_deltas.csv | fold/seed対応付きの事前学習効果・KD追加効果・交互作用 |
| comparison_status.csv | 条件不一致、比較先なし、未完了、比較先の曖昧さ |
| *_four_conditions.png | Random/ImageNet × GT/GT＋KDの条件別IoU（記述的比較） |
| machine_heatmap.csv / .png | 機台×条件のIoU |
| same_sample_predictions/ | 任意指定時、同じtest sampleの入力・GT・予測・誤差を縦に並べた図とindex.csv |

対応付き比較ではmanifest、実際の画像/GT内容hash、fold分割、学習画像subset、入力・正規化、
学習条件、augmentation、実装hashを確認します。初期化だけを変更した比較ではそれを除外し、
同一初期化のKD追加比較ではStudent重みhashも確認します。教師間の比較ではなく同じ教師・手法で
初期化を比較する場合、Teacher重み・repository commit・probe・KD係数も確認します。
過去データ内容の証明がないrunは、記述集計には表示しますが厳密な対応付き比較には含めません。
欠落したforeground指標などは空欄にし、存在する実測値だけを使います。

`--prediction-samples 3`は各foldのtest sample IDをソートして先頭3件を選び、CPUで再推論します。
性能を見てsampleを選びません。元データとcheckpointがローカルに必要です。
存在しない場合、統計集計は成功し、画像生成の不足はindex.csvへ記録します。

GKDの`budget_different`は、GTのみとの間にGT学習期間・凍結・2段階スケジュールの違いがある意味です。
改善は「2段階学習を含む追加効果」であり、KD損失だけの寄与とは断定しません。
foldは5つで学習セットが重なるため、標準偏差を独立試行の信頼区間とは解釈しません。
約0.86など過去の概算値を入力する必要はありません。保存された指標を使用します。

## 9. 結果保存とローカル検証

`results/phase5_5_pretrained_distillation/`全体と、使用したローカル設定・変換監査JSON・重みhashを
会社の承認済み保存先へ保存してください。GitHubへ会社データや学習結果をpushしないでください。
推論モデルは既存構造のStudentのみで、パラメータ数を増やしません。

合成データは既存の`create_synthetic_dataset.py` / `prepare_dataset.py` / `make_splits.py`で準備できます。
合成データとsplitが既にある環境では以下で確認できます。

```powershell
python scripts/run_pretrained_distillation.py --config configs/analyses/synthetic_pretrained_distillation_smoke.yaml --mode smoke --teachers dinov2 --students mobilenet_v3_lraspp
python -m pytest
python -m ruff check .
```

この合成実験は動作確認です。未知機台で性能が改善したことの証拠にはなりません。
