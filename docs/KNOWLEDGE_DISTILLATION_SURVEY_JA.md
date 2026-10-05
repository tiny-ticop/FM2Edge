# 未知機台への汎化を目的とした知識蒸留の調査・手法選定

調査日：2026-10-05

対象：FM2Edge／産業機械の固定カメラ画像／小型Semantic Segmentationモデル

状態：文献調査と実験提案。蒸留コードの実装・実機データでの効果検証は未実施。

## 1. 結論

今回の目的はモデル圧縮そのものではなく、**DINOが持つ視覚表現を小型CNNへ移し、学習していない機台でのセグメンテーション性能を改善すること**である。

当初のCosine／normalized MSEを必ず採用する制約は置かず、次の3手法を選定する。

| 選定 | 手法 | 本テーマで確かめたいこと | 採用上の条件 |
| --- | --- | --- | --- |
| 1 | **HeteroAKD**：異種アーキテクチャ間の、信頼性に応じた知識蒸留 | DINOとCNNの表現差を避け、Teacherの有用な判断をStudentへ移せるか | foldごとに学習したDINO segmentation probeが必要。未知機台での改善は未保証 |
| 2 | **MGD（Masked Generative Distillation）**：マスクしたStudent特徴からTeacher特徴を復元 | 少数の似た画像への局所的な依存を抑え、形状・文脈の表現を改善できるか | Frozen DINOの特徴だけで開始可能。DINO→CNNのDG効果は新たに検証する |
| 3 | **GKD型分離学習のCNN適用版**：表現蒸留とタスク学習を分離 | GTへの適応によってTeacher由来の汎用表現を失うことを抑えられるか | CNNへの移植は研究的改変。原論文GKDの再現・実証済み手法とは呼ばない |

3手法は、それぞれ「異種構造」「特徴学習の正則化」「汎化を保持する学習手順」という異なる問題を検証する組合せである。文献上の総合順位ではなく、本プロジェクトにおける検証価値を踏まえた選定である。

**実験実装の順番はMGD → HeteroAKD → GKD型CNN版を推奨する。** MGDは既存の特徴キャッシュを利用しやすい。HeteroAKDはTeacher probeの準備が必要。GKD型CNN版は凍結境界と学習予算を別途設計する必要がある。

重要な留保：3手法すべてが本データで改善するという証拠はない。とくに「同一データセット内での改善」「未知ドメインでの改善」「DINO→小型CNNでの改善」は別々の証拠として扱う。

## 2. 目的と評価対象を明確にする

### 2.1 成功条件

主たる問いは次のとおり。

> 同じ学習機台・GT枚数・Student構造を用いたとき、DINOから蒸留したStudentは、非蒸留Studentより未知機台への汎化が改善するか。その改善はCPU-only製品への搭載を妨げないか。

主指標は未知機台の機台平均mIoUと、最悪機台のmIoU。併せてforeground IoU、Dice、Recall、Boundary F1を確認する。平均値だけ改善して最悪機台や細い領域のRecallが悪化した場合は、無条件に成功としない。

製品側ではStudentのみで推論し、DINO、蒸留専用projector、generator、Teacherとのattention計算は不要とする。CPU latency、モデル容量、peak RAMを評価し、学習時のGPU速度と混同しない。

### 2.2 確認できている条件と未確認事項

| 項目 | 現在の前提 | 選定・設計への影響 |
| --- | --- | --- |
| 撮像 | 産業機械の固定カメラ。機台内では固定背景・周期運動により画像が似る | 枚数を増やしても独立した外観の種類が増えるとは限らない |
| Domain | machine_id。未知機台を最終評価に使う | 機台単位のtrain／val／test完全分離が必須 |
| 現在のGT規模 | 約5機台、1機台50〜100枚程度というユーザー報告 | 大規模proxyデータを前提とする手法を、そのまま少量データへ当てはめない |
| 元画像 | 横1000×縦650 pixel | リサイズによる細部消失とKDの効果を切り分ける |
| 入力 | 高さ416×幅640 | Teacher／Studentの座標対応を維持する |
| ラベル | 二値foreground／background。現行形式は0／1、ignoreは255 | 背景だけ当てる崩壊をforeground指標で検出する |
| Teacher | まずFrozen DINOv2 ViT-S/14。DINOv3は後続比較 | raw DINO特徴にはタスクのクラスlogitsがない |
| Student | PIDNet-S、MobileNetV3＋LR-ASPPを優先候補。PP-LiteSegの既存Baselineも維持 | 手法が特定decoderに依存しないか確認する。最終対象は実装計画で固定する |
| 境界・細線 | 初期要件にBoundary F1、将来的なSkeleton／Centerline／Connectivity評価が含まれる | Boundaryを重要視するが、実際の対象の幅や形状は未確認 |
| 未確認 | 実画像、foregroundの具体的対象、反射・露光差・遮蔽の程度、細線幅、位相別枚数 | 「金属反射が主因」「細線が主対象」などと断定しない |

実画像を見ていないため、文献から本データでの改善幅を予測することはできない。対象幅、foreground面積率、機台間の背景差は、業務環境の学習データで確認してから重み付けを決める。

### 2.3 検証したい失敗要因

以下は本データで確定した原因ではなく、検証仮説である。

1. **機台固有背景への依存**：同じ背景を繰り返し見ることで対象ではなく背景を覚える。
2. **外観変動への弱さ**：機台間の色・明るさ・配置・撮像条件の違いに追従できない。
3. **ViT→CNNの表現差**：Teacherの特徴値を強制的にコピーしてStudentの表現を損なう。
4. **タスク適応による汎用表現の喪失**：少量GTへの最適化がTeacher由来の特徴と競合する。
5. **細部の不足**：Teacherのpatch特徴自体に必要な境界情報がない。

KDは1〜4への候補となる一方、5を自動的には解決しない。Teacherが持っていない細部はStudentのGT監督や高解像度経路で学習する必要がある。

## 3. 調査範囲と証拠の読み方

分類用KDの基礎、セグメンテーション用KD、Transformer↔CNNの異種構造蒸留、DINO／基盤モデル蒸留、未知ドメイン評価、境界・周波数・少量ラベル条件を横断して調査した。

主に会議公式論文、arXivの著者原稿、著者の公開repositoryを確認した。検索語の例は以下。

- `DINO knowledge distillation lightweight CNN semantic segmentation domain generalization`
- `heterogeneous architecture knowledge distillation semantic segmentation`
- `foundation model distillation cross dataset segmentation limited labels`
- `boundary privileged knowledge distillation semantic segmentation`
- `masked generative distillation`、`frequency knowledge distillation`

本調査は関連する主要候補の比較であり、世界中の全論文を網羅したsystematic reviewではない。2026-10-05時点で確認できる資料を対象とし、本文が取得できない資料は要旨までの確認として明記する。

**以下の表の「適合性」「採否」は本プロジェクトに対する判断・推論であり、論文著者が実機データで証明した結果ではない。** 「generalization」という表現があっても、別データセットへの評価なのか、通常のvalidation評価なのかを区別する。

## 4. 文献・候補手法の比較

| 手法／文献 | 年・発表先 | 移す知識／主な検証 | 本テーマへの適合性と注意点 | 今回の判断・一次資料 |
| --- | --- | --- | --- | --- |
| Hinton et al., *Distilling the Knowledge in a Neural Network* | 2015 arXiv／NIPS 2014 Workshop関連 | Teacherのsoft prediction | 原点となる低コスト対照。raw DINOには二値logitsがない | 主選定外。probe Teacherを使う場合の対照。[論文](https://arxiv.org/abs/1503.02531) |
| Romero et al., *FitNets: Hints for Thin Deep Nets* | 2015、ICLR | 中間特徴のhintと次元適合 | feature KDの基礎。Cosine／normalized MSEそのものをFitNetsの原手法と同一視しない | 主選定外。単純特徴KDはsanity check向け。[論文](https://arxiv.org/abs/1412.6550) |
| Liu et al., *Structured Knowledge Distillation for Semantic Segmentation*（SKD） | 2019、CVPR | pixel、pair-wise関係、holistic GAN。Cityscapes／CamVid／ADE20K | チャネル数が違っても関係を比較できる。全SKDは複数損失を含み、pair-wiseのみとは別 | 次点。関係だけ移すablationには有用。[公式論文](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.html) |
| Shu et al., *Channel-wise Knowledge Distillation for Dense Prediction*（CWD） | 2021、ICCV | チャネル内の空間分布をKLで比較。dense prediction | 低コストで強い標準手法。ただしDINOの特徴軸とCNNの特徴軸の対応は自明でない | 主選定外。異種構造での負の転移例もある。[論文](https://arxiv.org/abs/2011.13256) |
| Yang et al., *Cross-Image Relational Knowledge Distillation for Semantic Segmentation*（CIRKD） | 2022、CVPR | 画像を跨ぐpixel／region関係。Cityscapes／CamVid／VOC | 複数機台の共有構造を移す仮説に合う。ただし似た画像を大量にqueueへ入れても多様性は増えない | 次点。機台・位相を考慮したsampling／memory管理が追加課題。[公式論文](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Cross-Image_Relational_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2022_paper.html)・[著者コード](https://github.com/winycg/CIRKD) |
| Yang et al., *Masked Generative Distillation*（MGD） | 2022、ECCV | マスク特徴からTeacher特徴を復元。分類・検出・segmentation | Teacher logits不要、CNNへ小さい蒸留枝を付けられる。未知機台への効果は未実証 | **選定2**。[論文](https://arxiv.org/abs/2205.01529)・[著者コード](https://github.com/yzd-v/MGD) |
| Fan et al., *Augmentation-Free Dense Contrastive Knowledge Distillation for Efficient Semantic Segmentation*（Af-DCD） | 2023、NeurIPS | dense contrastiveによる局所的な構造移転 | 画像augmentationなしでも使える。一方、contrastive条件や異種構造での効果は要確認 | 次点。GKD・HeteroAKD内の比較も参考にした。[論文](https://arxiv.org/abs/2312.04168)・[著者コード](https://github.com/OSVAI/Af-DCD) |
| Liu et al., *BPKD: Boundary Privileged Knowledge Distillation for Semantic Segmentation* | 2024、WACV | body／edgeの蒸留を分離。3つのsegmentation benchmark | 境界・細線を重視する目的に合う。ただしTeacherの境界が不正確なら誤りを強調する | **境界が主なボトルネックなら最優先の次点**。[論文](https://arxiv.org/abs/2306.08075)・[著者コード](https://github.com/AkideLiu/BPKD) |
| Zhang et al., *FreeKD: Knowledge Distillation via Semantic Frequency Prompt* | 2024、CVPR | 周波数prompt、位置を考慮した関係損失。dense prediction、DINO／SAMへの検証 | 周波数の構造情報に着目できるが、prompt学習を含む。単純FFT-MSEでは原手法にならない | 次点。Frozen Teacher条件とpromptの学習範囲を確認してから採用。[公式論文](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_FreeKD_Knowledge_Distillation_via_Semantic_Frequency_Prompt_CVPR_2024_paper.html) |
| Zhang et al., *Accessing Vision Foundation Models via ImageNet-1K*（Proteus） | 2024 arXiv初稿 | DINOv2等の表現蒸留、ImageNet proxy利用 | 汎用特徴移転の参考。ただしデータ規模、Student構造、学習予算が現状と異なる | 初回主選定外。proxy追加時の参考。[論文](https://arxiv.org/abs/2407.10366) |
| Huang et al., *Distilling Knowledge from Heterogeneous Architectures for Semantic Segmentation*（HeteroAKD） | 2025、AAAI | 異種構造のlogits空間、知識混合・評価。Transformer→CNNを含む | 本件の構造差に直接対応。ただしTeacherはタスク予測可能である必要がある | **選定1**。[会議公式](https://ojs.aaai.org/index.php/AAAI/article/view/32399)・[著者原稿](https://arxiv.org/abs/2504.07691) |
| Agnihotri et al., *From SAM to DINOv2: Towards Distilling Foundation Models to Lightweight Baselines for Generalized Polyp Segmentation*（Polyp-DiFoM） | 2026、WACV | 複数基盤モデル、低／高周波、U-Net系、cross-dataset評価 | 二値segmentation・基盤モデル→CNNという点で近い。ただし複数Teacherとdecoderの変更が交絡する | 次点。DINO単独の損失だけで論文の効果を再現できるとは言えない。[著者原稿](https://arxiv.org/abs/2512.09307)・[著者コード](https://github.com/shivanshu-agnihotri/PolypDiFoM) |
| Lv et al., *Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation*（GKD） | 2026、CVPR（著者公開情報） | DINOv2、段階的蒸留とencoder凍結、DG benchmark | 目的に最も直接的。ただし公開論文の主要StudentはViTで、CNN版の実証ではない | **選定3の設計根拠**。[論文](https://arxiv.org/abs/2603.02554)・[著者コード](https://github.com/Younger-hua/GKD) |
| Nahian et al., *Domain-Constrained Distillation of DINOv3 into a Lightweight Foundation Model Toward Point-of-Care Ultrasound* | 2026、MIDL／PMLR | DINOv3→ResNet-50、大規模超音波データ、少量ラベル条件 | DINO→CNNの参考。ただし大規模ドメインデータと専用augmentationが必要 | 要旨・書誌を確認。本文取得できず、損失詳細・未知機台相当splitの根拠には使わない。[会議公式](https://proceedings.mlr.press/v315/nahian26a.html) |
| Liu & Sun, *DINO-Mix: Distilling Foundational Knowledge with Cross-Domain CutMix for Semi-supervised Class-imbalanced Medical Image Segmentation* | 2026 arXiv初稿 | Frozen DINOv3特徴、normalized MSE、3D医療SSL＋CutMix | DINO→CNNの単純特徴蒸留の参考。少数クラス問題を扱うが、未知ドメイン評価とは別 | 初回主選定外。3D・SSL・augmentationを同時に導入しない。[著者原稿](https://arxiv.org/abs/2602.07819) |

会議名を一次資料で確認できないものはarXiv初稿年として記載した。コードリンクは取得できた著者公開先だけを載せており、リンクの存在は再現性・保守性・商用利用許可の保証ではない。

## 5. 選定を変える重要な証拠

### 5.1 代表的なKDでも異種構造で悪化し得る

HeteroAKDのTable 1(a)では、CityscapesのTransformer Teacher（DeepLabV3-MiT-B4）→CNN Student（DeepLabV3-MBV2）で、Student単独73.92、CWD71.59、HeteroAKD74.91 mIoUと報告される。同じ表内でCWDは−2.33、HeteroAKDは＋0.99ポイントである。

これは本データでの順位を意味しないが、「segmentation KDで有名だからDINO→CNNにも安全」とは言えない証拠となる。[HeteroAKD本文 Table 1](https://arxiv.org/html/2504.07691v1)

### 5.2 同一ドメイン精度とDGは別物

GKDはDINOの未知ドメイン性能を移す目的を明示し、表現蒸留とタスク学習の分離を検証している。F2Lの主要例はDINOv2→DeiT／ViTであり、ここでいう「local」はCNNの意味ではない。[GKD本文](https://arxiv.org/html/2603.02554v1)

したがって、本件ではGKDの問題設定を重視する一方、CNN版の性能を実証済みとして扱わない。また、TeacherよりStudentが低いだけでは「KDで悪化」と言えない。必ず同じStudentの非蒸留条件との差で判定する。

### 5.3 大きな改善幅をそのまま本件へ転用しない

MGDはCityscapesのDeepLabV3-Res18で73.20→76.02 mIoU（＋2.82ポイント）を報告する。ただしTeacherはsegmentation CNNで、未知機台への評価でもDINO→CNNの評価でもない。[MGD本文](https://arxiv.org/pdf/2205.01529)

論文ごとの数値にはTeacher、初期重み、入力、学習回数、データが異なる。HeteroAKDの＋0.99とMGDの＋2.82を比較してMGDが優れるとは判断しない。

## 6. 選定1：HeteroAKD

### 6.1 原論文と本件での狙い

原論文は、異なる構造の中間特徴をクラスlogits空間へ写し、GTに基づくTeacher／Student知識の混合（KMM）と信頼性差の評価（KEM）を行う。Transformer↔CNNのsegmentationを検証している。[会議公式](https://ojs.aaai.org/index.php/AAAI/article/view/32399)・[本文](https://arxiv.org/html/2504.07691v1)

本件では、DINOとCNNの特徴軸を直接一致させるのではなく、タスクに必要な判断を共通空間で移す候補とする。Teacherが苦手な場所も盲目的に模倣しない設計は、構造の異なるStudentにとって重要と考える。これは本データに対する仮説である。

### 6.2 適用案

1. foldのtrain機台だけでFrozen DINO＋segmentation probeを学習する。
2. val機台のみでprobe checkpointを選び、DINO本体・probeを凍結する。
3. Studentの中間特徴に学習時専用のクラスprojectionを付ける。
4. train画像とGTだけを用い、原論文の知識混合・評価の処理を移植する。
5. Student本来のGT segmentation lossは維持する。
6. 推論ではTeacherと蒸留専用projectionを除去する。

既存probeの再利用は、train／val／test split、manifest、入力、重みが一致するときだけ許可する。全5機台で学習したprobeを各foldのTeacherに使うことは禁止する。

### 6.3 期待・リスク・判断条件

| 観点 | 判断 |
| --- | --- |
| 期待する効果 | 表現差による過剰制約を減らし、タスクに有用なTeacher知識を移す |
| 少量GT条件 | probeにも過学習の危険がある。初期実験では単純なheadを用い、head選択予算を固定する |
| 二値条件 | 多クラスよりsoft class情報が少ないため、原論文ほど有利とは限らない |
| DGの限界 | task logitsはsource機台へ適応した情報。DINOの汎用特徴をすべて保持するわけではない |
| 比較対象 | 同じprobeを使う単純logit KD。これがないと信頼性機構の価値を分離しにくい |
| 中止条件 | val機台でTeacherが不安定、境界が劣る、Studentが一貫して悪化する場合は無理に蒸留しない |

単純なconfidence-weighted KLだけを実装したものを「HeteroAKD再現」とは呼ばない。原論文の損失・正規化・勾配経路との差分を実装時に記録する。

## 7. 選定2：MGD

### 7.1 原論文と本件での狙い

原手法はStudent特徴を次元適合し、空間位置をランダムにマスクした後、小型generatorでTeacher特徴を復元する。GTのタスク損失と復元損失を併用する。[MGD本文](https://arxiv.org/pdf/2205.01529)・[著者コード](https://github.com/yzd-v/MGD)

本件では「限られた特徴から対象・背景の文脈を学べるか」を検証する。feature KDなのでDINO probeがなくても使える。データaugmentationによる効果とは切り分けて開始できる。

### 7.2 適用案と損失

整列後Student特徴を `P(F_S)`、Teacher特徴を `F_T`、位置ごとの保持maskを `M`、蒸留専用generatorを `G` とする。

```text
R = G(P(F_S) * M)
L_KD = mean_valid((R - stop_gradient(F_T)) ** 2)
L_total = L_seg + lambda_KD * L_KD
```

ここでの `mean_valid` は本件の実装案であり、原論文のsum reductionと同じ係数を流用しない。DINOの特徴スケールに合わせた正規化を変更する場合も、MGDの改変として記録する。

feature maskは蒸留枝だけに適用し、Studentの通常segmentation経路はマスクしない。これは画像のぼかし・ノイズ・遮蔽augmentationを導入したことにはならない。

初回はDINO最終patch特徴を使い、Student側に1×1 projection＋小型Conv generatorを配置する。多層Teacherや前景重み付けは追加ablationに分離する。

### 7.3 期待・リスク・判断条件

| 観点 | 判断 |
| --- | --- |
| 期待する効果 | 単なる点ごとの一致より、文脈を利用する特徴の学習を促す |
| 実装適合 | TeacherをFrozen／cachedに保ち、Student構造を変えず蒸留枝を追加できる |
| 主なリスク | generatorだけが復元を覚え、Student本体の未知機台精度が改善しない |
| マスク率 | 少量・二値・細部条件に合わせvalで選ぶ。原論文のsegmentation設定はmask率0.75だが、既定値として盲目的に採用しない |
| 必要な対照 | 同じgeneratorを使うmask率0の条件。maskによる効果と追加枝による効果を分離する |
| 改善の判定 | 復元lossではなくStudentのみのval／最終test性能で判定する |

「マスク復元を学習したので実画像のぼかしに強い」「未知背景に頑健」とは未検証のまま結論しない。ぼかし・露光変化への耐性は、別途固定した評価条件で測定する。

## 8. 選定3：GKD型分離学習のCNN適用版

### 8.1 原論文の条件

GKDはproxy画像での蒸留、source画像での蒸留、encoderを凍結したタスク学習を組み合わせる。query-based soft distillation（QSD）に加えてmasked patch・CLS tokenの損失を持つ。主要実験はDINOv2とViT系Studentである。[GKD本文](https://arxiv.org/html/2603.02554v1)・[著者コード](https://github.com/Younger-hua/GKD)

したがって、少量の機台画像だけ、CNN Student、CLSなしで実装するものは、原手法の完全再現ではない。

### 8.2 本件向け提案：`gkd_cnn_source_only`

まず外部proxyデータを追加せず、次の2段階を研究的な適用版として比較する。

| 段階 | データ | 学習対象 | 損失 | 禁止する利用 |
| --- | --- | --- | --- | --- |
| A：表現蒸留 | foldのtrain機台画像のみ | Student表現部と蒸留adapter | DINO特徴に対するQSD型損失 | val／test画像を「ラベルなし」として混ぜない |
| B：タスク学習 | 同じtrain機台画像とGT | decoder／分類部のみ | 既存GT segmentation loss | 表現部の重み・BatchNorm統計を更新しない |

教師は全段階でFrozen。段階AにはGT segmentation lossを入れず、段階BではTeacherなしでStudentのみを動かせる構成にする。

QSDの原式は概略として以下の処理である。**attentionで重み付けするvalueは投影したStudent特徴であり、Teacher valueを直接decoderへ渡す設計ではない。** 原式の転記・実装確認には本文の式7〜12を使う。

```text
A = softmax(phi(F_S) @ F_T.transpose(-1, -2))
R = A @ psi(F_S)
L_QSD_feature = MSE(R, stop_gradient(F_T))
```

本件へのCNN適用では、同じ空間gridに整列した特徴を使用する。CLS tokenをそのまま実装できないため、最初はpatch項に限定する。CLSをglobal poolingで代用した場合は別改変として扱う。masked入力項はStudentの追加forwardを要するため、実装・予算を確認して独立ablationにする。

### 8.3 CNN固有の実装課題

| Student | 課題・適用方針 |
| --- | --- |
| MobileNetV3＋LR-ASPP | `backbone`と`head`が分離しているため最初の対象に適する。ただし現行`features["kd"]`はhead内のcontext特徴なので、段階Aではbackboneのraw high特徴を新たに公開する必要がある |
| PIDNet-S | P／I／D経路が相互作用し、単純なencoder／decoder分離ではない。現行`features["kd"]`は融合後特徴。融合前後のどこを表現部として凍結するかを明文化し、全経路への学習信号を確認する |
| PP-LiteSeg | 今回の初期対象を広げない。必要なら同じ分離原則を満たすadapterを後続追加する |

PIDNetの最終分類部だけを学習する版は実装可能な候補だが、細部の適応力を失う危険がある。MobileNetでの検証を先行させ、凍結境界が成立しないStudentのrunは別名にするか未実施と表示する。

### 8.4 なぜ選ぶか、何を保証しないか

本テーマに最も直接的な仮説は、「source GTへのタスク適応と、DINO表現の獲得を分けると、未知機台性能を保持しやすいか」である。そのため、移植負担があっても3手法目として選ぶ。

一方、source-onlyでは機台固有背景だけを蒸留する危険が残る。CNNでは局所受容野や容量の限界もあり、ViTでの効果が再現するとは限らない。段階Bの凍結によって必要な境界情報を学べなくなる場合もある。

追加proxy版は別実験 `gkd_cnn_with_proxy` とする。外部データの利用許可、種類、枚数、計算予算を記録する。Oxford-IIIT Petだけを使って「ImageNetの多様性を再現した」とは主張しない。

### 8.5 必須の切り分け

1. 同じ段階Aの後、encoderを凍結する条件とfine-tuneする条件を比較する。
2. 同じ総更新回数・予算の、非蒸留Student条件を用意する。
3. QSD型の効果と段階分離の効果を混同しないよう、単純特徴lossを使う分離学習を小規模ablationに置く。
4. proxyあり／なしを別系列として示す。

段階Aのcheckpointは固定回数で決めるか、train機台内の事前に固定したmonitor subsetだけを使う。val機台を段階Aのfeature fittingへ入れない。最終的な段階Bのcheckpoint選択は従来どおりval機台のmIoUで行う。

## 9. 主選定から外した手法の位置づけ

| 手法 | 不採用の理由 | 再検討する条件 |
| --- | --- | --- |
| CWD | 安価で優れた比較手法だが、異種構造・DINOのDGへの適合を優先して外す | 再現性の高い標準KDとの比較が必要な場合 |
| SKD pair-wise／CIRKD | 構造移転は魅力的だが、異種構造の負の転移、sampling／queueの設計負担が残る | Teacher特徴の関係構造がvalで有効で、直接特徴KDが失敗する場合 |
| Af-DCD | 強いsegmentation手法だが、初期3条件を異なる仮説に割り当てた | MGDが有望で、より強い構造対比を試す場合 |
| BPKD | 対象の境界・細線に合う可能性が高いが、DINO probeの境界精度を未確認 | **valでTeacherのBoundary F1／細線Recallが高く、Studentの主な失敗が境界である場合。MGDとの入替候補** |
| FreeKD／Polyp-DiFoM | Teacher prompt・複数Teacher・decoder変更などがFrozen DINO単独比較と異なる | 周波数構造や境界が主因と確認でき、追加設計の効果を切り分けられる場合 |
| Proteus／大規模ドメイン蒸留 | 多様なproxy／ドメイン画像の規模を現在のGT枚数だけでは満たせない | train機台に限定した追加未ラベル画像や、許諾済みproxyを準備する場合 |
| Cosine／normalized MSE | 元の想定だからという理由で主選定に入れない。単純さは対照として価値がある | 実装sanity check、QSD・mask機構の切り分け |

L2正規化後の特徴では `||u-v||² = 2(1-cos(u,v))` であり、reduction等を揃えるとCosineとnormalized MSEは本質的に近い。両方を独立した「有望な2手法」と数える価値は低い。

## 10. 実験設計案

以下は調査に基づく提案であり、今回の変更でコマンドや蒸留runnerを実装したものではない。

### 10.1 共通条件

- machine単位の既存5-foldを固定。基本はtrain 3機台／val 1機台／test 1機台。
- test機台は学習、未ラベル蒸留、疑似ラベル、閾値決定、head・loss選択に使わない。
- Teacher weight、fold、Student初期重み、seed、入力サイズ、segmentation lossを揃える。
- 初回は画像augmentationなし。MGDのfeature maskやQSDの損失内処理は区別して記録する。
- Studentは原則同一初期化で比較。scratchとImageNet事前学習を混ぜない。
- headやTeacher layerの選択は事前固定またはval機台のみで行い、test結果を見て変更しない。
- GT枚数を減らす試験では、Teacher probeも同じGT subsetで学習する。Teacherだけ全GTを使う別条件と混ぜない。

### 10.2 最小限の比較と実行順

| 段階 | 条件 | 目的 |
| --- | --- | --- |
| 0 | 既存Student Baseline、Frozen DINO probe | 非蒸留性能とTeacherのタスク適合を確認 |
| 1 | 偽特徴による単体テスト、1 foldの短いsmoke | 凍結、整列、gradient、Student-only推論を確認 |
| 2 | MGDとmask率0対照 | 特徴復元・maskの効果を確認 |
| 3 | HeteroAKDと単純logit KD対照 | 異種構造・信頼性機構の効果を確認 |
| 4 | GKD型CNN版と段階分離対照 | 蒸留表現の保持がDGに効くか確認 |
| 5 | 事前固定した3候補の5-fold本評価 | fold対応付き差分を取得 |
| 6 | 有望条件のみ複数seed、GT削減、KD＋augmentation | 再現性・少量データ条件・拡張との相乗効果を確認 |

全手法が2 Studentに実装できる場合、3手法×2 Student×5 folds＝30 KD run／seed。ただしprobe、非蒸留、ablation、GKDの複数段階は別の計算費用となる。30回の通常学習と同じ費用とは見積もらない。

GKD型CNN版のPIDNet適用が未完成なら、まずMobileNetのみを正式比較に置き、欠ける条件を明示する。異なる凍結境界のrunを同一手法の結果として黙って混ぜない。

### 10.3 公平性とtestの扱い

HeteroAKDにはGTを使ったTeacher probeがあり、MGD／QSDにはraw feature Teacherがある。この違いも手法の一部なので、TeacherのGT利用、probe学習時間、Student更新回数、cache時間をそれぞれ報告する。

追加学習段階のある条件は「同じ総更新回数」と「手法推奨手順」の両系列を分ける。既存Baseline結果と学習予算が異なる場合、厳密な手法効果とは解釈しない。

val機台での小規模比較を終え、条件を固定してからtestを一度評価する。既に全機台のtest結果を見ながら次の条件を設計している場合、この5機台評価は探索的な再利用であると記載する。最終的な確認には、追加の未使用機台または外側のhold-out評価が望ましい。

## 11. 既存コードへ接続する際の注意

### 11.1 座標対応

DINOv2は416×640入力を内部で420×644へ右・下paddingする。patch gridは30×46である。Studentの416×640領域と、Teacherのpadding込みgridを、単に同じshapeへresizeするだけでは座標がずれる可能性がある。

実装時はpaddingとcropの座標規約を1箇所に固定し、合成画像の位置マーカー等で確認する。境界にかかるpatchの有効割合とignore maskを考慮する。DINOv3ではpatch16で26×40 gridとなるため、Teacher変更時にも同じ座標テストを行う。

### 11.2 feature接続点

現行Studentの `features["kd"]` はすべて同じ意味の「encoder最終特徴」ではない。MobileNetはhead context、PIDNetは融合後特徴である。MGD／HeteroAKDの最初の接続点として利用する案はあるが、GKD型のencoder凍結でそのまま使えるとは限らない。

追加する場合は既存forward契約を維持し、`encoder_high` 等の明示的なkeyを追加する。各手法の接続層・stride・チャネル数を結果に保存する。

### 11.3 キャッシュと学習時枝

- Teacherは常に `eval()`、`requires_grad_(False)`、`no_grad()`。
- 初回は同じ無拡張画像をTeacher／Studentに使用し、cacheの前処理・重み・層・manifest整合を検査する。
- KD optimizerへの入力はtrain sampleの特徴だけ。val／testは評価以外に利用しない。
- GKD型のStudent masked入力では、Teacherは無mask画像を参照する。Teacher cacheを変える処理とStudent側の処理を区別する。
- 後続の画像augmentationではonline Teacherを基本とする。cacheへ色・幾何変換を無検証で適用しない。
- 蒸留枝はexport時に除去し、Teacherも蒸留枝もなく同じsegmentation出力を得られることを検査する。
- 外部repositoryをそのまま依存へ追加せず、必要なloss／adapterを小さく実装する。原コードを利用する場合はそのライセンス・出典を記録する。

## 12. 解析結果の示し方

| 解析 | 推奨する表示 | 解釈上の注意 |
| --- | --- | --- |
| 主評価 | 手法×Studentの機台平均mIoU／最悪機台／foreground IoU | 背景優位の平均値だけで結論しない |
| fold対応差分 | 同じStudent・fold・seedの `KD - noKD` をdot plotとCSVで表示 | 別split同士の比較はしない |
| 機台別 | 機台×手法の絶対性能と差分heatmap | 一部の機台だけ改善していないか確認 |
| 境界・細線 | Boundary F1、必要ならSkeleton Recall／Connectivity | Teacherのpatch gridから細部が復元できると仮定しない |
| 撮像位相 | delay別の性能・foreground面積率 | 位相により対象が細くなる／隠れる可能性を検証 |
| 少量GT | 1機台あたりのGT枚数と未知機台性能のcurve | 枚数と位相多様性を同時に変えない |
| 表現 | 共通座標のpixel特徴PCA、class別重心・分離度、CKA | 異なる座標系のTeacher／Student重心距離をそのまま比較しない |
| 背景依存 | train／valで背景摂動・背景のみ入力の診断 | 合成画像は分布外。補助診断であり原因の確定ではない |
| 負の転移 | Teacher誤り・Student誤り・KD差分の重なり | Teacherの誤りを引き継いでいないか確認 |
| 製品性能 | Student-only CPU latency／RAM／容量 | GPUで速いだけでは採用しない |

PCAはtrain特徴でfitした共通変換をval／testへ適用する。class別の機台重心を比較し、foreground比率の違いだけで距離が増えていないか確認する。test GTを使う可視化は条件固定後の解析に限る。

各fold・機台の差分とseed変動を優先して示す。5機台の画像を独立標本とみなした狭い信頼区間は使わない。foldのtrain機台も重なるため、有意差検定から強い一般化結論を出さない。

## 13. 実装前の確認事項と受入条件

### 実装前

1. 業務環境のtrainデータだけで、対象の幅・foreground比率・機台／位相の偏りを確認する。
2. DINO probeのval mIoU、Boundary F1、失敗画像を確認する。
3. 優先Student、Teacher、接続層、初期重み、比較予算を固定する。
4. HeteroAKDの原論文lossと本実装案の対応表を作る。
5. GKD型CNN版の凍結境界と、proxyなし／CLSなし等の改変を確定する。

### 受入条件

- 既存Baseline／要因分析／Frozen probeを壊さない。
- Teacherにgradientが流れない。凍結表現部のBatchNorm統計も変わらない。
- train／val／testの機台重複、Teacher probeのfold不一致を拒否する。
- KD損失の対象領域・ignore・padding・reductionをテストする。
- cache／online整合とStudent-only exportをテストする。
- Teacher／Student総パラメータ、蒸留枝パラメータ、学習時間、推論費用を分けて記録する。
- 論文再現、損失成分だけの利用、CNNへの研究的改変をrun名で区別する。
- smokeの成功を未知機台での有効性確認とは呼ばない。

## 14. 最終判断

本テーマへの推薦は、**HeteroAKD、MGD、GKD型分離学習のCNN適用版**とする。

HeteroAKDは構造差への実証的な根拠、MGDはFrozen feature Teacherを利用する実装・実験上の適合性、GKD型は未知ドメインへの汎化保持という目的への直接性を重視した。

ただし、3つとも本データで有効だと断定はできない。DINO probeの境界が有用で、細部の欠落が主因ならBPKDをMGDとの入替候補とする。GKD型のCNN移植は、特にPIDNetで原手法の前提を満たすか慎重に確認する。

当初のCosine／normalized MSEを残す場合は「元の計画を守るため」ではなく、複雑な蒸留機構の効果を測るための簡単な対照として使う。

本ファイルはGit管理用の調査資料である。会社画像、mask、feature cache、Teacher weight、認証情報は含めない。論文・コードの利用条件は実装時に別途確認し、商用製品への利用可否をこの調査だけで判断しない。
