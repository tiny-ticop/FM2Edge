# 画像生成プロンプト

## 最終版への修正

### 01 MGD：GT説明を簡略化

```text
Edit only bottom middle supervised-learning branch of this MGD slide, keep entire upper diagram/title/footer/callouts unchanged. REMOVE both 領域予測 and GTマスク thumbnail panels and all their green arrows. Replace that entire small bottom branch with ONE green rounded box beneath student, containing exact 2 lines '領域予測とGTマスクを比較' and 'CE＋Diceで正解に近づける'. Keep a single orange arrow from 小型モデル into this one green box. No other arrows or thumbnails in this bottom branch. Keep all MGD feature reconstruction flow, gradient arrows, formulas and other content unchanged. This is a surgical edit to simplify the GT explanation.
```

この修正の前に、領域予測・GTマスクを白黒の2クラス模式図へ変更しています。最終版はこの小図を削除し、1つの教師あり損失ボックスへ置換します。

### 02 HeteroAKD：接続の誤解を避けるため再生成

```text
Create ONE Japanese 16:9 technical meeting slide, white background navy headings, blue frozen teacher, orange student, purple training-only auxiliary, green losses, flat vector diagram, large crisp Japanese fonts. Title exact "02｜HeteroAKD応用：信頼できる知識を選んで学ぶ". Subtitle "DINOの領域予測を、正解との比較で調整して伝える".
Scientifically correct SIMPLE diagram with 3 horizontal rows, ample space, no backwards gradient arrows to avoid clutter.
Row1 blue "DINOv2 / DINOv3（凍結）" → blue "領域ヘッド（固定）" → blue "教師の予測". Small note under probe "train / valだけで事前学習".
Row2 orange "小型モデル" → orange "生徒の特徴" → purple "補助1×1 Conv" → purple "補助予測".
Row3 fork from SAME 生徒の特徴 downward → orange "元の領域ヘッド" → orange "最終予測".
On upper right one purple box "知識を調整（KMM / KEM）" with exactly three bullet lines "GTとの誤差を画素・クラスごとに比較", "教師と補助生徒の予測を混合", "学ぶべきクラスを重み付け". Exactly three arrows into box from 教師の予測, 補助予測, and green GTマスク. Box outputs arrow into green "重み付き蒸留損失". THIS LOSS HAS NO OTHER INCOMING OR OUTGOING ARROWS. Show no connector between weightedloss and finalprediction or KL.
Lower right SEPARATE green box "最終出力の蒸留（KL）" with exactly TWO incoming arrows: BLUE from 教師の予測 via outer edge; ORANGE from 最終予測. No connection to any other loss. Do not let these two arrows cross other boxes. GT supervision stated below without extra flow arrows.
All prediction/GT miniatures if used only white foreground on black background (binary). Prefer no image thumbnail to maximize clarity.
Bottom two callout cards exact "何を蒸留する？" / "前景・背景の確率と、正解に照らした信頼性"; "なぜ試す？" / "教師が苦手な箇所をそのまま模倣せず、有用な知識を優先".
Bottom loss banner "学習 = GT教師あり損失 ＋ λ ×（重み付き蒸留 ＋ 最終出力KL）".
Notes "最終予測・補助予測はGTでも学習。最初はGTでウォームアップ。" and "DINO-probe向けの応用実装。原論文の完全再現ではない。"
Footer "評価：未知機台のGTは学習に使わない ｜ 推論：元の小型モデルのみ（構造・パラメータ数は不変）".
Constraints arrows strictly as above, NEVER weightedloss input from finalprediction, NEVER weightedloss output to KL. No performance numbers, no gradients arrows, no clutter.
```

### 03 GKD-CNN：第2段階だけ修正

```text
Make a minimal scientific correction to this GKD Japanese slide. Keep the entire left stage1 diagram, all copy, typography, colors and layout unchanged. In right stage2 ONLY: replace the multicolor 領域予測 and GTマスク thumbnails with white foreground against black background (binary segmentation). Set the mathematical labels for both to H×W×2, replacing K with 2. Keep the two green arrows FROM 領域予測 and GTマスク TO CE＋Dice. Replace the extra solid green arrow FROM 元の領域ヘッド TO CE＋Dice with a dashed green feedback arrow FROM CE＋Dice TO 元の領域ヘッド labeled 更新, indicating loss updates only head, not frozen feature extractor. Do not add other changes.
```


生成方式：組み込み image_gen ツール。透明背景なし。各画像は独立した生成です。初稿を目視確認し、記録した再生成・修正で最終版を作成しました。

## 01 MGD

```text
Use case: scientific-educational / productivity-visual.
Create ONE finished Japanese technical meeting presentation slide, landscape 16:9, high-resolution raster, white background, navy headings, blue frozen teacher, orange trainable student, purple training-only modules, green loss/GT. Crisp readable Japanese sans-serif typography, ample spacing, minimal elegant flat vector-style diagrams rendered as image, not a photograph of a slide. Industrial machine schematic input image only, NO company photos. Clear arrows with arrowheads, logical flow. All copy below must be verbatim Japanese, not garbled. No invented performance numbers or guaranteed gains. Teach engineers unfamiliar with KD. Common slim footer EXACT: "評価：未知機台のGTは学習に使わない ｜ 推論：元の小型モデルのみ（構造・パラメータ数は不変）". Tiny legend "青：固定　橙：学習　紫：学習時のみ". Distinguish motivation from observed results. No watermarks, no extra slogans.
Title "01｜MGD：隠した特徴を復元して学ぶ"
Subtitle "DINOの特徴を再現する力を、小型モデルに移す"
Main upper 60% two parallel lanes from common "学習機台の画像":
upper blue "DINOv2 / DINOv3" with lock icon and "凍結" -> blue "教師の特徴".
lower orange "小型モデル" -> orange "生徒の特徴" -> purple "次元を合わせる" small "1×1 Conv" -> purple "特徴の一部を隠す" with visibly masked grid -> purple "復元する" small "小型Conv" -> purple "復元した特徴".
Teacher features and recovered features converge to green "特徴の差を小さくする" small "MSE". Green dashed update arrow returns only to student and purple modules, never teacher. Additional thin bottom branch from student "領域予測" + "GTマスク" -> "正解に近づける（CE＋Dice）".
Bottom two short callout cards:
"何を蒸留する？" / "画像の各位置にあるDINOの視覚特徴"
"なぜ試す？" / "周囲の情報から特徴を補う力を学ばせ、機台差への頑健性を狙う"
Small note "隠すのは入力画像ではなく、生徒の特徴。復元器は推論時に不要。"
Formula banner "学習 = セグメンテーション損失 ＋ λ × 特徴復元損失"
Do not imply reconstruction of RGB images, teacher fine tuning, or masks used at deployment. Both student and adapters optimized jointly. Fit all copy large and legible.
```

## 02 HeteroAKD応用

```text
Use case: scientific-educational / productivity-visual.
Create ONE finished Japanese technical meeting presentation slide, landscape 16:9, high-resolution raster, white background, navy headings, blue frozen teacher, orange trainable student, purple training-only modules, green loss/GT. Crisp readable Japanese sans-serif typography, ample spacing, minimal elegant flat vector-style diagrams rendered as image, not a photograph of a slide. Industrial machine schematic input image only, NO company photos. Clear arrows with arrowheads, logical flow. All copy below must be verbatim Japanese, not garbled. No invented performance numbers or guaranteed gains. Teach engineers unfamiliar with KD. Common slim footer EXACT: "評価：未知機台のGTは学習に使わない ｜ 推論：元の小型モデルのみ（構造・パラメータ数は不変）". Tiny legend "青：固定　橙：学習　紫：学習時のみ". Distinguish motivation from observed results. No watermarks, no extra slogans.
Title "02｜HeteroAKD応用：信頼できる知識を選んで学ぶ"
Subtitle "DINOの領域予測を、正解との比較で調整して伝える"
Main diagram two lanes sharing "学習機台の画像":
upper blue "DINOv2 / DINOv3" -> blue "学習済み領域ヘッド" -> blue "教師の予測". Enclose blue lane label "蒸留中は固定"; small label under head "train / valだけで事前学習".
lower orange "小型モデル" -> orange "生徒の特徴" fork: purple "補助1×1 Conv" -> purple "補助予測"; another orange "元の領域ヘッド" -> orange "最終予測".
Green "GTマスク" feeds purple reliability panel alongside teacher and auxiliary student predictions.
Reliability panel contains numbered short items:
"① 正解との誤差を比較"
"② 教師・生徒の予測を混合（KMM）"
"③ 学ぶべきクラスを重み付け（KEM）"
Panel -> green "重み付き蒸留損失" -> dashed learning arrow to student/auxiliary classifier, no teacher.
Separate thin connector from "教師の予測" and "最終予測" to green "最終出力にも蒸留（KL）".
Tiny GT-supervision label "最終予測・補助予測はGTでも学習".
Bottom callouts "何を蒸留する？" / "前景・背景の確率と、正解に照らした信頼性";
"なぜ試す？" / "教師が苦手な箇所をそのまま模倣せず、有用な知識を優先"
Note "最初はGTでウォームアップ。DINO-probe向けの応用実装で、原論文の完全再現ではない。"
Loss banner "学習 = GT教師あり損失 ＋ λ ×（重み付き蒸留 ＋ 最終出力KL）"
Important: reliability is per pixel AND class; not hard gate rejecting teacher, not teacher confidence alone; mixture uses teacher and auxiliary student outputs. Exact logical diagram primary.
```

## 03 GKD-CNN応用

```text
Use case: scientific-educational / productivity-visual.
Create ONE finished Japanese technical meeting presentation slide, landscape 16:9, high-resolution raster, white background, navy headings, blue frozen teacher, orange trainable student, purple training-only modules, green loss/GT. Crisp readable Japanese sans-serif typography, ample spacing, minimal elegant flat vector-style diagrams rendered as image, not a photograph of a slide. Industrial machine schematic input image only, NO company photos. Clear arrows with arrowheads, logical flow. All copy below must be verbatim Japanese, not garbled. No invented performance numbers or guaranteed gains. Teach engineers unfamiliar with KD. Common slim footer EXACT: "評価：未知機台のGTは学習に使わない ｜ 推論：元の小型モデルのみ（構造・パラメータ数は不変）". Tiny legend "青：固定　橙：学習　紫：学習時のみ". Distinguish motivation from observed results. No watermarks, no extra slogans.
Title "03｜GKD-CNN応用：特徴学習と領域学習を分ける"
Subtitle "まずDINOの見方を学び、次にセグメンテーションを学ぶ"
Two large horizontally sequenced panels with clear central arrow, labels "第1段階｜特徴を学ぶ" and "第2段階｜領域を学ぶ".
Stage1 diagram shared "学習機台の画像" feeds blue "DINO（凍結）" -> blue "教師の特徴" and orange "小型モデルの特徴抽出部" -> purple "Query・Valueへ変換".
Explicit relation diagram three labeled tensors "生徒Query", "教師特徴", "生徒Value". StudentQuery and teacherFeatures converge purple "対応関係を計算" -> purple "生徒Valueを再構成" using studentValue arrow -> purple "再構成特徴"; teacherFeatures and reconstruction converge green "教師特徴との差（MSE）". Small formula if room "A = softmax(Qs × Ftᵀ)　／　F̂ = A × Vs". NEVER show teacher features as Value; values come from student.
Stage1 small tags "特徴抽出部を更新" and "領域ヘッドは固定・GT領域損失は使わない".
Stage2 simpler pipeline "学習機台の画像" -> blue lock "学習済み特徴抽出部（凍結）" -> orange "元の領域ヘッド（学習）" -> orange "領域予測"; green "GTマスク" and prediction converge green "CE＋Dice". No teacher needed stage2.
Bottom callout "何を蒸留する？" / "DINO特徴との位置間の対応を使った、視覚表現";
"なぜ試す？" / "汎用的な特徴を先に移し、少量GTによる領域学習と切り分ける"
Note "CNN向け・学習機台データのみの応用実装。外部proxyデータは使わない。"
Do not imply teacher-value reconstruction or trained new inference modules. Inference retains unchanged original student feature extractor and head.
```
