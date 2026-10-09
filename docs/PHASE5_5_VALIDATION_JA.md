# Phase5.5 検証記録

検証日：2026-10-09。ローカルWindows / CPU環境。会社データは使用していません。

## 確認済み

- 過去フェーズのPythonコード・設定・テストは変更せず、新規ファイルだけで追加。
- 通常テストに、重み変換、SEキー対応、部分読込監査、ヘッドの非読込、SHA不一致、
  計画70run、Phase5設定のコピー、条件不一致、Teacher不一致、比較先の曖昧さ、
  交互作用差分、GKDのbudget表示、epoch再開のテストを追加。
- 実際の2生徒×4条件（GTのみ＋KD3手法）×偽DINOv2/v3で、合成データの学習、
  test評価、Student-only復元、予測画像、cached特徴、再実行スキップを確認。
- 公式MobileNetV3-Large IMAGENET1K_V2重みの特徴抽出部は学習パラメータ読込率100%。
  SE・BN bufferも対応し、LR-ASPPヘッドは全て未読込。
- 公式MobileNetV3＋公式DINOv2 ViT-S/14で、GTのみ・MGD・HeteroAKD応用・GKD-CNN応用を
  online smokeとcached本実験経路の両方で各1fold実行。jointは1epoch、GKDは各段階1epoch。
- cache再利用、完了済みrunの再実行スキップ、DINOv3重み不足の明示的スキップを確認。
- 実際の過去合成Phase5結果を取り込み、4条件の対応付き差分、機台ヒートマップ、
  条件別IoU図、同一sampleの4条件比較画像を生成。
- `python -m ruff check .`通過。`python -m pytest --disable-warnings`は62 passed。

## 実重みで未確認の範囲

- **PIDNet-Sの公式ImageNet重み**：現行公式配布ルートは確認したものの、ImageNetサブフォルダの
  非ログイン取得はGoogle Driveのサインイン画面となり、実重みを取得できませんでした。
  対応形式の合成重みで変換・学習を確認していますが、公式ファイルで確認済みとは主張しません。
  業務PCで正規に取得し、変換監査と1fold smokeを必ず確認してください。
  未知のキー・形状などは黙って無視せず拒否するため、取得したファイルが想定形式と異なる場合は
  エラー全文と重みのキー一覧・出典情報を確認してから対応してください。
- **DINOv3**：ローカルに承認済み重みがないため、偽backboneのテストまで。
  業務PCの公式repository・正規重みでsmokeを実行してください。
- GPU上の実測、会社データ5foldでの改善、有効な学習率、実機CPU推論性能は未評価。

合成データの1epoch性能は動作確認に限定され、事前学習やKDによる汎化改善の証明には使いません。
業務PCの操作は [Phase5.5手順書](PHASE5_5_PRETRAINED_KD_PC_JA.md)を参照してください。
