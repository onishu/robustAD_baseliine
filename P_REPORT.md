# 準備P レポート

状態: 診断は終了し、追加の網羅的診断は行わない。主仕様は `P_DECISIONS.md`。その後の144枚 fit と合成回転は `EXPERIMENT_REPORT.md` にまとめた。診断ドメインの AUROC の上下は、仕様の採否に使っていない。

## 実行したこと

- 再現コード、anomalib 2.7.0 の PaDiM と PatchCore、`results_a1` の JSON、`K:\RobustAD` のフォルダを読んだ。
- `python -m unittest tests.test_prepare`。再重み付けのキーワード引数を直したあと通過した。成分が縮小で消えても分母に残るテストを追加し、通過した。
- `python -m robustad.prepare inventory --data-root K:\RobustAD --output-dir results_p0`。対象外の Packets を未分解から外した再実行で、未分解は 0。
- `python -m robustad.prepare manifest --repo-root . --output-dir results_p0`
- `python -m robustad.prepare proposal --data-root K:\RobustAD --output-dir results_p0`。144/36 は提案のまま適用していない。
- `python -m robustad.prepare smoke --image-size 64 --seed 0 --output-dir results_p0`。装置の不整合を直したあと通過した。研究結果ではない。
- 実行前に `results_p0/diag/infer_manifest.json` を書いた。seed 0、入力 512、正常 180 枚、`val_ratio` 0、`num_workers` 4、144/36 なし、ROI なし。
- `python -m robustad.prepare diagnose --confirm-fit180 --output-dir results_p0 --data-root K:\RobustAD --reference-dir results_a1 --repo-root .`
  - Metal Parts / PatchCore: SD、Scale
  - PCB / PatchCore: SD、Rotation、Position、Shadows
  - PCB / PaDiM: Lighting
- 最初のドメイン（Metal PatchCore SD）の Image AUROC 差は 0 だったので、残りのドメインも実行した。`results_a1` への書き込みはない。

fit 済みバンクは `results_p0/diag/banks`。PatchCore は両方とも `(73728, 1536)` の float32 で、1 本 452,984,832 バイト。PaDiM は平均 `(100, 16384)`、逆共分散 `(16384, 100, 100)`、合計 661,913,600 バイト。バックボーンの重みは保存していない。

各画像の npz は、パッチ格子と sigma 4 の平滑化マップ（float16）、再現画像スコア、パッチ最大（float32）。平滑化前の全画素マップは、PatchCore は nearest、PaDiM は bilinear（`align_corners=False`）でパッチ格子から再構成する。実行前の空きは 3,561,525,248 バイト、推定は 2,209,890,304 バイトだった。CNN 埋め込みは保存していない。実ファイルはマップ 644,376,692 バイト、バンク 1,567,888,966 バイト、表 997,428 バイト。

## 1. 既存結果との再現互換

7 ドメインとも、Image AUROC と Pixel AUROC は `results_a1` の seed 0 と差 0 だった。

| データ | モデル | ドメイン | Image AUROC | Pixel AUROC |
|---|---|---|---:|---:|
| Metal Parts | PatchCore | SD | 0.8831699346405228 | 0.9702954904636321 |
| Metal Parts | PatchCore | Scale | 0.8057291666666666 | 0.9761190072201908 |
| PCB | PatchCore | SD | 0.9624444444444444 | 0.9754237163504103 |
| PCB | PatchCore | Rotation | 0.6877083333333334 | 0.9676006431439781 |
| PCB | PatchCore | Position | 0.6666666666666666 | 0.9033869021678111 |
| PCB | PatchCore | Shadows | 0.45416666666666666 | 0.9333766272467346 |
| PCB | PaDiM | Lighting | 0.04129629629629629 | 0.8475744595288653 |

メモリバンクの長さは、事前計算の `int(180*64*64*0.1) = 73728` と一致した。

連結成分は 823 個で、診断対象の元マスク成分数と一致する。この 823 が分母である。512 への最近傍縮小で消えた 4 成分は、状態「入力変換による評価不能」のまま分母に残している。

| ドメイン | 成分 ID | 元面積 | 単独縮小後の面積 |
|---|---:|---:|---:|
| PCB SD | Test_PCB_None_None_Scratch_27 の 3 | 10 | 0 |
| PCB Shadows | Test_PCB_Shadow_Object Alternate Location_Scratch_10 の 2 | 11 | 0 |
| PCB Shadows | Test_PCB_Shadow_Object Alternate Location_Soldering Melt_5 の 8 | 13 | 0 |
| PCB Lighting | Test_PCB_Lighting_Evening_Missing Component_2 の 1 | 1 | 0 |

他の成分の状態は `evaluable` である。Metal Parts の SD と Scale に評価不能成分はない。

## 2. スコア経路の再計算整合

実行中、再現スコアと `pred_score` の差が 1e-4 を超えるバッチは無かった。超えていればその場で停止する。

PatchCore では、最近傍拡大の最大値とパッチ最大の差は 0 だった。再現スコアはパッチ最大の 0.979 倍から 1.000 倍で、保存した比を掛け戻すと再現スコアに戻る。平滑化後最大との差は最大で、SD 3.78、Scale 3.75、Rotation 2.39、Position 2.45、Shadows 2.01 だった。再現スコアは平滑化マップの最大ではない。

PaDiM Lighting では、再現スコアと sigma 4 のマップ最大の差は 0 だった。パッチ格子の最大と bilinear 拡大の最大は、最大 2.38 ずれた。画像スコアの定義は後者ではなく、平滑化後の最大である。

保存した float16 のパッチ格子から同じ拡大と sigma 4 のぼかしをやり直すと、保存済み平滑化マップとの最大絶対差は、抜き出した PatchCore 画像で約 0.028、PaDiM 画像で約 0.11 だった。これは格子を float16 にしたときの差である。最悪パッチの再重み付けをバンクだけからやり直すには、そのパッチの埋め込みが要る。埋め込みは保存していない。比のスカラーは画像 CSV にある。

## 3. 最大スコアの領域帰属

ROI は作っていない。非欠陥は `nondefect_unpartitioned` のままである。最大値の台地が複数領域にまたがるときは、一つの領域に決め打ちしていない。境界幅 0 では、どの画像でも最大値は境界に乗らなかった。

割合は、そのクラスの画像のうち、最大値の台地がその領域だけにある、またはパッチブロックが欠陥と交わる割合である。

PatchCore の画像スコアを動かすのは最悪パッチのブロックである。異常画像で、そのブロックが欠陥と交わった割合は、Metal SD 36/51、Scale 53/80、PCB SD 20/75、Rotation 21/60、Position 20/60、Shadows 3/60 だった。ブロックが欠陥の外だけだった割合は、それぞれ 15/51、27/80、55/75、39/60、40/60、57/60 である。

sigma 4 マップの画素最大が欠陥だけにあった異常画像は、Metal SD 28/51、Scale 33/80、PCB SD 9/75、Rotation 12/60、Position 12/60、Shadows 3/60 だった。残りは非欠陥だけにあり、欠陥と非欠陥の両方にまたがる台地は無かった。

PCB PaDiM Lighting の異常 90 枚では、パッチ最大も sigma 4 の画素最大も、すべて非欠陥だけにあった。欠陥領域の最大の中央値は 67.5、非欠陥の最大の中央値は 174.9 だった。画像最大は欠陥の外にあり、境界幅 16 以上に乗ったのは 1/90 である。正常 120 枚の最大は境界に乗らなかった。ここから、正常ワークと背景のどちらが最大かは言えない。

正常画像の最大は、全ドメインで非欠陥だけにあった。欠陥マスクが空なので、これは定義どおりである。

## 4. 集約・平滑化・境界の診断

同じ fit から読んだ画像 AUROC を並べる。これは観測であり、主スコアの採否には使っていない。主スコアは既存の再現スコアのままである。

| ドメイン | 再現スコア | パッチ最大 | 拡大後最大 | sigma 4 | sigma 8 |
|---|---:|---:|---:|---:|---:|
| Metal SD | 0.8832 | 0.8835 | 0.8835 | 0.8752 | 0.8637 |
| Metal Scale | 0.8057 | 0.8063 | 0.8063 | 0.8091 | 0.8053 |
| PCB SD | 0.9624 | 0.9638 | 0.9638 | 0.9687 | 0.9778 |
| Rotation | 0.6877 | 0.6829 | 0.6829 | 0.6700 | 0.6321 |
| Position | 0.6667 | 0.6660 | 0.6660 | 0.6638 | 0.6348 |
| Shadows | 0.4542 | 0.4529 | 0.4529 | 0.4544 | 0.4629 |
| Lighting | 0.0413 | 0.0897 | 0.0897 | 0.0413 | 0.0407 |

sigma 4 は Metal Scale と PCB SD では再現スコアより高く、Metal SD と Rotation と Position では低い。向きが揃っていない。

sigma 4 の画素最大が外側 16 画素（ぼかし核の半径）に触れた割合は、PCB SD の異常 2/75、Metal Scale の正常 4/120、Lighting の異常 1/90 で、他の集計は 0 だった。外側 32 画素と 64 画素では触れる画像が増える。64 画素では、Shadows の異常 39/60 と正常 34/80、PCB SD の異常 29/75 で、sigma 4 の最大値の台地が境界に触れた。16 画素は受容野を消したことにはならない。境界の切り方は決めていない。

Lighting のテスト最大 F1 は 0.6 で、異常 90 枚をすべて異常と予測したときの F1 と一致する。運用点ではない。

## 5. その後に記録した判断

診断の数値からは決めていない。あとから記録した主仕様は `P_DECISIONS.md` にある。画像 FPR の目標は 5% で、しきい値は 95% 分位点である。成分の主統計量は最大値、研究用分割は 144/36 である。改善幅、同等性幅、ROI は未確定のままである。標本が足りない比較は結論不能とする。

## 解釈

画像 AUROC と画素 AUROC の差は、少なくとも今回のドメインでは、画像最大が欠陥の外に出ることと併記できる。Shadows では異常 60 枚中 57 枚で画像最大が欠陥の外にあり、画素 AUROC は 0.933、画素のテスト最大 F1 は 0.029 だった。Lighting では 90 枚すべてで画像最大が欠陥の外にあり、画像 AUROC は 0.041、画素 AUROC は 0.848 だった。これは非欠陥を正常ワークと背景に分けた説明ではない。

PatchCore の再重み付けは、今回の画像ではパッチ最大に対して最大でも約 2% の縮小だった。平滑化は最大値を 2 から 4 程度下げる。画像 AUROC への影響はドメインで向きが違う。

## 未確認

- Metal Parts の同名ファイルが同一個体の別撮影か。
- CNN の受容野が画像端から何画素まで影響するか。境界幅 16 はぼかし核の半径である。
- 独立した正常校正での検出率。今回は主評価ではない。
- 最悪パッチの埋め込みを保存していないため、バンク単体からの再重み付けの再計算。

## 反証と例外

縮小で消えた 4 成分は分母 823 に含まれる。大多数の成分は残る。

Packets の 690 ファイルは欠陥名パーサの失敗ではない。評価対象外である。

smoke のランダムバンクで、再重み付けスコアと平滑化最大は一致しなかった。RobustAD 上でも、PatchCore のその差は 0 ではなかった。

## 次の実験への影響

テーマ A の前に、上の 6 項目の記入が要る。領域の結論は、ROI が無い間は欠陥、未分割の非欠陥、境界までである。

## 成果物

- `P_IMPLEMENTATION_AUDIT.md`
- `P_PROTOCOL.md`
- `P_SCORE_DEFINITION.md`
- `P_REGION_DEFINITION.md`
- `P_CALIBRATION.md`
- `P_DECISIONS.md`
- `results_p0/inventory.json`
- `results_p0/image_index.csv`
- `results_p0/components_original.csv`
- `results_p0/mask_audit.csv`
- `results_p0/run_manifest.json`
- `results_p0/split_proposal.json`
- `results_p0/planning_halfwidths.csv`
- `results_p0/smoke_patchcore.json`
- `results_p0/status.json`
- `results_p0/diag/infer_manifest.json`
- `results_p0/diag/summary.json`
- `results_p0/diag/banks/`
- `results_p0/diag/maps/`
- `results_p0/diag/tables/`
