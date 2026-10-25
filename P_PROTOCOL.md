# 準備P プロトコル

状態: 診断は終了。主仕様は `P_DECISIONS.md`。テーマ A の fit は未実行。

## 再現と研究用の区別

再現は `python -m robustad` と `results_a1` のままである。fit は正常 180 枚、F1 はテストラベルで最大化する。

研究用は `python -m robustad.prepare` と `results_p0` である。ここでの主評価は、正常 calibration で決めたしきい値の検出率である。テスト最大 F1 は診断列である。

## 診断する条件

| データ | ドメイン | 役割 |
|---|---|---|
| Metal Parts | SD, Scale | PatchCore の画像・画素の乖離 |
| PCB | SD, Rotation, Position | PatchCore の幾何条件 |
| PCB | Shadows | 測光の対照 |
| PCB | Lighting | PaDiM の順位が崩れる条件の監査 |

フォルダ名は `catalog.py` の `test0` から `test5` である。Pile of Packets は画素マスクがないので、連結成分の対象から外す。

## 採否

集約、平滑化、有効領域の主仕様は、スコアを欠陥、未分割の非欠陥、境界へ割り当てられるかと、定義が再現できるかで決める。Scale、Rotation、Position の AUROC が上がったことは採用理由にしない。診断 AUROC は保存するが、目的関数にしない。

現時点で定義として固定したのは次である。

- 再現画像スコアは `pred_score`。PatchCore では再重み付け。PaDiM では平滑化マップの最大値。
- 平滑化の対照は sigma 0、4、8。主マップの sigma 4 は再現の定義である。
- 境界幅の対照は 0、16、32、64。16 は sigma 4 のカーネル半径であり、受容野の除去ではない。
- 連結成分は 8 近傍。評価用マスクは最近傍の 512 化。
- 有効領域の外に出た成分は、元 ID と残存面積率を残し、断片を新しい欠陥にしない。

決定済みは、主画像スコア、sigma 4、成分最大、画像 FPR 5%、144/36、画像全体を有効領域とすることである。改善幅、同等性幅、ROI、float16 キャッシュ、境界の受容野は未解決だが、診断の延長理由にはしない。

## 主解析と補助解析

主解析は、Source の正常 calibration で固定した画像しきい値による欠陥画像検出率と、正常パッチで校正したしきい値による連結成分の検出率である。共通内部領域と全画像の両方を出す。

補助は、Image AUROC、Pixel AUROC、スコア分布、欠陥種別と寸法の層別、テスト最大 F1、条件ごとに取り直したしきい値、である。

## 実行してよいコマンド

```text
python -m robustad.prepare inventory --data-root K:\RobustAD --output-dir results_p0
python -m robustad.prepare manifest --repo-root . --output-dir results_p0
python -m robustad.prepare proposal --data-root K:\RobustAD --output-dir results_p0
python -m robustad.prepare smoke --image-size 64 --seed 0 --output-dir results_p0
python -m robustad.prepare freeze-split --data-root K:\RobustAD --output-dir results_p0
python -m unittest tests.test_prepare tests.test_theme_a_warp
```

180 枚の再構築も、144 枚の研究用 fit も、この入口からは始まらない。研究用 fit の計画は `A_MINIMAL_PLAN.md`。
