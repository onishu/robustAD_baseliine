# スコア定義

状態: 主画像スコアは `reproduction_image`。主マップの平滑化は sigma 4。診断 7 ドメインで `results_a1` の seed 0 と Image / Pixel AUROC が一致した。

## 再現互換

`reproduction_image` は、モデルが返す `pred_score` である。

PatchCore では、各パッチの最近傍距離のうち最大のものを `s` とし、そのパッチの最近傍がメモリバンク内でもつ近傍 9 個への距離を softmax した第 0 成分を使って `(1 - softmax)[0] * s` とする。実装は anomalib の `compute_anomaly_score`。`prepare/scores.py` の `patchcore_reweight` は、同じ支持距離を渡したときこの関数と一致する。単体テストで確認した。

PaDiM では、平滑化後の異常マップの最大値である。再重み付けはない。

このスコアを、平滑化後マップの最大値と同じものとして集計しない。

## 診断用の別スコア

同じパッチ距離から、次を別に持つ。

| 名前 | 定義 |
|---|---|
| `diag_max_patch` | 平滑化前、拡大前のパッチ距離の最大値 |
| `diag_max_upsampled` | 拡大後、平滑化前のマップの最大値 |
| `diag_max_smoothed` | 平滑化後マップの最大値 |
| `reproduction_image` | 上の再現互換スコア |

PatchCore の `diag_max_patch` は再重み付け前の最悪パッチ距離である。`reproduction_image` はそれに重みがかかった値である。`diag_max_smoothed` はマップ側の最大値である。

入力 64、ランダムなメモリバンク 24 本、バッチ 2 の smoke では、`reproduction_image` は `pred_score` と一致し、平滑化後最大値との差の絶対値の最大は 1.072 だった。この差は RobustAD の検出性能ではない。`results_p0/smoke_patchcore.json` の `research_result` は false。

## マップの生成

PatchCore の拡大は `F.interpolate` の既定（nearest）。PaDiM の拡大は bilinear、`align_corners=False`。どちらもその後に sigma 4 の Gaussian をかける。カーネルは 33、半径は 16。

診断で比較する sigma は 0、4、8 に固定する。4 は再現の定義である。0 と 8 は平滑化への依存を見る対照であり、AUROC が良い方を採用しない。

## 座標

モデル入力の 512×512 が基準座標である。元画像 `(幅 W, 高さ H)` からこの座標への縮尺は `512/W` と `512/H` で、正方形でない画像では一致しない。

PatchCore のパッチ格子は、入力 512 で layer2 の 64×64。1 パッチは入力座標の 8 画素に対応する。smoke の入力 64 ではパッチマップが `(batch, 1, 8, 8)` で、同じ 8 画素刻みだった。PaDiM の格子は layer1 の 128×128、刻み 4 画素。

パッチ中心を元画像の物理寸法へ戻す変換は、上記の縮尺の逆を使う。受容野の端がどこまで届くかは、この刻みだけでは決まらない。未測定のまま「受容野を除去した」とは書かない。

## キャッシュ

中間値を保存するときは、設定の sha256 をヘッダに入れる。`prepare/cache.py` は識別子が違うファイルを読むと `CacheMismatch` を出す。RobustAD の中間キャッシュは、まだ作っていない。

保存する段階は、パッチマップ、拡大後・平滑化前、平滑化後、再現互換の画像スコア、正解マスク、画像ラベル、欠陥名、ドメインである。再重み付けの支持距離は、再現スコアそのものを保存すれば診断の 3 比較には不要である。支持距離を後からマップ最大値で代用することはしない。
