# 準備P 実装調査

既存の再現コードと保存結果を読んだ記録。推測で埋めていない。行番号は調査時点のファイルに対応する。

## 実行入口

再現の入口は `python -m robustad`。`src/robustad/__main__.py` が `benchmark.main` を呼ぶ。設定ファイルはなく、引数は `benchmark.parse_args`（666行付近）。既定は `--data-root K:\RobustAD`、`--output-dir results`、両モデル、3データセット、seed 0–4、`--weights timm`、`--image-size 512`、`--val-ratio 0.0`。

準備Pの入口は別にした。`python -m robustad.prepare`。出力先のディレクトリ名が `results`、`results_a1`、`results_tv` のときは書き込みを拒否する。

## モデルの実装箇所

PaDiM と PatchCore の学習・推論は、このリポジトリには再実装がない。`benchmark.build_model`（138行付近）が anomalib 2.7.0 の `Padim` と `Patchcore` を組む。

- PaDiM: backbone `resnet18`、layer1–layer3、`n_features=100`
- PatchCore: backbone `wide_resnet50_2`、layer2 と layer3、`coreset_sampling_ratio=0.1`、`num_neighbors=9`

`WEIGHT_SETS["timm"]`（31行付近）は anomalib の短い名前を使う。実行時に `pretrained_cfg` の tag または url が `a1_in1k` と `racm_in1k` を含むことを確認し、外れていれば停止する。この調査の環境では anomalib 2.7.0、timm 1.0.30。

## 前処理

`PaperPreProcessor`（47–88行）は画像を 512×512 に `Resize(..., antialias=True)` し、画像だけ ImageNet の平均 `[0.485, 0.456, 0.406]` と標準偏差 `[0.229, 0.224, 0.225]` で正規化する。center crop はない。レターボックスの padding もない。非正方形は縦横を独立に 512 へ縮める。

Metal Parts の実ファイルは 2681×1500、PCB の正常画像の一部は 2560×2560、異常画像の多くは 2048×2048 だった。Metal Parts は 512 化で縦横比が変わる。

マスクがバッチに付いているとき、同じ Resize のあと `(gt_mask > 0)` で二値化する。画素指標の本体はこちらではなく、`score_domain` がマスクファイルを最近傍で 512 に合わせる経路である（下）。

## 正常モデル

`data.build_train_module` は `train/normal` だけを読む。`--val-ratio 0` のとき 180 枚すべてが fit で、`n_val` は 0 として記録される。検証用に先頭 1 枚のプレースホルダを作るが、その画像は fit のリストにも入り、`limit_val_batches=0` なので校正には使われない。

`results_a1/runs/pcb__patchcore__seed0.json` の `train` は `n_normal_pool=180`、`n_train=180`、`n_val=0`、`val_ratio=0.0`。レポートの「正常画像 180 枚」と一致する。

seed は `seed_everything(seed, workers=True)`（343行付近）でモデル構築の前に置く。`val_ratio=0` では画像集合は seed で変わらない。変わるのは PaDiM の `torch.randperm` による特徴次元の選択（anomalib の `padim/torch_model.py` 140行付近）と、PatchCore の coreset 初期中心である。coreset ループは `benchmark.accelerate_patchcore_coreset` が GPU 上の同じ選択規則に差し替えている。距離計算の装置が変わると選択が数値的にずれうる。既存結果はこの差し替え込みの結果である。

## 画像スコアと異常マップ

PatchCore の `forward`（anomalib `patchcore/torch_model.py`）:

1. layer2 と layer3 を取り、各マップに `AvgPool2d(3, 1, 1)` をかける。
2. layer3 を layer2 の解像度へ bilinear で拡大して結合する。
3. 異常マップ用のパッチ距離は `nearest_neighbors(..., n_neighbors=1)`（185行付近）。メモリバンク全体への最小距離である。
4. 画像スコアは `compute_anomaly_score`（385行付近）。最悪パッチの距離に、その最近傍の近傍 `num_neighbors` 個から作った重みをかける。`num_neighbors` が 1 のときだけ最悪パッチ距離そのものを返す。
5. 重みは `weights = (1 - softmax(support_distances))[:, 0]`（440行付近）。平滑化マップの最大値ではない。
6. パッチ距離を空間に戻し、`AnomalyMapGenerator` が `F.interpolate`（mode 引数なし、PyTorch の既定は nearest）で入力サイズへ拡大し、`sigma=4` の Gaussian をかける。カーネル幅は `2 * int(4*sigma+0.5)+1 = 33`。半径 16。padding は reflect で、出力サイズは変わらない。

入力 512 の実測: Wide-ResNet-50-2 の layer2 は `(1, 512, 64, 64)`、layer3 は `(1, 1024, 32, 32)`。結合次元は 1536。PaDiM の ResNet-18 は layer1 `(1, 64, 128, 128)`、layer2 `(1, 128, 64, 64)`、layer3 `(1, 256, 32, 32)`。PaDiM は後段を layer1 の格子へ nearest で揃える。

PaDiM の画像スコアは平滑化後マップの `amax`（`padim/torch_model.py` 194行付近）。距離は Mahalanobis、拡大は bilinear、`align_corners=False`、平滑化は同じ sigma 4。

`num_neighbors=9` が作用するのは、最悪パッチの再重み付けだけである。異常マップの各位置は最近傍 1 個の距離である。

## 予測値は正規化されていない

`predict_step`（anomalib の `anomalib_module.py` 297行付近）は torch モデルを直接呼ぶ。PostProcessor は min/max が未設定なら入力を返す。Engine は `limit_val_batches=0`。保存 JSON の image threshold は、PCB PatchCore SD で 35.89 のように 1 を超える。0–1 正規化後の値ではない。

既存 F1 は `metrics.best_detection`。評価と同じラベルで F1 最大のしきい値を選ぶ。関数コメントはこれを検査の動作点と書いている。研究プロトコルでは事後値として扱い、運用しきい値とは呼ばない。このコメントと再現の計算は変えていない。

## 保存されているもの

`results`、`results_a1`、`results_tv` の `runs/*.json` はドメイン単位の指標である。画像ごとのスコア、パッチ距離、平滑化前マップ、メモリバンク、チェックポイントはない。`--save-checkpoint` は既定でオフで、リポジトリ内に研究用の `.pt` はなかった。

`results_a1/vis/.../cases.json` は seed 0 の高スコア・低スコア例のパス、画像スコア、画素の IoU だけである。異常マップの配列は描画に使ったあと破棄される。PNG からスコアは復元しない。

したがって、領域帰属と集約比較には再推論が要る。既存 JSON の Image AUROC を、欠陥反応と正常過反応に分解することは、このキャッシュだけではできない。

## データ

レイアウトは `catalog.SUBSETS` と一致した。`metadata.jsonl` の列は `file_name`、`label`、`mask` だけである。ワーク ROI はない。Pile of Packets に `masks` はない。

診断対象の枚数は `results_p0/inventory.json` に書いた。PCB と Metal Parts の異常ファイル名はすべて既知の欠陥名に分解できた。未分解 0 件。Packets は対象外なので欠陥名を付けていない。

診断対象マスク 476 枚は、読んだ範囲では画素値が 0 と 255 だけだった。非二値マスクは 0 件。元解像度の連結成分は 823 個。最近傍で 512 にしたとき、単独リサイズで消えた成分は 4 個（面積 1、10、11、13）。画像全体を一度に 512 へ縮めたときの成分数が元より少ない画像は 16 枚。これは消失と融合が混ざった数で、原因ごとの内訳は成分対応をまだ付けていない。

Metal Parts の train 正常と test0 正常は 104 個のファイル名を共有し、sha256 が一致したのは 0 個だった。同名でも同一ファイルではない。同じ個体の別撮影かは未確認。

## レポートとの一致

`REPORT.md` のモデル、入力 512、crop なし、180 枚、seed 0–4、層と coreset 比、近傍 9 は、上記のコードおよび `results_a1` の JSON と一致する。コードをレポートに合わせて書き換える必要はなかった。

JSON の seed 0 の PCB PatchCore Position は image AUROC 0.6667（66.67）。レポートの 67.50 は 5 seed の平均である。同じ量ではない。矛盾ではない。

## テスト

既存の自動テストはなかった。`tests/test_prepare.py` を追加した。再現の `benchmark.py` の計算式は変えていない。
