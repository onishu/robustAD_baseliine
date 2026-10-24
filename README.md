# RobustAD 再現実験

[Pemula et al., CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/VAND/papers/Pemula_Robust_AD_A_Real_World_Benchmark_Dataset_For_Robustness_in_CVPRW_2025_paper.pdf) の Table 2 にある教師なしベースラインのうち、PaDiM と PatchCore を [anomalib](https://github.com/open-edge-platform/anomalib) で再現する。

データはダウンロード済みの `K:\RobustAD` を直接読む。画像のコピーは作らない。

## 論文に合わせた設定

- モデルは anomalib の既定値。PaDiM は ResNet-18、`layer1`–`layer3`、特徴次元 100。PatchCore は Wide-ResNet-50-2、`layer2` と `layer3`、coreset 比 0.1、近傍 9。
- 入力だけ 512×512。ImageNet の平均と標準偏差で正規化する。PatchCore 論文の 224 center crop は入れない。RobustAD 論文は「anomalib の既定値、ただし画像サイズは 512」と書いており、欠陥が小さいので解像度を上げている。
- 学習に使うのは各 train フォルダの正常画像すべて（180 枚）。anomalib の Folder は `normal_dir` をそのまま学習に使い、検証用の間引きはテスト側から取る。PaDiM / PatchCore のメモリバンクは検証画像を使わないので、公開テストフォルダは枚数を減らさず評価する。付録 Table 9 の 144/36 は教師あり分割で、この再現では使わない。
- 評価は test0 をソースドメイン、残りのフォルダを Table 2 の各シフトとして全画像で行う。主指標は image AUROC。PCB と Metal Parts は pixel AUROC と pixel AUPR も保存する。
- ARD は論文の式 (1) で、ターゲットについて `min(0, T - S)` の平均。
- 5 シード（0–4）は学習画像を固定したまま、モデル側の乱数だけを変える。PaDiM は特徴選択、PatchCore は coreset の初期点。

## 準備

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python -m pip install -e .
```

## 実行

論文どおり 3 サブセット × 2 モデル × 5 シード:

```powershell
.\.venv\Scripts\python -m robustad --data-root K:\RobustAD --output-dir results
```

1 シードだけ先に確認する場合:

```powershell
.\.venv\Scripts\python -m robustad --seeds 0 --output-dir results
```

終わった run の JSON があれば飛ばす。集計だけやり直すときは `--summarize-only`。

結果は `results/runs/*.json` と `results/summary.txt`。各ドメインの image AUROC を論文 Table 2 と並べる。
