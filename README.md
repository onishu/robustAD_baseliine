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

## 実行の How to

作業ディレクトリはリポジトリのルート。PowerShell で打つ。コンソールの文字化けを避けるため、実行前に次を入れる。

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
```

終わった run の JSON があればその条件は飛ばす。やり直すときは `--force`。集計だけやり直すときは `--summarize-only`。3 サブセット × 2 モデル × 5 シードで、GPU 上のおおよそ 6–8 時間。

### 1. 論文 Table 2 の再現（既定の timm 重み）

anomalib 2.7 の既定バックボーン（`resnet18.a1_in1k`、`wide_resnet50_2.racm_in1k`）。Image AUROC を `results/` に残す。

```powershell
.\.venv\Scripts\python -m robustad --data-root K:\RobustAD --output-dir results --weights timm --num-workers 4
```

1 シードだけ先に確認する場合:

```powershell
.\.venv\Scripts\python -m robustad --data-root K:\RobustAD --output-dir results --weights timm --seeds 0 --num-workers 4
```

結果は `results/runs/*.json` と `results/summary.txt`。各ドメインの image AUROC を論文 Table 2 と並べる。

### 2. 全指標と検出画像（1 と同じ重み）

Image AUROC / Image F1、Pixel AUROC / Pixel F1、画像 ARD、画素 ARD、変動要因ごとの低下を計算する。F1 最大のしきい値で検出マスクを描き、シード 0 の画像ログを残す。比較相手は 1 の `results/`。

```powershell
.\.venv\Scripts\python -m robustad --data-root K:\RobustAD --output-dir results_a1 --weights timm --num-workers 4 --baseline-dir results
```

結果は `results_a1/summary.txt`、`results_a1/figures/`、`results_a1/vis/`。画像ログを書かないときは `--no-vis` を足す。

### 3. torchvision の ImageNet-1K V1 との比較

Table 2 に寄せるための重み確認。再現の本体は 1 と 2 で、この結果は `results_tv/` に分ける。

```powershell
.\.venv\Scripts\python -m robustad --data-root K:\RobustAD --output-dir results_tv --weights torchvision --num-workers 4 --baseline-dir results
```

集計だけやり直す例:

```powershell
.\.venv\Scripts\python -m robustad --output-dir results_a1 --summarize-only --baseline-dir results
```

## 準備PとテーマA

論文再現は上の 180 枚 fit のままである。付録 Table 9 の 144/36 は、研究用の fit と校正にだけ使う。リストは `results_p0/research_split_manifest.json`。`results_a1` と診断出力 `results_p0/diag` は上書きしない。

準備Pの主仕様は `P_DECISIONS.md`。主画像スコアは検出器ごとの再現スコア、平滑化は sigma 4、欠陥成分の主統計量は最大値、画像 FPR の目標は 5% である。改善幅と同等性幅は未確定で、差と区間を出しても実用的改善や同等とは書かない。

テーマAの最初の実験は PCB、PatchCore、seed 0 で、PCB SD の 195 枚に合成した `−5°` と `+5°` を掛ける。実撮影の Rotation フォルダは使わない。計画は `A_MINIMAL_PLAN.md`。出力は `results_a0` である。端を折り返した比較は `results_a0_boundary` で、どちらも上書きしない。

作用点の次の比較は `A_ACTION_PLAN.md`、基板面の補助評価は `A_BOARD_ROI.md` である。大規模な再fitは、その計画を確認するまで実行しない。
