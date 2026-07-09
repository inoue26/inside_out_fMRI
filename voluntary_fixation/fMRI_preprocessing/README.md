# studyforrest fMRI 前処理 → SRM 入力 npz 作成手順

*日本語版。English: [README.en.md](README.en.md).*

ds000113（StudyForrest, `ses-movie/task-movie`, 8 run）から、denoise 済み BOLD・ROI 中間物・
SRM 入力 npz を作るパイプライン。生成物は次の 3 つ:

```
data/forrestgump/studyforrest/
  bold/       ← 手順1: denoise 済み BOLD (sub-XX_run-Y_space-MNI152NLin2009cAsym_bold.nii.gz)
  bold_int/   ← 手順2 が自動生成する中間物 (subXX_base.bin / subXX_sideLR_roi44_all.bin ...)
  roi22LR_np/ ← 手順2 の最終出力 (subXX_sideLR_roi44_data-12345678.npz)  ← SRM 入力
```

パイプライン全体:
```
BIDS raw (ds000113, ses-movie/task-movie, run 01..08)
  │  手順0: fMRIPrep
  ▼  datasets_out/studyforrest-data-all_out/fmriprep-latest/fmriprep/.../*_desc-preproc_bold.nii.gz (+ confounds)
  │  手順1: preprocessing_align_d_studyforrest.py
  ▼  bold/*_bold.nii.gz  (confound回帰・バンドパス・zscore 済み)
  │  手順2: prepare_npz.py   ← bold_int/ を自前で作りつつ npz を書く
  ▼  roi22LR_np/*.npz
```

前提:
- 入力タスクは `ses-movie/task-movie`（視聴覚版 Forrest Gump、8 run）。`ses-forrestgump`（音声のみ版）ではない。
- 空間は `MNI152NLin2009cAsym`。TR = 2.0 s。被験者は機能データが揃う 12 名 `01,02,03,04,06,10,14,15,16,17,18,19`（05,09,20 除外）。

---

## 環境設定（毎回必須）

```bash
export PATH=/home/kazu/envs/srm_env/bin:$PATH   # 専用venv srm_env の python を使う
export MPLBACKEND=Agg                            # ★必須。無いと nltools の import が固まる
cd /home/kazu/shared_response/codes/submit_code2/voluntary_fixation/fMRI_preprocessing
```

- **`MPLBACKEND=Agg` は必須**。付けないと `nltools → nilearn.plotting → matplotlib` の対話バックエンド探索で
  import が固まる（ヘッドレスで 5 分待っても終わらない）。付ければ約 1.3 秒。
- パスは全て `paths.py` に絶対パスで集約済み。**CWD には依存しない**（どこから実行してもよい）。
  データを別の場所に移した場合だけ、`paths.py` の `REPO_ROOT` を直す。
- 対象被験者（12 名）は各スクリプトの `main()` に設定済み。特に編集は不要。

---

## 手順 0. fMRIPrep（BIDS raw → 標準空間 BOLD）

**目的**: `ses-movie/task-movie`（8 run）を前処理し、MNI 空間の `desc-preproc_bold` と confounds を得る。

**実行コマンド**:
```bash
# 必要なら先に DataLad で実体取得:
# cd /home/kazu/shared_response/datalad/DataLad-101/ds000113 && datalad get sub-*/ses-movie/func/*task-movie*

fmriprep-docker \
    /home/kazu/shared_response/datalad/DataLad-101/ds000113 \
    /home/kazu/shared_response/datasets_out/studyforrest-data-all_out/fmriprep-latest \
    participant \
    --participant-label 01 02 03 04 06 10 14 15 16 17 18 19 \
    --output-spaces MNI152NLin2009cAsym \
    --fs-license-file /path/to/license.txt
```

**できるもの**: `datasets_out/studyforrest-data-all_out/fmriprep-latest/fmriprep/sub-XX/ses-movie/func/` に
`*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz` と `*_desc-confounds_timeseries.tsv`。

> このリポジトリには fMRIPrep 実行スクリプトは無い（外部実行）。手順1 以降はこの出力を入力にする。

---

## 手順 1. denoise → `bold/`

**目的**: fMRIPrep 出力から confound を回帰除去・バンドパス（0.00667–0.1 Hz）・zscore して `bold/*.nii.gz`（96 個）を作る。
（スクリプト: `preprocessing_align_d_studyforrest.py` / 中核 `preprocessing_utils.preprocess()`）

**実行コマンド**:
```bash
python preprocessing_align_d_studyforrest.py
```

**できるもの**: `bold/sub-XX_run-Y_space-MNI152NLin2009cAsym_bold.nii.gz`（12 名 × 8 run = 96 個）。

---

## 手順 2. npz 化 → `roi22LR_np/`（bold_int も自動生成）

**目的**: `bold/` を HCP-MMP1（左右分割 44 ROI）でマスク・run×ROI 正規化・連結して SRM 入力 npz を作る。
その過程で `bold_int/`（`subXX_base.bin` / `subXX_sideLR_roi44_all.bin` / `_rois.bin`）を**自前で生成**する
（既にあれば再利用）。スクリプト: `prepare_npz.py`。

**実行コマンド**:
```bash
python prepare_npz.py --side LR --n-roi 22    # 左右分割 44 ROI → roi22LR_np/
# python prepare_npz.py --side '' --n-roi 22  # 左右まとめ 22 ROI → roi22_np/
```

**できるもの**:
- `roi22LR_np/subXX_sideLR_roi44_data-12345678.npz`（12 名分、中身は `roi_data, roi_name, n_scan_per_run, tr`）
- `bold_int/subXX_base.bin` ほか中間物（bold/ からの初回は各被験者 ~12–15 分）

> 正規化: `prepare_npz.py:133` の `if True:`（run×ROI で平均0分散1）が既定。無正規化にしたい時だけ `if False:` に。

---

## （オプション）ROI 間アラインメント解析

`align_multiple_rois_studyforrest.py` は ROI 間アラインメントの**別解析**で、`roi22LR_np` の生成には**不要**
（bold_int の中間物は手順2 が自前で作る）。実行すると `bold_int/` に `subXX_roi22_*` 系の bin を作り、
その後 ISC/アラインメント（`Pool` 並列・ヒートマップ出力）を回す。必要な時だけ:
```bash
python align_multiple_rois_studyforrest.py
```

---

## 最小レシピ

```bash
export PATH=/home/kazu/envs/srm_env/bin:$PATH
export MPLBACKEND=Agg
cd /home/kazu/shared_response/codes/submit_code2/voluntary_fixation/fMRI_preprocessing

# 手順0（fMRIPrep）は上記コマンドで別途実行しておく
python preprocessing_align_d_studyforrest.py   # 手順1: → bold/
python prepare_npz.py --side LR --n-roi 22      # 手順2: → bold_int/ + roi22LR_np/
```

依存: `nltools`, `nilearn`, `numpy`, `pandas`, `scipy`, `scikit-learn`（すべて `srm_env` に導入済み）、
HCP-MMP1 アトラス（`atlas/`）。

---

## パス設計（`paths.py`）

全スクリプトはデータ位置を `paths.py` から読む。CWD 非依存にするための集約点。

| 定数 | 値 | 用途 |
|---|---|---|
| `REPO_ROOT` | `/home/kazu/shared_response` | data/ と datasets_out/ の親。**移設時はここだけ直す** |
| `DATASETS_OUT` | `$REPO_ROOT/datasets_out` | 手順0 の出力ルート |
| `BOLD_DIR` | `.../studyforrest/bold` | 手順1 出力 / 手順2 入力 |
| `BOLD_INT_DIR` | `.../studyforrest/bold_int` | 手順2 の中間物 |
| `npz_dir(n_roi, side)` | `.../studyforrest/roi{n_roi}{side}_np` | 手順2 の npz 出力先 |

atlas パスは `atlas_definition.py` が `__file__` 起点で解決（`atlas/` はこのディレクトリ内）。

---

## 検証状況（2026-07-10）

このリポジトリ上で実測して確認済み:
- 4 スクリプト + `paths.py` は py_compile OK。3 つの異なる CWD から実行してもパスが同一絶対パスに解決（CWD 非依存）。
- atlas は `__file__` 起点で常に解決。
- **手順2 の実走で既存 `roi22LR_np` を再現**（sub-01、44 ROI・形状一致・値の最大差 9.5e-07 = float32 丸め）。
- **手順2 が空の `bold_int` から自前で中間物を構築**することを確認（bold/ の 8 run を読み、44 ROI をマスク）。
- `prepare_npz.py --side LR --n-roi 22` の CLI 起動で 12 名・全パスが正しく配線されることを確認。

未検証:
- 手順0（fMRIPrep）は docker と生データが要るため未実行。
- 手順1 は入力（fMRIPrep 出力）がローカルに無いため実走は未検証（構文・import・パスのみ確認）。
