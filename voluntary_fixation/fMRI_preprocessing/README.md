
### 手順 0. fMRIPrep（BIDS raw → 標準空間 BOLD）

**目的**: ds000113 の視聴覚版フォレスト・ガンプ（`ses-movie/task-movie`, 8 run）を前処理し、
MNI 標準空間の `desc-preproc_bold` と confounds を得る。

**事前に直す箇所**: なし（対象被験者と出力先パスだけ確認）。

**実行コマンド**:
```bash
# 必要なら先に DataLad で実体を取得
# cd $REPO/datalad/DataLad-101/ds000113 && datalad get sub-*/ses-movie/func/*task-movie*

fmriprep-docker \
    $REPO/datalad/DataLad-101/ds000113 \
    $REPO/datasets_out/studyforrest-data-all_out/fmriprep-latest \
    participant \
    --participant-label 01 02 03 04 06 10 14 15 16 17 18 19 \
    --output-spaces MNI152NLin2009cAsym \
    --fs-license-file /path/to/license.txt
```

**できるもの**:
`datasets_out/studyforrest-data-all_out/fmriprep-latest/fmriprep/sub-XX/ses-movie/func/`
に `..._space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz` と `..._desc-confounds_timeseries.tsv`。

---

### 手順 1. denoise → `bold/`

**目的**: fMRIPrep 出力から confound を回帰除去・バンドパス・zscore して `bold/*.nii.gz`（96 個）を作る。

**事前に直す箇所**（`codes/shared_response_model/align_fmri_old/preprocessing_align_d_studyforrest.py`）:
- `subject_ids_grouped` = 対象 12 名になっているか確認
- 出力先を `bold/` にする（下では `cd` で対応）

**実行コマンド**:
```bash
cd $REPO/data/forrestgump/studyforrest/bold          # 出力をここに落とすためカレントを bold/ に
python $REPO/voluntary_fixation/fMRI_preprocessing/preprocessing_align_d_studyforrest.py
```

**できるもの**: `bold/sub-XX_run-Y_space-MNI152NLin2009cAsym_bold.nii.gz`（12 名 × 8 run = 96 個）。

---

### 手順 2. ROI 抽出 → `bold_int/`

**目的**: `bold/` を HCP-MMP アトラスでマスクし、全脳 base と ROI 別データを pickle 化して `bold_int/*.bin` を作る。

**事前に直す箇所**（`voluntary_fixation/fMRI_preprocessingalign_multiple_rois_studyforrest.py`）:
- `main()` の `bold_base` を `bold/` の場所に
- `subs` を対象 12 名に（現状 `['01']` だけなので要変更）
- 出力 `.bin` はカレントに落ちるので、カレントを `bold_int/` にする（下で `cd`）

**実行コマンド**:
```bash
cd $REPO/data/forrestgump/studyforrest/bold_int      # 出力をここに落とす
python $REPO/voluntary_fixation/fMRI_preprocessing/align_multiple_rois_studyforrest.py
```

**できるもの**: `bold_int/subXX_base.bin`（全脳・約 3.4GB）ほか `subXX_roiNN_rois.bin` / `subXX_roiNN_all.bin`。

---

### 手順 3. npz 化 → `roi22LR_np/`

**目的**: `bold_int/` を左右分割 44 ROI（`ROIs_22_LR`）で正規化・連結し、SRM 等の入力 `.npz` を作る。

**事前に直す箇所**（`voluntary_fixation/fMRI_preprocessingprepare_npz.py`）:
- `main()` の `subs` を対象 12 名に（現状 `['01']` だけなので要変更）
- `main()` の `npz_base` を **`'../../data/forrestgump/studyforrest/roi22LR_np/'`** に変更
  （現行は `fmri/roi22LR_np_unnorm/` を指している）
- `133` 行目を **`if True:`**（run×ROI 正規化 ON。無正規化にしたい時だけ `if False:`）

**実行コマンド**:
```bash
cd $REPO/submit_code2/voluntary_fixation/fMRI_preprocessing/
python prepare_npz.py --side LR --n-roi 22
```

**できるもの**: `roi22LR_np/subXX_sideLR_roi44_data-12345678.npz`（12 名分）。
中身は `roi_data, roi_name, n_scan_per_run, tr` の 4 配列。

> 補足: `--side '' --n-roi 22` にすると左右まとめ 22 ROI 版になり、`roi22_np/subXX_side_roi22_data-12345678.npz` ができる（`npz_base` も `roi22_np/` に合わせること）。

