# studyforrest_voluntary_fixation

Analysis code accompanying the paper on voluntary vs. involuntary fixation during
naturalistic movie viewing, using the [StudyForrest](https://www.studyforrest.org/data.html)
fMRI + eye-tracking dataset.

This repository is provided for **reference and reproducibility** of the analyses and
figures reported in the paper. See [Notes on reproducibility](#notes-on-reproducibility)
before running anything.


## Data

This code requires data from [StudyForrest](https://www.studyforrest.org/data.html):

* fMRI and eye-tracking data — ds000113 (https://openneuro.org/datasets/ds000113/versions/1.3.0)
* Movie stimulus — *Forrest Gump* (Matroska container, H.264, 1920×1080, 23.98 fps,
  duration 02:22:09.44, ~30573 kb/s). The movie is **not** redistributed here (third-party copyright).


## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

All analysis scripts are meant to be launched **from the repository root**
(`submit_code2/`), so that the relative paths defined in
[`voluntary_fixation/envs.py`](voluntary_fixation/envs.py) resolve correctly.


## Data placement

Every input/output location is configured in
[`voluntary_fixation/envs.py`](voluntary_fixation/envs.py) as a path relative to the
repository root (all begin with `../../`). Concretely, the scripts expect a *data root*
**two levels above** this repository, laid out as follows:

```
<data_root>/                          # = repo's ../../ ; edit envs.py to relocate
├── codes/submit_code2/               # this repository
├── original_movies/                  # SEGMENT_MOVIE_DIR : Forrest Gump movie segments (.mkv)
├── datalad/DataLad-101/studyforrest-data-confoundsannotation/annotation/visual/
│                                     # BRIGHTNESS_DIR    : per-frame brightness annotations (.tsv)
├── data/forrestgump/studyforrest/
│   ├── roi22LR_np/                   # BOLD44_ROOT       : SRM-input npz (see fMRI preprocessing)
│   └── annot/studyforrest-data-eyemovementlabels/   # EYEMOVE_ROOT : eye-movement labels
└── results/voluntary_fixation/       # SAVE_ROOT         : all intermediate + final outputs
```

Notes:

* The relevant keys in `envs.py` are `SAVE_ROOT`, `BOLD44_ROOT`, `SEGMENT_MOVIE_DIR`,
  `BRIGHTNESS_DIR`, and `EYEMOVE_ROOT`. Change them if your data lives elsewhere.
* Analysis scripts run from the repo root, so `../../...` points at `<data_root>`.
  The plotting notebooks live two directories deeper (e.g. `voluntary_fixation/figure5/`)
  and therefore prepend extra `../` to `SAVE_ROOT` at the top of the notebook — this is
  intentional and keeps both resolving to the same `<data_root>/results/...`.
* The 12 subjects with complete functional data are used throughout:
  `01, 02, 03, 04, 06, 10, 14, 15, 16, 17, 18, 19` (05, 09, 20 are excluded).


## fMRI preprocessing

Produces the SRM-input npz files in `data/forrestgump/studyforrest/roi22LR_np/` — i.e. the
`BOLD44_ROOT` that the analyses below consume. Full steps are in the
[fMRI preprocessing README](voluntary_fixation/fMRI_preprocessing/README.en.md)
(BIDS raw → fMRIPrep → confound regression → ROI npz).

Two things that section adds on top of this one:

* **Its own path config.** The preprocessing scripts read locations from
  [`voluntary_fixation/fMRI_preprocessing/paths.py`](voluntary_fixation/fMRI_preprocessing/paths.py)
  (absolute `REPO_ROOT`), *not* from `envs.py`. `REPO_ROOT` must point at the **same data root**
  as `envs.py`'s `../../` (here, `/home/kazu/shared_response`). If you relocate the data, edit
  **both** `envs.py` and `paths.py`.
* **Runtime requirement.** Run the preprocessing with `export MPLBACKEND=Agg` set — without it the
  `nltools` import hangs (interactive-backend probing). See that README for the exact recipe.


## Data processing

1. **Create mask** — `python voluntary_fixation/data_processing/create_mask.py`
   Boolean fixation (PURS, FIXA) mask images matching the movie resolution (720×1280).
2. **Extract movie frames** — `python voluntary_fixation/data_processing/extract_frames_from_mkv.py`
   Extract one frame per TR (default 2 s) with an offset (default 0 s) from the movie.
3. **Compute saliency** — `python voluntary_fixation/data_processing/create_saliency_stats.py`
   Saliency predicted by DeepGazeIIe, used to derive the *saliency shift*.
4. **Masked image features**
   ```bash
   # visual features of masked images (gaze region)
   python voluntary_fixation/data_processing/create_masked_visual_features.py --modality masked_image --hidden_state --frame_offset 0 --mask_offset 0
   # visual features of masked images (gaze region, temporally shuffled)
   python voluntary_fixation/data_processing/create_masked_visual_features.py --modality shuffled_masked_image2 --hidden_state --frame_offset 0 --mask_offset 0
   # visual features of masked images (high-saliency region)
   python voluntary_fixation/data_processing/create_saliency_mask-control.py
   python voluntary_fixation/data_processing/create_masked_visual_features.py --modality saliency_masked_image --hidden_state --frame_offset 0 --mask_offset 0
   ```
5. **Feature pooling** (average-pool each masked feature set into a single 1408-d vector)
   ```bash
   python voluntary_fixation/data_processing/avg_pool_masked_feature.py --modality masked_image --hidden_state --frame_offset 0 --mask_offset 0
   python voluntary_fixation/data_processing/avg_pool_masked_feature.py --modality shuffled_masked_image2 --hidden_state --frame_offset 0 --mask_offset 0
   python voluntary_fixation/data_processing/avg_pool_masked_feature.py --modality saliency_masked_image --hidden_state --frame_offset 0 --mask_offset 0
   ```
6. **Mask IoU** — `python voluntary_fixation/data_processing/iou_mask.py`
   IoU of masks between consecutive frames (the degree of gaze shift).
7. **Distributions of saliency / gaze-shift IoU** — `python voluntary_fixation/data_processing/get_grand_quantiles.py`
   Writes `results/voluntary_fixation/behavior/grand_quantiles.csv` and `behaviors.csv`.
   `behaviors.csv` is consumed directly by the FC evaluation and the Figure 5 notebooks.
   > **Prerequisite / ordering.** This step reads (but does **not** create) the per-subject
   > `behavior/saliency_eyetrack_TR2/*.csv` intermediates. Those are generated on demand by
   > `eval_utils.get_label`, i.e. as a side effect of the **BOLD2VisualFeat evaluation below**
   > (`eval_specific_ree2.py`). Run that evaluation **before** this step on a fresh dataset,
   > otherwise `get_grand_quantiles.py` fails with `FileNotFoundError`.


## BOLD → visual feature (BOLD2VisualFeat)

1. Train linear models predicting masked (gaze / saliency / temporally-shuffled gaze)
   image features from BOLD:
   ```bash
   modalities=(masked_image saliency_masked_image shuffled_masked_image2)
   subjects=('01' '02' '03' '04' '06' '10' '14' '15' '16' '17' '18' '19')
   delay=(-3 -2 -1 0 1 2 3 4 5)

   for mod in "${modalities[@]}"; do
     for sub in "${subjects[@]}"; do
       for d in "${delay[@]}"; do
         python voluntary_fixation/bold2visualfeat/train_strict_ree2.py \
           --sbj "$sub" -m "$mod" -hs -fo 0 -mo 0 -d "$d" \
           -sm segment -ncf 2 -ncb 250 -rb
       done
     done
   done
   ```

2. On the test set, predict the image features from BOLD and compute the temporal
   correlation (writes to `results/voluntary_fixation/bold2feat_control2_strict_ree2/eval_specificpublic/`):
   ```bash
   python voluntary_fixation/bold2visualfeat/eval_specific_ree2.py
   ```
   > This evaluation reads the trained models from
   > `bold2feat_strict_ree2/delay{d}-fo0-mo0/pca2-{modality}-segment-.../{sbj}/`. That path is
   > produced by step 1 **only when `-ncb 250`** (the value above); other `-ncb` values write to
   > an extra `ncb{N}/` sub-directory that this evaluation does not read. It also creates the
   > `behavior/saliency_eyetrack_TR2/*.csv` intermediates required by data-processing step 7.

3. Visualization → [notebook](voluntary_fixation/figure2_and_3/visualize_time_series_specific_strict_ree2_0.5_0.7.ipynb)


## BOLD → gaze location (BOLD2Location)

1. Train linear models predicting gaze coordinates from BOLD:
   ```bash
   subjects=('01' '02' '03' '04' '06' '10' '14' '15' '16' '17' '18' '19')
   delay=(-3 -2 -1 0 1 2 3 4 5)

   for sub in "${subjects[@]}"; do
     for d in "${delay[@]}"; do
       python voluntary_fixation/bold2location/train.py \
         --sbj "$sub" -d "$d" \
         -sm segment -ncf 2 -ncb 250 -ree
     done
   done
   ```
2. On the test set, predict gaze coordinates and compute the temporal correlation:
   ```bash
   python voluntary_fixation/bold2location/eval.py
   ```
3. Visualization → [notebook](voluntary_fixation/figure4/visualize_p_time_series20251029.ipynb)


## Functional connectivity analysis

1. Compute FC and its correlation with gaze-shift frequency:
   ```bash
   # R-TPOJ – L-PCL
   python voluntary_fixation/functional_connectivity/eval_static_fc.py --roi -1 -nbo -iq 0.5 -sq 0.7 --src_roi 36 -sd -1 --tgt_roi 6 -td 1
   # R-TPOJ – L-ACC/MPFC
   python voluntary_fixation/functional_connectivity/eval_static_fc.py --roi -1 -nbo -iq 0.5 -sq 0.7 --src_roi 36 -sd -1 --tgt_roi 18 -td -1
   # R-TPOJ – R-LTL
   python voluntary_fixation/functional_connectivity/eval_static_fc.py --roi -1 -nbo -iq 0.5 -sq 0.7 --src_roi 36 -sd -1 --tgt_roi 35 -td 0
   # R-TPOJ – R-SPL
   python voluntary_fixation/functional_connectivity/eval_static_fc.py --roi -1 -nbo -iq 0.5 -sq 0.7 --src_roi 36 -sd -1 --tgt_roi 37 -td 2
   ```
2. Correlation between FC and gaze-shift frequency →
   [notebook](voluntary_fixation/figure5/figure5a_fc_and_gaze_shift_freq.ipynb)
3. Correlation between R-TPOJ decoding performance and gaze-shift frequency →
   [notebook](voluntary_fixation/figure5/figure5b_bold2feat_score_vs_gaze_shift_freq.ipynb)


## Supplementary figures (Fig. S2)

- **Relative self-reconstruction error (`RSSE.ipynb`)** needs an extra evaluation step that the main
  pipeline does not run. First produce the RSSE outputs, then open the notebook:
  ```bash
  python voluntary_fixation/figureS2/eval_strict_ree2.py   # defaults: --iou_q 0.5 --saliency_TR_q 0.7
  ```
  This writes `results/voluntary_fixation/bold2feat_control_RSSE_strict_ree2/{evalpublic,infos}/`,
  which [RSSE.ipynb](voluntary_fixation/figureS2/RSSE.ipynb) reads. Like the other evaluations it
  consumes the BOLD2VisualFeat training outputs (`bold2feat_strict_ree2/…`, step 1 above with
  `-ncb 250`) and the data-processing intermediates, so run those first.
- **Visual-feature autocorrelation**
  ([visual_feat_auto_correlation.ipynb](voluntary_fixation/figureS2/visual_feat_auto_correlation.ipynb))
  needs no extra step — it reads the standard BOLD2VisualFeat training outputs and the
  data-processing intermediates directly.


## Figures index

| Figure | Notebook |
| --- | --- |
| Fig. 1 | [check_iou_and_saliency_time_series.ipynb](voluntary_fixation/figure1/check_iou_and_saliency_time_series.ipynb), [check_NSS_score.ipynb](voluntary_fixation/figure1/check_NSS_score.ipynb) |
| Fig. 2 & 3 | [visualize_time_series_specific_strict_ree2_0.5_0.7.ipynb](voluntary_fixation/figure2_and_3/visualize_time_series_specific_strict_ree2_0.5_0.7.ipynb) |
| Fig. 4 | [visualize_p_time_series20251029.ipynb](voluntary_fixation/figure4/visualize_p_time_series20251029.ipynb) |
| Fig. 5a | [figure5a_fc_and_gaze_shift_freq.ipynb](voluntary_fixation/figure5/figure5a_fc_and_gaze_shift_freq.ipynb) |
| Fig. 5b | [figure5b_bold2feat_score_vs_gaze_shift_freq.ipynb](voluntary_fixation/figure5/figure5b_bold2feat_score_vs_gaze_shift_freq.ipynb) |
| Fig. S2 | [RSSE.ipynb](voluntary_fixation/figureS2/RSSE.ipynb), [visual_feat_auto_correlation.ipynb](voluntary_fixation/figureS2/visual_feat_auto_correlation.ipynb) |

The Figure 5 notebooks write their PDFs to a `figures/` folder next to the notebook
(created automatically on first save).


## Notes on reproducibility

* This repository documents the analysis pipeline reported in the paper. It **cannot be
  run end-to-end without the movie stimuli** from the StudyForrest dataset, which are
  subject to third-party copyright and are not redistributed here.
* The pipeline is staged: each step consumes the outputs of the previous step under
  `SAVE_ROOT` (`results/voluntary_fixation/`). Given correctly placed data, each stage —
  and each figure notebook — runs independently.
* No support or maintenance is provided.
