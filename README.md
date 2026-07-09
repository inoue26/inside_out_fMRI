# studyforrest_voluntary_fixation

# Data
This code requires data used in [StudyForrest](https://www.studyforrest.org/data.html)
* fMRI and eyetracking data (https://openneuro.org/datasets/ds000113/versions/1.3.0)
* Movie data (forrest_gump (format=matrosika, H264 codec, 1920x1080, 23.98 fps Duration: 02:22:09.44, bitrate: 30573 kb/s))

# Setup
1. run `pip install -r requirements.txt`
2. run `pip install -e .`

# fMRI preprocessing
Please refer [fMRI preprocessing README](voluntary_fixation/fMRI_preprocessing/README.md)

# Data Processing
1. create mask                     :: run `python voluntary_fixation/data_processing/create_mask.py` to get boolean fixation(PURS, FIXA) mask image whose size is same as movie(720,1280).
2. extract movie frames            :: run `python voluntary_fixation/data_processing/extract_frames_from_kv.py` to extract frames by TR(default=2) with offset(default=0) seconds from the movie.
3. calculate saliency              :: run `python voluntary_fixation/data_processing/create_saliency_stats.py` to get saliency(predicted by DeepGazeIIe) info. We use this information to get saliency shift.
4. calculate masked image features ::
```bash
#  to get visual features of masked images (gaze region).
python voluntary_fixation/data_processing/create_masked_visual_features.py  --modality masked_image --hidden_state  --frame_offset 0 --mask_offset 0
#  to get visual features of masked images (gaze region temporally shuffled).
python voluntary_fixation/data_processing/create_masked_visual_features.py  --modality shuffled_masked_image2 --hidden_state  --frame_offset 0 --mask_offset 0
# to get visual features of masked images (high saliency region).
python voluntary_fixation/data_processing/create_saliency_mask-control.py
python voluntary_fixation/data_processing/create_masked_visual_features.py  --modality saliency_masked_image --hidden_state  --frame_offset 0 --mask_offset 0

```
5. feature pooling
```bash
#  to average-pool masked (gaze region) visual feature vector set into a vector (length is 1408).
python voluntary_fixation/data_processing/avg_pool_masked_feature.py --modality masked_image --hidden_state --frame_offset 0 --mask_offset 0
#  to average-pool masked (gaze region temporally shuffled) visual feature vector set into a vector (length is 1408).
python voluntary_fixation/data_processing/avg_pool_masked_feature.py --modality shuffled_masked_image2 --hidden_state --frame_offset 0 --mask_offset 0
#  to average-pool masked (high saliency region) visual feature vector set into a vector (length is 1408).
python voluntary_fixation/data_processing/avg_pool_masked_feature.py --modality saliency_masked_image --hidden_state --frame_offset 0 --mask_offset 0

```
6. calculate mask iou              :: run `python voluntary_fixation/data_processing/iou_mask.py` to calculate iou of masks between continuous frames (the degree of gaze shift).
7. get distribution of saliency and gaze shift iou :: run `python voluntary_fixation/data_processing/get_grand_quantiles.py`


# BOLD2VisualFeat
1. boldからmasked (gaze region, saliency region, temporaly shuffled gaze region) 画像特徴量を予測する線形モデルの学習
```bash
modalities=(masked_image saliency_masked_image shuffled_masked_image2)
subjects=('01' '02' '03' '04' '06' '10' '14' '15' '16' '17' '18' '19')
delay=(-3 -2 -1 0 1 2 3 4 5)

for mod in "${modalities[@]}"; do
  for sub in "${subjects[@]}"; do
    for d in "${delay[@]}"; do
      python voluntary_fixation/bold2feat/train_strict_ree2.py \
        --sbj "$sub" -m "$mod" -hs -fo 0 -mo 0 -d "$d" \
        -sm segment -ncf 2 -ncb 250 -rb
    done
  done
done
```

2. testデータを使って、boldからmasked (gaze region, saliency region, temporaly shuffled gaze region) 画像特徴量を予測し、時間方向の相関を計算
```bash
python voluntary_fixation/bold2visualfeat/eval_specific_ree2.py
```

3. visualization [notebook](voluntary_fixation/figure2_and_3/visualize_time_series_specific_strict_ree2_0.5_0.7_20251029.ipynb)



# BOLD2Location
1. boldからgazeの座標を予測する線形モデルの学習
```bash
subjects=('01' '02' '03' '04' '06' '10' '14' '15' '16' '17' '18' '19')
delay=(-3 -2 -1 0 1 2 3 4 5)

for sub in "${subjects[@]}"; do
for d in "${delay[@]}"; do
    python svoluntary_fixation/bold2location/train.py \
    --sbj "$sub" -d "$d" \
    -sm segment -ncf 2 -ncb 250 -ree
done
  done
```
2. testデータを使って、boldからgazeの座標を予測し、時間方向の相関を計算
```bash
python voluntary_fixation/bold2location/eval.py
```
3. visualization [notebook](voluntary_fixation/figure4/visualize_p_time_series20251029.ipynb)



# Functional Connectivity Analysis
1. FC 計算とgaze shift frequencyとの相関計算
```bash
# R-TPOJ - L-PCL
python voluntary_fixation/functional_connectivity/eval_static_fc.py --roi -1 -nbo -iq 0.5 -sq 0.7 --src_roi 36 -sd -1 --tgt_roi 6 -td 1
# R-TPOJ - L-ACC/MPFC
python voluntary_fixation/functional_connectivity/eval_static_fc.py --roi -1 -nbo -iq 0.5 -sq 0.7 --src_roi 36 -sd -1 --tgt_roi 18 -td -1
# R-TPOJ - R-LTL
python voluntary_fixation/functional_connectivity/eval_static_fc.py --roi -1 -nbo -iq 0.5 -sq 0.7 --src_roi 36 -sd -1 --tgt_roi 35 -td 0
# R-TPOJ - R-SPL
python voluntary_fixation/functional_connectivity/eval_static_fc.py --roi -1 -nbo -iq 0.5 -sq 0.7 --src_roi 36 -sd -1 --tgt_roi 37 -td 2

```

2. visualization for correlation between FC and gaze shift frequency [notebook](submit_code2/voluntary_fixation/figure5/figure5a_fc_and_gaze_shift_freq.ipynb)

3. visualization for correlation between R-TPOJ performance and gaze shift frequency [notebook](submit_code2/voluntary_fixation/figure5/figure5b_bold2feat_score_vs_gaze_shift_freq.ipynb)


# Note
This repository is provided for reference and reproducibility of the analysis described in the paper. It cannot be run end-to-end without the movie stimuli from the StudyForrest dataset, which are subject to third-party copyright and are not redistributed here. No support or maintenance is provided.