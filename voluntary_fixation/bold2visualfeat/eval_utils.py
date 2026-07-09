import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from himalaya.scoring import correlation_score
from himalaya.backend import set_backend
import time
import torch
from voluntary_fixation.envs import TR, RUN_IDS, SUBJECT_IDS, SAVE_ROOT, NUM_ROIS, BRIGHTNESS_DIR, RUN_VOLUMES, MOVIE_FPS, MOVIE_WIDTH, MOVIE_HEIGHT, EYEMOVE_ROOT
from voluntary_fixation.dataset.bold_dataset import get_non_overlap_indices_for_concatenate_data
import pandas as pd
from voluntary_fixation.behavior.src.temporal_alignment_filtering import sacc_cnt_shift_relation_df, alignment2VOLUME
from voluntary_fixation.behavior.src.probability import count_gaze_shift_with_saliency
from voluntary_fixation.behavior.src.data_loader import read_csvs
from voluntary_fixation.utils import calculate_q
from voluntary_fixation.dataset.utils import delayed_label
from tqdm import tqdm
from scipy.stats import ttest_1samp


def get_label(iou_path:str, saliency_eyetrack_path:str, eyemovement_dir:str, saliency_dir:str, sampling_mode:str, eye_tracking_exist_flag_dir:str,
              saliency_label:str, iou_q:float, eyemovement_q:float, saliency_q:float, sbj_id:str, gaze_shift_label:List[str], delay:int,
              remove_empty_eyetrack:bool, use_large_cnt:bool=True,
              jupyter:bool=False, ret_non_overlap_indices:bool=False, high_iou:bool=False,
              saliency_TR_q:float=0.5, ret_test_length:bool=False, remove_iou_na:bool=False,
              sal_interval:float=0.1)->Tuple[List[int], List[int], List[int], List[int]]:
    # train and test
    ## get non overlap label
    non_overlap_train_indices, non_overlap_test_indices = get_non_overlap_indices_for_concatenate_data(sampling_mode, jupyter=jupyter)

    if use_large_cnt:
        lc_str = '_large'
    else:
        lc_str = ''
    # get iou
    iou_df = pd.read_csv(iou_path)
    iou_df = iou_df.dropna(axis=0)

    # iou_df.set_index('idx_in_exp')

    if os.path.exists(saliency_eyetrack_path):
        print(f'{saliency_eyetrack_path} is already exist')
        saliency_eyetrack_df = pd.read_csv(saliency_eyetrack_path)
        for label in ['max_shift_norm', 'avg_shift_norm']:
            if f'SACC_cnt_with{lc_str}_lower_sal_{label}' in saliency_eyetrack_df.columns:
                saliency_eyetrack_df = saliency_eyetrack_df.rename(columns={f'SACC_cnt_with{lc_str}_lower_sal_{label}':f'SACC_cnt{lc_str}_with_lower_sal_{label}'})
    else:
        os.makedirs(os.path.dirname(saliency_eyetrack_path), exist_ok=True)
        print('saliency_eyetrack_df is not exist. Creating...')
        saliency_eyetrack_df_list = []
        idx_offsets = np.cumsum([0] + RUN_VOLUMES)
        for run_id in tqdm(RUN_IDS):
            saliency_csv_file_path = os.path.join(saliency_dir, f'segment{run_id}.csv')
            original_eyetrack_path = os.path.join(eyemovement_dir,  f'sub-{sbj_id}/sub-{sbj_id}_task-movie_run-{run_id}_events.tsv')
            # get saliency and eyemovement
            saliency_df, original_eyetrack_df = read_csvs(saliency_csv_file_path, original_eyetrack_path,
                                                        gaze_shift_label, video_width=MOVIE_WIDTH,
                                                        video_height=MOVIE_HEIGHT, fps=MOVIE_FPS)
            original_eyetrack_df['power'] = original_eyetrack_df['diff_x']**2 + original_eyetrack_df['diff_y']**2
            # time alignment eyetracking data to saliency data
            saliency_eyetrack_df = sacc_cnt_shift_relation_df(saliency_df,
                                                            original_eyetrack_df,
                                                            gaze_shift_label,
                                                            eyemovement_q, have_duration=False)
            # cnt sal with saliency
            saliency_eyetrack_df = count_gaze_shift_with_saliency(saliency_eyetrack_df, saliency_q)
            # time alignment saliency_eyetrack_df to VOLUME
            saliency_eyetrack_df = alignment2VOLUME(saliency_eyetrack_df, run_id, agg='sum')
            assert len(saliency_eyetrack_df) == RUN_VOLUMES[run_id-1], f'len(saliency_df)={len(saliency_eyetrack_df)}, RUN_VOLUMES[run_id-1]={RUN_VOLUMES[run_id-1]}'
            idx_in_exp = list(np.arange(RUN_VOLUMES[run_id-1]) + idx_offsets[run_id-1])
            saliency_eyetrack_df['idx_in_exp'] = idx_in_exp
            saliency_eyetrack_df_list.append(saliency_eyetrack_df)
        saliency_eyetrack_df = pd.concat(saliency_eyetrack_df_list, axis=0)

        saliency_eyetrack_df.to_csv(saliency_eyetrack_path)
        print(f'saliency_eyetrack_df is saved at {saliency_eyetrack_path}')

    iou_label_th = iou_df['iou'].quantile(iou_q) # ex: 0.329526995
    if high_iou:
        small_iou_indices = iou_df[iou_df['iou'] > iou_label_th]['idx_in_exp'].to_list()
    else:
        small_iou_indices = iou_df[iou_df['iou'] < iou_label_th]['idx_in_exp'].to_list()

    saliency_th_upper = saliency_eyetrack_df[saliency_label].quantile(saliency_TR_q)  # sal_labels = ['max_shift_norm', 'avg_shift_norm']
    saliency_th_lower = saliency_eyetrack_df[saliency_label].quantile(1-saliency_TR_q)

    if remove_empty_eyetrack:
        eye_tracking_exist_flags = []
        for run_id in RUN_IDS:
            eye_tracking_exist_flag_path = os.path.join(eye_tracking_exist_flag_dir, f'fg_b_{run_id}-flag.npy')
            eye_tracking_exist_flags.append(np.load(eye_tracking_exist_flag_path))
            assert len(eye_tracking_exist_flags[-1]) == RUN_VOLUMES[run_id-1], 'eye_tracking_exist_flag:{} vs RUN_VOLUMES:{}'.format(len(eye_tracking_exist_flags[-1]), RUN_VOLUMES[run_id-1])
        eye_tracking_exist_flags = np.concatenate(eye_tracking_exist_flags, axis=0) # n_samples of bool
        non_overlap_train_indices_label = np.array([i for i in non_overlap_train_indices if eye_tracking_exist_flags[i]])
        non_overlap_test_indices_label  = np.array([i for i in non_overlap_test_indices if eye_tracking_exist_flags[i]])
    else:
        non_overlap_train_indices_label = non_overlap_train_indices
        non_overlap_test_indices_label = non_overlap_test_indices

    iou_df = pd.read_csv(iou_path)
    iou_label_th_test = iou_df.iloc[non_overlap_train_indices_label]['iou'].quantile(iou_q) # ex: 0.329526995
    ######
    tgt_indices = np.where(np.isnan(iou_df['iou']))[0][:-1]+1 # np.where(iou_df['iou']==0)[0]
    iou_df.loc[tgt_indices, ['iou']]=np.nan

    iou_label_th_test = iou_df.iloc[non_overlap_train_indices_label]['iou'].quantile(iou_q) # ex: 0.329526995

    iou_df = iou_df.dropna(axis=0)
    grand_quantile_path = os.path.join(SAVE_ROOT, 'behavior','grand_quantiles.csv')
    if jupyter:
        if not os.path.exists(grand_quantile_path):
            grand_quantile_path = '../' + grand_quantile_path
        if not os.path.exists(grand_quantile_path):
            grand_quantile_path = '../' + grand_quantile_path

    grand_quantiles = pd.read_csv(grand_quantile_path)
    target_df = grand_quantiles.query(f'iou_q=={iou_q} and saliency_TR_q=={saliency_TR_q}')
    # import pdb; pdb.set_trace()
    if saliency_TR_q >= 1:
        saliency_TR_q = 0.9
        target_df = grand_quantiles.query(f'iou_q=={iou_q} and saliency_TR_q=={saliency_TR_q}')
        true_saliency_TR_q = 1
    else:
        true_saliency_TR_q = saliency_TR_q
    if iou_q >= 1:
        iou_label_th_test = 1.1
        dummy_iou_q = 0.9
        target_df = grand_quantiles.query(f'iou_q=={dummy_iou_q} and saliency_TR_q=={saliency_TR_q}')
    else:
        iou_label_th_test = target_df['iou_th'].item() # 0.2288900255 # 0.2288900255# 0.31296276874999995 # ex: 0.329526995

    if high_iou:
        small_iou_indices = iou_df[iou_df['iou'] > iou_label_th_test]['idx_in_exp'].to_list()
    else:
        small_iou_indices = iou_df[iou_df['iou'] < iou_label_th_test]['idx_in_exp'].to_list()
    # import pdb; pdb.set_trace()
    print(f'sbj{sbj_id}  th2: ', iou_label_th_test)
    print(f'sbj{sbj_id} zero:', len(tgt_indices))

    saliency_th_upper_train = target_df['sal_upper_th'].item() # saliency_eyetrack_df.iloc[non_overlap_train_indices][saliency_label].quantile(saliency_TR_q)  # sal_labels = ['max_shift_norm', 'avg_shift_norm']
    saliency_th_lower_train =  target_df['sal_lower_th'].item() #saliency_eyetrack_df.iloc[non_overlap_train_indices][saliency_label].quantile(1-saliency_TR_q)
    saliency_th_upper_test = saliency_eyetrack_df.iloc[non_overlap_test_indices][saliency_label].quantile(saliency_TR_q)  # sal_labels = ['max_shift_norm', 'avg_shift_norm']
    saliency_th_lower_test = saliency_eyetrack_df.iloc[non_overlap_test_indices][saliency_label].quantile(1-saliency_TR_q)
    print(f'sbj{sbj_id} saliency_th_upper_train: ', saliency_th_upper_train)
    print(f'sbj{sbj_id} saliency_th_lower_train: ', saliency_th_lower_train)
    ## ２秒間でのsaliency shiftが大きいかつ(saliency shift)、iouが小さい(gaze shift)
    gaze_shift_with_saliency_indices = saliency_eyetrack_df.query(f'{saliency_label} > {saliency_th_upper_train}')['idx_in_exp'].to_list()
    # import pdb; pdb.set_trace()
    if true_saliency_TR_q == 1:
        gaze_shift_with_saliency_indices = list(small_iou_indices)
    else:
        gaze_shift_with_saliency_indices = list(set(gaze_shift_with_saliency_indices) & set(small_iou_indices))
    ## ２秒間でのsaliency shiftが小さいかつ(saliency fixed)、iouが小さい(gaze shift)
    gaze_shift_without_saliency_indices = saliency_eyetrack_df.query(f'{saliency_label} < {saliency_th_lower_train}')['idx_in_exp'].to_list()
    if true_saliency_TR_q == 1:
        gaze_shift_without_saliency_indices = list(small_iou_indices)
    else:
        gaze_shift_without_saliency_indices = list(set(gaze_shift_without_saliency_indices) & set(small_iou_indices))

    if remove_empty_eyetrack:
        eye_tracking_exist_flags = []
        for run_id in RUN_IDS:
            eye_tracking_exist_flag_path = os.path.join(eye_tracking_exist_flag_dir, f'fg_b_{run_id}-flag.npy')
            eye_tracking_exist_flags.append(np.load(eye_tracking_exist_flag_path))
            assert len(eye_tracking_exist_flags[-1]) == RUN_VOLUMES[run_id-1], 'eye_tracking_exist_flag:{} vs RUN_VOLUMES:{}'.format(len(eye_tracking_exist_flags[-1]), RUN_VOLUMES[run_id-1])
        eye_tracking_exist_flags = np.concatenate(eye_tracking_exist_flags, axis=0) # n_samples of bool
        if remove_iou_na:
            iou_exist_index = iou_df.index
            for i in range(len(eye_tracking_exist_flags)):
                if i not in iou_exist_index:
                    eye_tracking_exist_flags[i] = False
        non_overlap_train_indices = np.array([i for i in non_overlap_train_indices if eye_tracking_exist_flags[i]])
        non_overlap_test_indices  = np.array([i for i in non_overlap_test_indices if eye_tracking_exist_flags[i]])

    non_overlap_train_indices, _ = delayed_label(non_overlap_train_indices, delay)
    non_overlap_test_indices, _ = delayed_label(non_overlap_test_indices, delay)
    non_overlap_train_indices = list(non_overlap_train_indices)
    non_overlap_test_indices = list(non_overlap_test_indices)
    gaze_shift_with_saliency_indices_train = list(set(gaze_shift_with_saliency_indices) & set(non_overlap_train_indices))
    gaze_shift_with_saliency_indices_test = list(set(gaze_shift_with_saliency_indices) & set(non_overlap_test_indices))
    gaze_shift_without_saliency_indices_train = list(set(gaze_shift_without_saliency_indices) & set(non_overlap_train_indices))
    gaze_shift_without_saliency_indices_test = list(set(gaze_shift_without_saliency_indices) & set(non_overlap_test_indices))

    print('==========================================\nlabel1:: gaze shift with saliency shift and small iou')
    print(f'\t TRAIN:: {len(gaze_shift_with_saliency_indices_train)}. TEST:: {len(gaze_shift_with_saliency_indices_test)}')
    print('label2:: gaze shift without saliency shift and small iou')
    print(f'\t TRAIN:: {len(gaze_shift_without_saliency_indices_train)}. TEST:: {len(gaze_shift_without_saliency_indices_test)}')
    print('==========================================')
    w_sal_in_train = [non_overlap_train_indices.index(i) for i in gaze_shift_with_saliency_indices_train]
    w_sal_in_test = [non_overlap_test_indices.index(i) for i in gaze_shift_with_saliency_indices_test]
    wo_sal_in_train = [non_overlap_train_indices.index(i) for i in gaze_shift_without_saliency_indices_train]
    wo_sal_in_test = [non_overlap_test_indices.index(i) for i in gaze_shift_without_saliency_indices_test]
    if ret_non_overlap_indices:
        return w_sal_in_train, w_sal_in_test, wo_sal_in_train, wo_sal_in_test, non_overlap_train_indices, non_overlap_test_indices
    if ret_test_length:
        return w_sal_in_train, w_sal_in_test, wo_sal_in_train, wo_sal_in_test, len(non_overlap_test_indices)
    else:
        return w_sal_in_train, w_sal_in_test, wo_sal_in_train, wo_sal_in_test



def get_pred_and_gt(dir_path:str, roi:int, hidden_state:bool, slice_=slice(1,5,1))->Tuple[np.ndarray, np.ndarray]:
    hs_str = '-hs' if hidden_state else ''
    pred_path = os.path.join(dir_path, f'pred-roi_{roi}{hs_str}-test.npy')
    gt_path = os.path.join(dir_path, f'gt{hs_str}-test.npy')
    pred = np.load(pred_path)
    gt = np.load(gt_path)
    if hidden_state:
        pred = pred[slice_]
        gt = gt[slice_]
    return pred, gt

def statistical_test(df:pd.DataFrame, alternative, popmean=0)->pd.DataFrame:
    # ttest
    p_values = {'roi':[], 'w_sal_corr_ttest':[], 'wo_sal_corr_ttest':[], 'w_sal_corr_fdr':[], 'wo_sal_corr_fdr':[], 'wo-w_sal_corr_ttest':[], 'wo-w_sal_corr_fdr':[]}
    rois = df['roi'].unique()
    # for roi in range(NUM_ROIS):
    for roi in rois:
        p_values['roi'].append(roi)
        roi_df = df.query(f'roi=={roi}')
        assert len(roi_df) == len(SUBJECT_IDS), f'len(roi_df)={len(roi_df)}, len(SUBJECT_IDS)={len(SUBJECT_IDS)}'
        ## w_sal_corr
        ret = ttest_1samp(roi_df['w_sal_corr'], popmean=popmean, alternative=alternative)
        p_values['w_sal_corr_ttest'].append(ret.pvalue)
        ## wo_sal_corr
        ret = ttest_1samp(roi_df['wo_sal_corr'], popmean=popmean, alternative=alternative)
        p_values['wo_sal_corr_ttest'].append(ret.pvalue)
        ## wo-w_sal_corr
        ret = ttest_1samp(roi_df['wo_sal_corr']-roi_df['w_sal_corr'], popmean=popmean, alternative=alternative)
        p_values['wo-w_sal_corr_ttest'].append(ret.pvalue)

    # FDR
    ## w_sal_corr
    p_values['w_sal_corr_fdr'] = calculate_q(p_values['w_sal_corr_ttest'])
    ## wo_sal_corr
    p_values['wo_sal_corr_fdr'] = calculate_q(p_values['wo_sal_corr_ttest'])
    ## wo_sal_corr
    p_values['wo-w_sal_corr_fdr'] = calculate_q(p_values['wo-w_sal_corr_ttest'])
    return pd.DataFrame(p_values)

def plot_p_value(p_value_df:pd.DataFrame, p_criteria:float, save_path:str):
    # xs = np.arange(NUM_ROIS)
    xs = p_value_df['roi'].unique()
    ys11 = p_value_df['w_sal_corr_ttest']
    ys12 = p_value_df['wo_sal_corr_ttest']
    ys21 = p_value_df['w_sal_corr_fdr']
    ys22 = p_value_df['wo_sal_corr_fdr']
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))

    axes[0].plot(xs, ys11, 'mo-', label='w_sal_corr')
    axes[0].plot(xs, ys12, 'co-', label='wo_sal_corr')
    axes[0].hlines(p_criteria/NUM_ROIS, 0, NUM_ROIS, 'black', ':', label=f'p={p_criteria} (Bonfferoni)')
    axes[1].plot(xs, ys21, 'mo-', label='w_sal_corr')
    axes[1].plot(xs, ys22, 'co-', label='wo_sal_corr')
    axes[1].hlines(p_criteria, 0, NUM_ROIS, 'black', ':', label=f'p={p_criteria} (FDR)')
    axes[0].set_title('p_value')
    axes[1].set_title('FDR')
    axes[0].legend()
    axes[1].legend()
    axes[0].set_xlabel('roi')
    axes[1].set_xlabel('roi')
    axes[0].set_ylabel('p_value')
    axes[1].set_ylabel('FDR')
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    plt.savefig(save_path, bbox_inches='tight')

    print(f'save as {save_path}')
    pass
