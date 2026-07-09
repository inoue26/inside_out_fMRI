import os
import numpy as np
from typing import List, Tuple
from himalaya.scoring import correlation_score
from voluntary_fixation.envs import TR, SUBJECT_IDS, SAVE_ROOT, NUM_ROIS, EYEMOVE_ROOT, TMP_SAVE_ROOT
import pandas as pd

from voluntary_fixation.bold2visualfeat.eval_utils import get_label, get_pred_and_gt, statistical_test
from sklearn.decomposition import PCA



def run(delay:int, iou_q:float, eyemovement_q:float, frame_offset:float, mask_offset:float,
        saliency_q:float, gaze_shift_label:List[str], sampling_mode:str, saliency_label:str,
        remove_brightness:bool, remove_empty_eyetrack:bool, n_components:int, modality:str,
        hidden_state:bool, p_criteria:float,
        use_large_cnt:bool, slice_:slice=slice(1,5,1), saliency_TR_q:float=0.5, savedir:str=None):
    remove_str = ''
    if remove_brightness:
        remove_str += '-remove_brightness'
    if remove_empty_eyetrack:
        remove_str += '-remove_empty_eyetrack'

    saliency_dir = os.path.join(SAVE_ROOT, 'saliency', 'deepgaze2e_predict')
    ret = {'sbj':[], 'roi':[], 'w_sal_corr':[], 'wo_sal_corr':[]}
    for sbj in SUBJECT_IDS:
        iou_path = os.path.join(SAVE_ROOT, 'mask', f'resolution{TR}-start{frame_offset}', 'iou',
                                f'sub-{sbj}.csv')
        pred_and_gt_dir = os.path.join(SAVE_ROOT, 'bold2feat_strict_ree2', f'delay{delay}-fo{frame_offset}-mo{mask_offset}',
                                       f'pca{n_components}-{modality}-{sampling_mode}{remove_str}', sbj)

        saliency_eyetrack_path = os.path.join(SAVE_ROOT, 'behavior', f'saliency_eyetrack_TR{TR}',
                                              f'{sbj}-{"_".join(gaze_shift_label)}_q{eyemovement_q}-sal_q{saliency_q}.csv')
        eye_tracking_exist_flag_dir = os.path.join(SAVE_ROOT, 'features', f'pooled_{modality}-tr{TR}-fo{frame_offset}-mo{mask_offset}', sbj)
        w_sal_in_train, w_sal_in_test, wo_sal_in_train, wo_sal_in_test = get_label(iou_path, saliency_eyetrack_path, EYEMOVE_ROOT, saliency_dir, sampling_mode, eye_tracking_exist_flag_dir,
                                                                                    saliency_label,
                                                                                    iou_q, eyemovement_q, saliency_q, sbj, gaze_shift_label, delay,
                                                                                    remove_empty_eyetrack, use_large_cnt=use_large_cnt,
                                                                                    saliency_TR_q=saliency_TR_q)


        for roi in range(NUM_ROIS):
            ret['sbj'].append(sbj)
            ret['roi'].append(roi)
            pred, gt = get_pred_and_gt(pred_and_gt_dir, roi, hidden_state, slice_=slice_)
            assert pred.shape[2] == n_components, f'pred.shape[1]={pred.shape[2]}, n_components={n_components}'
            n_hidden_layers = len(pred)
            # pred, gt: num_hidden_layer x n_samples x n_components -> n_samples x (num_hidden_layer * n_components)
            pred = np.transpose(pred, (1, 0, 2)).reshape(-1, n_components*n_hidden_layers)
            gt = np.transpose(gt, (1, 0, 2)).reshape(-1, n_components*n_hidden_layers)
            # w_sal
            w_sal_score = correlation_score(pred[w_sal_in_test,:], gt[w_sal_in_test,:])
            # wo_sal
            wo_sal_score = correlation_score(pred[wo_sal_in_test,:], gt[wo_sal_in_test,:])
            ret['w_sal_corr'].append(np.mean(w_sal_score))
            ret['wo_sal_corr'].append(np.mean(wo_sal_score))
            # print(f'sbj::{sbj}, roi::{roi}, w_sal::{w_sal_score}, wo_sal::{wo_sal_score}')
    ret_df = pd.DataFrame(ret)
    p_value_df = statistical_test(ret_df, alternative='greater')
    if savedir is not None:
        csv_path1 = os.path.join(savedir,  f'statistics_test-{"_".join(gaze_shift_label)}_q{eyemovement_q}-sal_q{saliency_q}-iou_q{iou_q}-layer{"_".join(["1","10","20","30","40"][slice_])}.csv')
        p_value_df.to_csv(csv_path1)
        csv_path2 = os.path.join(savedir,  f'raw_corr-{"_".join(gaze_shift_label)}_q{eyemovement_q}-sal_q{saliency_q}-iou_q{iou_q}-layer{"_".join(["1","10","20","30","40"][slice_])}.csv')
        ret_df.to_csv(csv_path2)
        print('raw_corr is saved at ', csv_path2)
    else:
        csv_path1 = os.path.join(*(pred_and_gt_dir.split('/')[:-1]),  f'statistics_test-{"_".join(gaze_shift_label)}_q{eyemovement_q}-sal_q{saliency_q}-iou_q{iou_q}-layer{"_".join(["1","10","20","30","40"][slice_])}.csv')
        p_value_df.to_csv(csv_path1)
        csv_path2 = os.path.join(*(pred_and_gt_dir.split('/')[:-1]),  f'raw_corr-{"_".join(gaze_shift_label)}_q{eyemovement_q}-sal_q{saliency_q}-iou_q{iou_q}-layer{"_".join(["1","10","20","30","40"][slice_])}.csv')
        ret_df.to_csv(csv_path2)
        print('raw_corr is saved at ', csv_path2)

    FDR_significant_roi_w_sal = []
    FDR_significant_roi_wo_sal = []
    Bonfferoni_significant_roi_w_sal = []
    Bonfferoni_significant_roi_wo_sal = []
    for i, row in p_value_df.iterrows():
        if row['w_sal_corr_fdr'] < p_criteria:
            if ret_df.query(f'roi=={int(row["roi"])}')['w_sal_corr'].mean() > 0:
                FDR_significant_roi_w_sal.append(int(row['roi']))
        if row['wo_sal_corr_fdr'] < p_criteria:
            if ret_df.query(f'roi=={int(row["roi"])}')['wo_sal_corr'].mean() > 0:
                FDR_significant_roi_wo_sal.append(int(row['roi']))
        if row['w_sal_corr_ttest'] < p_criteria/NUM_ROIS:
            if ret_df.query(f'roi=={int(row["roi"])}')['w_sal_corr'].mean() > 0:
                Bonfferoni_significant_roi_w_sal.append(int(row['roi']))
        if row['wo_sal_corr_ttest'] < p_criteria/NUM_ROIS:
            if ret_df.query(f'roi=={int(row["roi"])}')['wo_sal_corr'].mean() > 0:
                Bonfferoni_significant_roi_wo_sal.append(int(row['roi']))
    print(ret_df.groupby('roi').mean())
    print(p_value_df)
    print('====================STATS=================')
    print('\t FDR_significant_roi_w_sal         ::', FDR_significant_roi_w_sal)
    print('\t FDR_significant_roi_wo_sal        ::', FDR_significant_roi_wo_sal)
    print('\t Bonfferoni_significant_roi_w_sal  ::', Bonfferoni_significant_roi_w_sal)
    print('\t Bonfferoni_significant_roi_wo_sal ::', Bonfferoni_significant_roi_wo_sal)
    return ret_df, p_value_df


if __name__ == '__main__':
    gaze_shift_label = ['FIXA', 'PURS']
    frame_offset = 0
    mask_offset = 0
    eyemovement_q = 0.9
    saliency_q = 0.9
    sampling_mode = 'segment' # session, sandwitch
    saliency_label = 'avg_shift_norm' # 'avg_shift_norm' 'max_shift_norm'
    remove_brightness = True# True
    remove_empty_eyetrack = True # True
    n_components = 2
    hidden_state = True
    p_criteria = 0.05
    use_large_cnt = False

    label_mode = 'public'
    slice_ = slice(4,5,1)
    iou_q = 0.5
    saliency_TR_q = 0.7

    remove_str = ''
    if remove_brightness:
        remove_str += '-remove_brightness'
    if remove_empty_eyetrack:
        remove_str += '-remove_empty_eyetrack'

    remove_str += f'{iou_q}_{saliency_TR_q}'
    savedir = os.path.join(SAVE_ROOT, 'bold2feat_control2_strict_ree2', f'eval_specific{label_mode}', remove_str)
    os.makedirs(savedir, exist_ok=True)
    for modality in ['masked_image', 'saliency_masked_image', 'shuffled_masked_image2']:
        print('control_modality:: ', modality)
        for delay in [-3, -2, -1, 0, 1, 2, 3, 4, 5]: # [-3, -2, -1, 0, 1, 2, 3, 4, 5]:
            ret_df, p_value_df = run(delay, iou_q, eyemovement_q, frame_offset, mask_offset, saliency_q, gaze_shift_label,
                                    sampling_mode, saliency_label, remove_brightness, remove_empty_eyetrack, n_components, modality,
                                    hidden_state, p_criteria, use_large_cnt, slice_, saliency_TR_q=saliency_TR_q, savedir=savedir)
            p_value_savepath = os.path.join(savedir, f'p_value-delay{delay}-{modality}-by-{modality}-layer{"_".join(["1","10","20","30","40"][slice_])}.csv')
            raw_value_savepath = os.path.join(savedir, f'raw_value-delay{delay}-{modality}-by-{modality}-layer{"_".join(["1","10","20","30","40"][slice_])}.csv')
            ret_df.to_csv(raw_value_savepath)
            p_value_df.to_csv(p_value_savepath)
            print('p_value is saved at ', p_value_savepath)
            print('raw_value is saved at ', raw_value_savepath)
    print('target hidden layer :: ', [1,10,20,30,40][slice_])
