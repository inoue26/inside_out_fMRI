import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from himalaya.scoring import correlation_score
from himalaya.backend import set_backend
from voluntary_fixation.envs import TR, SUBJECT_IDS, SAVE_ROOT, NUM_ROIS, EYEMOVE_ROOT

import pandas as pd
from voluntary_fixation.bold2visualfeat.eval_utils import get_label, get_pred_and_gt, statistical_test, plot_p_value



def run(delay:int, iou_q:float, eyemovement_q:float, frame_offset:float,
        saliency_q:float, gaze_shift_label:List[str], sampling_mode:str, saliency_label:str,
        n_components_bold:int, hidden_state:bool, p_criteria:float, use_large_cnt:bool, rm_empty_eyetrack:bool=False,
        saliency_TR_q:float=0.5, scatter_plot:bool=False, savedir:str=None):
    rs_str = '-rm_empty_eyetrack' if rm_empty_eyetrack else ''
    saliency_dir = os.path.join(SAVE_ROOT, 'saliency', 'deepgaze2e_predict')
    ret = {'sbj':[], 'roi':[], 'w_sal_corr':[], 'wo_sal_corr':[]}
    for sbj in SUBJECT_IDS:
        iou_path = os.path.join(SAVE_ROOT, 'mask', f'resolution{TR}-start{frame_offset}', 'iou',
                                f'sub-{sbj}.csv')
        pred_and_gt_dir = os.path.join(SAVE_ROOT, f'bold2location', f'delay{delay}-{"_".join(gaze_shift_label)}',
                                       f'{sampling_mode}-ncb{n_components_bold}{rs_str}', sbj)

        saliency_eyetrack_path = os.path.join(SAVE_ROOT, 'behavior', f'saliency_eyetrack_TR{TR}',
                                              f'{sbj}-{"_".join(gaze_shift_label)}_q{eyemovement_q}-sal_q{saliency_q}.csv')

        eye_tracking_exist_flag_dir = os.path.join(SAVE_ROOT, 'features', f'pooled_masked_image-tr{TR}-fo{frame_offset}-mo{mask_offset}', sbj)
        w_sal_in_train, w_sal_in_test, wo_sal_in_train, wo_sal_in_test = get_label(iou_path, saliency_eyetrack_path, EYEMOVE_ROOT, saliency_dir,
                                                                                   sampling_mode, eye_tracking_exist_flag_dir, saliency_label,
                                                                                    iou_q, eyemovement_q, saliency_q, sbj, gaze_shift_label,
                                                                                    delay, rm_empty_eyetrack, use_large_cnt,
                                                                                    saliency_TR_q=saliency_TR_q, high_iou=False)
        # 現状の設定だとw_sal_in_test+wo_sal_in_testのuniqueはindices_in_testと変わらないはず

        for roi in range(NUM_ROIS):
            ret['sbj'].append(sbj)
            ret['roi'].append(roi)
            pred, gt = get_pred_and_gt(pred_and_gt_dir, roi, hidden_state)

            if label_mode == 12:
                assert len(w_sal_in_test) + len(wo_sal_in_test) == len(pred)
            # pred, gt: n_samples x n_components
            # w_sal
            w_sal_score = correlation_score(pred[w_sal_in_test,:], gt[w_sal_in_test,:])
            # wo_sal
            wo_sal_score = correlation_score(pred[wo_sal_in_test,:], gt[wo_sal_in_test,:])
            ret['w_sal_corr'].append(np.mean(w_sal_score))
            ret['wo_sal_corr'].append(np.mean(wo_sal_score))
            if scatter_plot:
                fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(10,5))
                ax1.scatter(pred[w_sal_in_test,:], gt[w_sal_in_test,:])
                ax2.scatter(pred[wo_sal_in_test,:], gt[wo_sal_in_test,:])
                ax1.set_title('w_sal')
                ax2.set_title('wo_sal')
                plt.savefig(pred_and_gt_dir, 'scatters', f'{roi}.png')
                plt.close()


    ret_df = pd.DataFrame(ret)
    p_value_df = statistical_test(ret_df, alternative='greater')

    if savedir is not None:
        savedir = os.path.join(savedir, f'delay{delay}')
        os.makedirs(savedir, exist_ok=True)
        csv_path1 = os.path.join(savedir,  f'statistics_test-{"_".join(gaze_shift_label)}_q{eyemovement_q}-sal_q{saliency_q}.csv')
        p_value_df.to_csv(csv_path1)
        csv_path2 = os.path.join(savedir,  f'raw_corr-{"_".join(gaze_shift_label)}_q{eyemovement_q}-sal_q{saliency_q}.csv')
        ret_df.to_csv(csv_path2)
        print('raw_corr is saved at ', csv_path2)
        save_path = os.path.join(savedir,  f'statistics_test-{"_".join(gaze_shift_label)}_q{eyemovement_q}-sal_q{saliency_q}.png')
        # plot_p_value(p_value_df, p_criteria=p_criteria, save_path=save_path)

    else:
        csv_path1 = os.path.join(*(pred_and_gt_dir.split('/')[:-1]),  f'statistics_test-{"_".join(gaze_shift_label)}_q{eyemovement_q}-sal_q{saliency_q}.csv')
        p_value_df.to_csv(csv_path1)
        csv_path2 = os.path.join(*(pred_and_gt_dir.split('/')[:-1]),  f'raw_corr-{"_".join(gaze_shift_label)}_q{eyemovement_q}-sal_q{saliency_q}.csv')
        ret_df.to_csv(csv_path2)
        print('raw_corr is saved at ', csv_path2)
        save_path = os.path.join(*(pred_and_gt_dir.split('/')[:-1]),  f'statistics_test-{"_".join(gaze_shift_label)}_q{eyemovement_q}-sal_q{saliency_q}.png')
        # plot_p_value(p_value_df, p_criteria=p_criteria, save_path=save_path)

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
    print(ret_df.groupby('roi').agg(['mean', 'std']))
    print('====================STATS=================')
    print('\t FDR_significant_roi_w_sal         ::', FDR_significant_roi_w_sal)
    print('\t FDR_significant_roi_wo_sal        ::', FDR_significant_roi_wo_sal)
    print('\t Bonfferoni_significant_roi_w_sal  ::', Bonfferoni_significant_roi_w_sal)
    print('\t Bonfferoni_significant_roi_wo_sal ::', Bonfferoni_significant_roi_wo_sal)

    print(p_value_df)

if __name__ == '__main__':
    gaze_shift_label = ['FIXA', 'PURS'] # ['FIXA', 'PURS']  # ['SACC'] ['FIXA', 'PURS', 'SACC']
    frame_offset = 0
    mask_offset = 0
    eyemovement_q = 0.9
    saliency_q = 0.9
    sampling_mode = 'segment' # session, sandwitch
    saliency_label = 'avg_shift_norm' # 'avg_shift_norm' 'max_shift_norm'
    n_components_bold = 250 # 10000
    rm_empty_eyetrack = True # False
    hidden_state = False
    p_criteria = 0.05# 0.05
    use_large_cnt = False # if False, eyemovement_q is ignored.
    label_mode = 'public'
    saliency_TR_q = 0.7
    iou_q = 0.5

    savedir = os.path.join(SAVE_ROOT, 'bold2location', f'eval_specific{label_mode}')
    os.makedirs(savedir, exist_ok=True)
    for delay in [-3,-2,-1,0,1,2,3,4,5]:
        run(delay, iou_q, eyemovement_q, frame_offset, saliency_q, gaze_shift_label,
            sampling_mode, saliency_label, n_components_bold, hidden_state, p_criteria,
            use_large_cnt, rm_empty_eyetrack, saliency_TR_q=saliency_TR_q, savedir=savedir)
