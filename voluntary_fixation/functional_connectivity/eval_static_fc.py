

import os
import numpy as np
from typing import List, Tuple
from voluntary_fixation.envs import SUBJECT_IDS, SAVE_ROOT, NUM_ROIS
import matplotlib.pyplot as plt
import pandas as pd
from voluntary_fixation.functional_connectivity.fc_refine import fc_definition
import copy
from pingouin import partial_corr
import json
import random
from termcolor import cprint




def get_behavior(iou_q, saliency_TR_q, behavior_mode:str):
    behavior_path = os.path.join(SAVE_ROOT, 'behavior','behaviors.csv')
    behavior_df = pd.read_csv(behavior_path)
    target_df = behavior_df.query(f'iou_q=={iou_q} and saliency_TR_q=={saliency_TR_q}')
    # import pdb; pdb.set_trace()
    try:
        total = np.array([int(i) for i in target_df['total'].item().replace('[','').replace(']','').replace(',', '').split(' ')])
    except:
        print(total)
        raise ValueError('total is invalid')
    sal = np.array([int(i) for i in target_df['sal'].item().replace('[','').replace(']','').replace(',', '').split(' ')])
    # not_sal = total - sal
    not_sal = np.array([int(i) for i in target_df['not_sal'].item().replace('[','').replace(']','').replace(',', '').split(' ')]) #
    eye = np.array([int(i) for i in target_df['eye'].item().replace('[','').replace(']','').replace(',', '').split(' ')])
    joint_eye_sal = np.array([int(i) for i in target_df['joint_eye_sal'].item().replace('[','').replace(']','').replace(',', '').split(' ')])
    joint_eye_not_sal = np.array([int(i) for i in target_df['joint_eye_not_sal'].item().replace('[','').replace(']','').replace(',', '').split(' ')])
    prob_eye = eye/total
    prob_inv_eye = joint_eye_sal/sal
    prob_vol_eye = joint_eye_not_sal/not_sal

    if behavior_mode=='SSC gaze shift':
        behavior = prob_vol_eye
    elif behavior_mode=='LSC gaze shift':
        behavior = prob_inv_eye
    elif behavior_mode=='all gaze shift':
        behavior = prob_eye
    else:
        raise ValueError(f'behavior_mode {behavior_mode} is not defined')
    return behavior



def get_srm(roi, nbo):
    srm_results_dir = os.path.join( SAVE_ROOT, 'srm')
    src_roi = 36
    src_delay = -1
    tgt_rois = np.arange(NUM_ROIS)
    tgt_delay = [2] * NUM_ROIS
    srm_ircs_tests = []
    for tgt_roi, tgt_delay in zip(tgt_rois, tgt_delay):
        # print('tgt_roi', tgt_roi, tgt_delay)
        srm_ircs_train = []
        srm_ircs_test = []
        for sbj in SUBJECT_IDS:
            json_path = f'{sbj}/s{src_roi}_t{tgt_roi}/sd{src_delay}_td{tgt_delay}/isc.json'
            with open(os.path.join(srm_results_dir, json_path)) as f:
                data = json.load(f)
                srm_ircs_train.append(data['isc_train_mean'])
                srm_ircs_test.append(data['isc_test_mean'])
        srm_ircs_train = np.array(srm_ircs_train)
        srm_ircs_test = np.array(srm_ircs_test)
        srm_ircs_tests.append(srm_ircs_test)
    srm_ircs_tests = np.stack(srm_ircs_tests, axis=0) # rois x 12
    if nbo:
        srm_ircs_test = srm_ircs_tests[roi] / srm_ircs_tests.mean(axis=0)
    else:
        srm_ircs_test = srm_ircs_tests[roi]
    return srm_ircs_test[np.newaxis, :, np.newaxis]




# python voluntary_fixation/functional_connectivity/eval_static_fc.py --roi -1 -nbo -iq 0.5 -sq 0.7 --src_roi 36 -sd -1 --tgt_roi 37 -td 2
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--roi",
        nargs="*",
        required=True,
        type=int,
        help="subject_name"
    )
    parser.add_argument(
        "-tr",
        "--tgt_roi",
        nargs="*",
        required=True,
        type=int,
        help="tgt roi"
    ) # 6(L-PCL:2sec), 18(L-ACC/MPFC:-2sec), 35(R-LTL:0sec), 37(R-SPL:4sec)
    parser.add_argument(
        "-sr",
        "--src_roi",
        default=36,
        type=int,
        help="src roi"
    )
    parser.add_argument(
        "-nbo",
        "--normalize_by_other_rois",
        action='store_true',
        help="with or without normalization"
    )
    parser.add_argument(
        "-iq",
        "--iou_q",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-sq",
        "--saliency_TR_q",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-sd",
        "--src_delay",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "-td",
        "--tgt_delay",
        type=int,
        default=2,
    )
    opt = parser.parse_args()

    moving_window = -1
    grids_list = [[-1]]
    normalize_by_other_rois = opt.normalize_by_other_rois
    shuffle_times = 1000

    meaning_roi = opt.src_roi# 36 R-TPOJ
    tgt_rois = opt.tgt_roi
    if opt.roi[0] < 0:
        move_rois = list(range(NUM_ROIS))
        move_rois.remove(meaning_roi) if opt.tgt_roi[0] != meaning_roi else move_rois
    else:
        move_rois = opt.roi




    ############### FROZEN PARAMS ################
    pca = False
    n_components_bold_src = 10000
    sampling_mode = 'segment'
    frame_offset = 0
    mask_offset = 0
    iou_q = opt.iou_q
    saliency_TR_q = opt.saliency_TR_q
    ############### FROZEN PARAMS ################
    use_saliency = False
    behavior_modes = ['SSC gaze shift', 'LSC gaze shift', 'all gaze shift', 'partial_corr_SSC gaze shift', 'partial_corr_LSC gaze shift']
    corr_dict = {k:[] for k in behavior_modes}
    corr_dict['grid'] = []
    coeff_dict = {k:[] for k in behavior_modes}
    coeff_dict['grid'] = []
    rank_dict = {k:[] for k in behavior_modes}
    rank_dict['grid'] = []
    for shuffle_seed in range(10):
        np.random.seed(shuffle_seed)
        random.seed(shuffle_seed)
        for grids in grids_list:
            corr_dict['grid'].append(grids[0])
            coeff_dict['grid'].append(grids[0])
            rank_dict['grid'].append(grids[0])
            static_fcs_sbjs_grids_ = fc_definition(grids, meaning_roi, move_rois, tgt_rois,
                    n_components_bold_src, sampling_mode, pca, normalize_by_other_rois, opt.src_delay, opt.tgt_delay)
            cprint(f'FC: {static_fcs_sbjs_grids_.squeeze()}', 'green')
            for behavior_mode in behavior_modes:
                static_fcs_sbjs_grids = copy.deepcopy(static_fcs_sbjs_grids_)
                static_fcs_sbjs_grids = static_fcs_sbjs_grids[0,:,0]

                print('behavior mode:', behavior_mode)
                subdir_name = 'debug_grid_search-src{}-tgt{}'.format(meaning_roi, '_'.join([str(r) for r in move_rois]))
                if use_saliency:
                    subdir_name += '_saliency'
                else:
                    if behavior_mode=='SSC gaze shift':
                        subdir_name += '_SSC_gaze_shift'
                    elif behavior_mode=='LSC gaze shift':
                        subdir_name += '_LSC_gaze_shift'
                    elif behavior_mode=='all gaze shift':
                        subdir_name += '_all_gaze_shift'
                    elif behavior_mode=='partial_corr_SSC gaze shift':
                        subdir_name += '_partial_corr_SSC_gaze_shift'
                    elif behavior_mode=='partial_corr_LSC gaze shift':
                        subdir_name += '_partial_corr_LSC_gaze_shift'
                    else:
                        raise ValueError(f'behavior_mode {behavior_mode} is not defined')

                if 'partial_corr' in behavior_mode:
                    target_behavior_name = behavior_mode.replace('partial_corr_', '')
                    target_behavior = get_behavior(iou_q, saliency_TR_q, target_behavior_name)
                    control_behavior = get_behavior(iou_q, saliency_TR_q, 'all gaze shift')
                    df = pd.DataFrame({'fc': static_fcs_sbjs_grids, 'target': target_behavior,
                       'control': control_behavior})
                    test_corr = partial_corr(df, x='fc', y='target', covar='control')['r'].item()
                    sbj_indices = np.arange(len(SUBJECT_IDS))
                    shuffled_test_corrs = []
                    for _ in range(shuffle_times):
                        shuffle_sbj_indices = np.random.permutation(sbj_indices)
                        shuffled_target_behavior = target_behavior[shuffle_sbj_indices]
                        shuffled_control_behavior = control_behavior[shuffle_sbj_indices]
                        df = pd.DataFrame({'fc': static_fcs_sbjs_grids, 'shuffled_target': shuffled_target_behavior,
                       'shuffled_control': shuffled_control_behavior})
                        shuffled_test_corr = partial_corr(df, x='fc', y='shuffled_target', covar='shuffled_control')['r'].item()
                        shuffled_test_corrs.append(shuffled_test_corr)

                else:
                    behavior = get_behavior(iou_q, saliency_TR_q, behavior_mode)
                    # import pdb; pdb.set_trace()
                    df = pd.DataFrame({'fc': static_fcs_sbjs_grids, 'target': behavior})
                    test_corr = partial_corr(df, x='fc', y='target')['r'].item()
                    shuffled_test_corrs = []
                    sbj_indices = np.arange(len(SUBJECT_IDS))
                    for _ in range(shuffle_times):
                        shuffle_sbj_indices = np.random.permutation(sbj_indices)
                        shuffled_behavior = behavior[shuffle_sbj_indices]
                        df = pd.DataFrame({'fc': static_fcs_sbjs_grids, 'shuffled_target': shuffled_behavior})
                        shuffled_test_corr = partial_corr(df, x='fc', y='shuffled_target')['r'].item()
                        shuffled_test_corrs.append(shuffled_test_corr)


                corr_dict[behavior_mode].append(test_corr)
                surrogate_flags = [0] + [1]*len(shuffled_test_corrs)

                corrs = [test_corr] + shuffled_test_corrs
                ret = {'surrogate_flag':surrogate_flags, 'corr':corrs}
                corrs_arr = np.array(corrs)
                sorted_corrs = np.sort(corrs_arr)[::-1] # descending order
                corr_rank = np.where(sorted_corrs==test_corr)[0][0]
                print(f'============= ratio of surrogate({behavior_mode}) =============')
                print(f'\t CORR(Descending order) rank{corr_rank+1}/{len(sorted_corrs)} ratio is ', corr_rank/len(sorted_corrs))
                print('==============================================')
                rank_dict[behavior_mode].append(corr_rank+1)

    print(corr_dict)
    corr_dict = pd.DataFrame(corr_dict)
    print('all stats::::: ')
    print(corr_dict.mean())
    rank_dict = pd.DataFrame(rank_dict)
    print(rank_dict)
    print('all rank::::: ')
    print(rank_dict.mean())
    summary_dict = pd.concat([corr_dict, rank_dict], axis=1)

    # ===== save fc and behavior for figure5 =====
    figure5_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figure5')
    os.makedirs(figure5_dir, exist_ok=True)

    # FC は shuffle_seed に依存しないため、最後に計算した値をそのまま保存
    fc_per_sbj = np.asarray(static_fcs_sbjs_grids_[0, :, 0])  # (n_sbj,) src=meaning_roi -> tgt=tgt_rois[0]
    base_behavior_modes = ['SSC gaze shift', 'LSC gaze shift', 'all gaze shift']
    save_df = {'sbj': SUBJECT_IDS, 'fc': fc_per_sbj}
    for bm in base_behavior_modes:
        save_df[bm] = get_behavior(iou_q, saliency_TR_q, bm)
    save_df = pd.DataFrame(save_df)

    nbo_str = '-nbo' if normalize_by_other_rois else ''
    stem = f'src{meaning_roi}-tgt{tgt_rois[0]}-sd{opt.src_delay}-td{opt.tgt_delay}-iq{iou_q}-sq{saliency_TR_q}{nbo_str}'
    csv_path = os.path.join(figure5_dir, f'fc_behavior-{stem}.csv')
    npy_path = os.path.join(figure5_dir, f'fc-{stem}.npy')
    save_df.to_csv(csv_path, index=False)
    np.save(npy_path, fc_per_sbj)
    cprint(f'saved fc & behavior -> {csv_path}', 'cyan')
    cprint(f'saved fc (npy)      -> {npy_path}', 'cyan')