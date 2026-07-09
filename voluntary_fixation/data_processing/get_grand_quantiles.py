import os
from voluntary_fixation.envs import TR, RUN_IDS, SUBJECT_IDS, SAVE_ROOT, RUN_VOLUMES, TMP_SAVE_ROOT
from voluntary_fixation.dataset.bold_dataset import get_non_overlap_indices_for_concatenate_data
import pandas as pd
import numpy as np

############## FROZEN PARAMETERS ##############
gaze_shift_label = ['FIXA', 'PURS'] # ['FIXA', 'PURS']  # ['SACC'] ['FIXA', 'PURS', 'SACC']
frame_offset = 0
mask_offset = 0
eyemovement_q = 0.5
saliency_q = 0.5
modality = 'masked_image'
sampling_mode = 'segment'
jupyter = False
saliency_label = 'avg_shift_norm'
###############################################

iou_q = 0.5
saliency_TR_q = 0.7
qs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


grand_quantiles = {'iou_q':[], 'saliency_TR_q':[], 'iou_th':[], 'sal_upper_th':[], 'sal_lower_th':[]}
behaviors = {'iou_q':[], 'saliency_TR_q':[], 'joint_eye_sal':[], 'joint_eye_not_sal':[], 'eye':[], 'sal':[], 'not_sal':[], 'total':[]}
sal_indices = {sbj: {q :[] for q in qs} for sbj in SUBJECT_IDS}
eye_sal_indices = {sbj: {q :[] for q in qs} for sbj in SUBJECT_IDS}
for sbj in SUBJECT_IDS:
    sal_indices[sbj][0] = set()
    eye_sal_indices[sbj][0] = set()
for iou_q in qs:
    for saliency_TR_q in qs:
        iou_th_list = []
        sal_upper_th_list = []
        sal_lower_th_list = []

        joint_eye_sal_list = []
        joint_eye_not_sal_list = []
        eye_list = []
        sal_list = []
        not_sal_list = []
        total_list = []

        for sbj in SUBJECT_IDS:
            saliency_dir = os.path.join(SAVE_ROOT, 'saliency', 'deepgaze2e_predict')
            iou_path = os.path.join(SAVE_ROOT, 'mask', f'resolution{TR}-start{frame_offset}', 'iou', f'sub-{sbj}.csv')
            saliency_eyetrack_path = os.path.join(SAVE_ROOT, 'behavior', f'saliency_eyetrack_TR{TR}',
                                                    f'{sbj}-{"_".join(gaze_shift_label)}_q{eyemovement_q}-sal_q{saliency_q}.csv')
            eye_tracking_exist_flag_dir = os.path.join(SAVE_ROOT, 'features', f'pooled_{modality}-tr{TR}-fo{frame_offset}-mo{mask_offset}', sbj)

            non_overlap_train_indices, non_overlap_test_indices = get_non_overlap_indices_for_concatenate_data(sampling_mode, jupyter=jupyter)
            eye_tracking_exist_flags = []
            for run_id in RUN_IDS:
                eye_tracking_exist_flag_path = os.path.join(eye_tracking_exist_flag_dir, f'fg_b_{run_id}-flag.npy')
                eye_tracking_exist_flags.append(np.load(eye_tracking_exist_flag_path))
                assert len(eye_tracking_exist_flags[-1]) == RUN_VOLUMES[run_id-1], 'eye_tracking_exist_flag:{} vs RUN_VOLUMES:{}'.format(len(eye_tracking_exist_flags[-1]), RUN_VOLUMES[run_id-1])
            eye_tracking_exist_flags = np.concatenate(eye_tracking_exist_flags, axis=0) # n_samples of bool
            non_overlap_train_indices = np.array([i for i in non_overlap_train_indices if eye_tracking_exist_flags[i]])
            non_overlap_test_indices  = np.array([i for i in non_overlap_test_indices if eye_tracking_exist_flags[i]])

            iou_df = pd.read_csv(iou_path)
            tgt_indices = np.where(np.isnan(iou_df['iou']))[0][:-1]+1
            iou_df.loc[tgt_indices, ['iou']]=np.nan
            iou_df = iou_df.dropna(axis=0)
            iou_label_th_test = iou_df.iloc[non_overlap_train_indices]['iou'].quantile(iou_q)

            saliency_eyetrack_df = pd.read_csv(saliency_eyetrack_path)
            saliency_th_upper_train = saliency_eyetrack_df.iloc[non_overlap_train_indices][saliency_label].quantile(saliency_TR_q)  # sal_labels = ['max_shift_norm', 'avg_shift_norm']
            saliency_th_lower_train = saliency_eyetrack_df.iloc[non_overlap_train_indices][saliency_label].quantile(1-saliency_TR_q)

            iou_th_list.append(iou_label_th_test)
            sal_upper_th_list.append(saliency_th_upper_train)
            sal_lower_th_list.append(saliency_th_lower_train)

            small_iou_indices = iou_df[iou_df['iou'] < iou_label_th_test]['idx_in_exp'].to_list()
             ## ２秒間でのsaliency shiftが大きいかつ(saliency shift)、iouが小さい(gaze shift)
            with_saliency_indices = saliency_eyetrack_df.query(f'{saliency_label} > {saliency_th_upper_train}')['idx_in_exp'].to_list()
            gaze_shift_with_saliency_indices = list(set(with_saliency_indices) & set(small_iou_indices))
            ## ２秒間でのsaliency shiftが小さいかつ(saliency fixed)、iouが小さい(gaze shift)
            without_saliency_indices = saliency_eyetrack_df.query(f'{saliency_label} < {saliency_th_lower_train}')['idx_in_exp'].to_list()
            gaze_shift_without_saliency_indices = list(set(without_saliency_indices) & set(small_iou_indices))

            sal_indices[sbj][saliency_TR_q] = (set(saliency_eyetrack_df['idx_in_exp'].to_list()) - set(with_saliency_indices))& set(non_overlap_test_indices) # - sal_indices[np.round(saliency_TR_q-0.1,1)]
            eye_sal_indices[sbj][saliency_TR_q] = sal_indices[sbj][saliency_TR_q]& set(small_iou_indices)
                # print( len(set(saliency_eyetrack_df['idx_in_exp'].to_list()) - set(with_saliency_indices))& set(non_overlap_test_indices))
            # if sbj == '01':
            #     print(saliency_th_lower_train, '<', len(gaze_shift_without_saliency_indices), saliency_th_upper_train, '>', len(gaze_shift_with_saliency_indices))

            gaze_shift_with_saliency_indices_test = list(set(gaze_shift_with_saliency_indices) & set(non_overlap_test_indices))
            gaze_shift_without_saliency_indices_test = list(set(gaze_shift_without_saliency_indices) & set(non_overlap_test_indices))

            joint_eye_sal_list.append(len(gaze_shift_with_saliency_indices_test))
            joint_eye_not_sal_list.append(len(gaze_shift_without_saliency_indices_test))
            test_data_indices =  set(iou_df['idx_in_exp'].to_list()) & set(non_overlap_test_indices)
            total_list.append(len(list(test_data_indices)))
            eye_indices = list(set(small_iou_indices) & test_data_indices)
            eye_list.append(len(eye_indices))
            saliency_indices = list(set(with_saliency_indices) &test_data_indices)
            wo_saliency_indices = list(set(without_saliency_indices) &test_data_indices)
            sal_list.append(len(saliency_indices))
            not_sal_list.append(len(wo_saliency_indices))


        print('iou: ', np.mean(iou_th_list),iou_th_list)
        print('sal upper: ', np.mean(sal_upper_th_list), sal_upper_th_list)
        print('sal lower: ', np.mean(sal_lower_th_list), sal_lower_th_list)
        grand_quantiles['iou_q'].append(iou_q)
        grand_quantiles['saliency_TR_q'].append(saliency_TR_q)
        grand_quantiles['iou_th'].append(np.mean(iou_th_list))
        grand_quantiles['sal_upper_th'].append(np.mean(sal_upper_th_list))
        grand_quantiles['sal_lower_th'].append(np.mean(sal_lower_th_list))

        behaviors['iou_q'].append(iou_q)
        behaviors['saliency_TR_q'].append(saliency_TR_q)
        behaviors['joint_eye_sal'].append(joint_eye_sal_list)
        behaviors['joint_eye_not_sal'].append(joint_eye_not_sal_list)
        behaviors['eye'].append(eye_list)
        behaviors['sal'].append(sal_list)
        behaviors['not_sal'].append(not_sal_list)
        behaviors['total'].append(total_list)

grand_quantiles_df = pd.DataFrame(grand_quantiles)
grand_quantiles_df.to_csv(os.path.join(SAVE_ROOT, 'behavior','grand_quantiles.csv'), index=False)

behaviors_df = pd.DataFrame(behaviors)
behaviors_df.to_csv(os.path.join(SAVE_ROOT, 'behavior','behaviors.csv'), index=False)
