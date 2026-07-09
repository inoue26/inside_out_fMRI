import os
import numpy as np
from typing import List, Tuple

from voluntary_fixation.envs import TR, RUN_IDS, SUBJECT_IDS, SAVE_ROOT, NUM_ROIS, BRIGHTNESS_DIR, RUN_VOLUMES, MOVIE_FPS, MOVIE_WIDTH, MOVIE_HEIGHT, EYEMOVE_ROOT
from voluntary_fixation.dataset.bold_dataset import get_non_overlap_indices_for_concatenate_data
import pandas as pd
from voluntary_fixation.dataset.utils import delayed_label, orthogonal_mat_against_vector

from voluntary_fixation.dataset.utils import delayed_label

from voluntary_fixation.bold2visualfeat.eval_utils import get_label, get_pred_and_gt
from sklearn.decomposition import PCA
import random
random.seed(1126)

def prepare_dataset(sbj_id:str, modality:str, frame_offset:int, mask_offset:int, hidden_state:bool,
                    delay:int, sampling_mode:str, remove_brightness:bool, remove_empty_eyetrack:bool,
                    return_brightness:bool=False, pca_components:int=None, slice_=-1, use_own_pca=False)->Tuple[List[np.ndarray],List[np.ndarray], np.ndarray, np.ndarray]:

    # load features
    feature_dir = os.path.join(SAVE_ROOT, 'features', f'pooled_{modality}-tr{TR}-fo{frame_offset}-mo{mask_offset}')
    feature_dir_masked_image = os.path.join(SAVE_ROOT, 'features', f'pooled_masked_image-tr{TR}-fo{frame_offset}-mo{mask_offset}', sbj_id)
    # if modality == 'masked_image' or modality == 'both' or modality == 'image' or modality == 'reversed_masked_image':
    #     feature_dir = os.path.join(feature_dir, sbj_id)
    feature_dir = os.path.join(feature_dir, sbj_id)
    if modality == 'shuffle':
        feature_dir = feature_dir_masked_image
    features = []
    eye_tracking_exist_flags = []
    brightness_means = []
    brightness_lrs = []
    brightness_ud = []
    for run_id in RUN_IDS:
        if hidden_state:
            feature_path = os.path.join(feature_dir, f'fg_b_{run_id}-hs.npy')
            feature = np.load(feature_path) # num_layer x num_samples x 1408
        else:
            feature_path = os.path.join(feature_dir, f'fg_b_{run_id}.npy')
            feature = np.load(feature_path) # n_samples x 1408
            feature = feature[np.newaxis,:,:] # num_layer x n_samples x 1408
        # feature_path_debug = os.path.join(feature_dir_masked_image, f'fg_b_{run_id}-hs.npy')
        # feature_debug = np.load(feature_path_debug) # num_layer x num_samples x 1408 ## (5, 451, 1408)
        # import pdb; pdb.set_trace()
        feature = feature[slice_]
        eye_tracking_exist_flag = np.load(os.path.join(feature_dir_masked_image, f'fg_b_{run_id}-flag.npy')) # n_samples
        features.append(feature)
        eye_tracking_exist_flags.append(eye_tracking_exist_flag)
        assert len(eye_tracking_exist_flag) == RUN_VOLUMES[run_id-1], 'eye_tracking_exist_flag:{} vs RUN_VOLUMES:{}'.format(len(eye_tracking_exist_flag), RUN_VOLUMES[run_id-1])
        assert len(feature[0]) == RUN_VOLUMES[run_id-1], 'feature:{} vs RUN_VOLUMES:{}'.format(len(feature[0]), RUN_VOLUMES[run_id-1])

        brmean = pd.read_csv(os.path.join(BRIGHTNESS_DIR, f'fg_av_ger_seg{run_id-1}_brmean.tsv'), sep = "[\t]", engine='python')
        brlr = pd.read_csv(os.path.join(BRIGHTNESS_DIR, f'fg_av_ger_seg{run_id-1}_brlr.tsv'), sep = "[\t]", engine='python')
        brud = pd.read_csv(os.path.join(BRIGHTNESS_DIR, f'fg_av_ger_seg{run_id-1}_brud.tsv'), sep = "[\t]", engine='python')

        brightness_means += brmean.iloc[::TR*MOVIE_FPS]['brmean'].to_list()[:RUN_VOLUMES[run_id-1]]
        brightness_lrs += brlr.iloc[::TR*MOVIE_FPS]['brlr'].to_list()[:RUN_VOLUMES[run_id-1]]
        brightness_ud += brud.iloc[::TR*MOVIE_FPS]['brud'].to_list()[:RUN_VOLUMES[run_id-1]]
    features = np.concatenate(features, axis=1) # num_layer x num_samples x embedding_dims
    eye_tracking_exist_flags = np.concatenate(eye_tracking_exist_flags, axis=0) # n_samples
    brightness = np.array([brightness_means, brightness_lrs, brightness_ud]).T # num_samples x 3
    print('features.shape: ', features.shape)
    assert features.shape[1] == brightness.shape[0], 'features:{} vs brightness:{}'.format(features.shape[1], brightness.shape[0])
    # if remove_brightness:
    #     print('Removing Brightness component from features')
    #     # feat_pca1, corr_score1 = debug_pca(features, features.shape[0], 2, ortho_target_vector=brightness[:,:1])
    #     for i in range(features.shape[0]):
    #         features[i] = orthogonal_mat_against_vector(brightness[:,:1], features[i], z_score=True)

    # load label
    label_train, label_test = get_non_overlap_indices_for_concatenate_data(sampling_mode)
    eye_tracking_exist_flags_test = label_test # np.array(eye_tracking_exist_flags)
    eye_tracking_exist_flags_test = np.array([True  if eye_tracking_exist_flags[i] else False for i in label_test])
    # import pdb; pdb.set_trace()
    if delay < 0:
        eye_tracking_exist_flags_test = eye_tracking_exist_flags_test[-delay:]
    elif delay > 0:
        eye_tracking_exist_flags_test = eye_tracking_exist_flags_test[:-delay]

    if remove_empty_eyetrack:
        print('Removing empty eyetrack')
        label_train = np.array([i for i in label_train if eye_tracking_exist_flags[i]])
        label_test = np.array([i for i in label_test if eye_tracking_exist_flags[i]])

    label_train_image, label_train_bold = delayed_label(label_train, delay)
    label_test_image, label_test_bold = delayed_label(label_test, delay)
    assert len(label_train_bold) == len(label_train_image), 'label_train_bold:{} vs label_train:{}'.format(len(label_train_bold), len(label_train_image))
    assert len(label_test_bold) == len(label_test_image), 'label_test_bold:{} vs label_test:{}'.format(len(label_test_bold), len(label_test_image))

    feature_train = features[:,label_train_image,:]
    feature_test = features[:,label_test_image,:]
    print('NUM_SAMPLES:: TRAIN: {}\t TEST: {}'.format(len(label_train), len(label_test)))
    if remove_brightness:
        brightness_train = brightness[label_train_image]
        brightness_test = brightness[label_test_image]
        for i in range(features.shape[0]):
            feature_train[i] = orthogonal_mat_against_vector(brightness_train[:,:1], feature_train[i], z_score=True)
            feature_test[i] = orthogonal_mat_against_vector(brightness_test[:,:1], feature_test[i], z_score=True)
    else:
        mean_train = np.mean(feature_train, axis=1, keepdims=True)
        std_train = np.std(feature_train, axis=1, keepdims=True)
        feature_train = (feature_train - mean_train) / std_train
        feature_test = (feature_test - mean_train) / std_train

    # pca
    num_layers = feature_train.shape[0]
    num_samples_train = feature_train.shape[1]
    num_samples_test = feature_test.shape[1]
    embedding_dims = feature_train.shape[2]
    sliced_layers = np.arange(5)[slice_]
    if pca_components is not None:
        pca_features_train = np.zeros((num_layers, num_samples_train, pca_components))
        pca_features_test = np.zeros((num_layers, num_samples_test, pca_components))
        print('PCA running... {} to {}'.format(embedding_dims, pca_components))
        pca = PCA(n_components=pca_components)
        for l, layer_id in enumerate(sliced_layers):
            if use_own_pca:
                pca_features_train[l] = pca.fit_transform(feature_train[l])
                pca_features_test[l] = pca.transform(feature_test[l])
            else:
                if remove_brightness:
                    pca_loadings_path = os.path.join(SAVE_ROOT, 'bold2feat_strict_ree2', f'delay{delay}-fo{frame_offset}-mo{mask_offset}',
                                            f'pca{pca_components}-masked_image-segment-remove_brightness-remove_empty_eyetrack',
                                            sbj_id, f'pca_loadings_layer{layer_id}.npy')
                else:
                    pca_loadings_path = os.path.join(SAVE_ROOT, 'bold2feat_strict_ree2', f'delay{delay}-fo{frame_offset}-mo{mask_offset}',
                                            f'pca{pca_components}-masked_image-segment-remove_empty_eyetrack',
                                            sbj_id, f'pca_loadings_layer{layer_id}.npy')
                pca_loadings = np.load(pca_loadings_path) # pca_components x embedding_dims
                offset = np.mean(feature_train[l], axis=0, keepdims=True)@pca_loadings.T
                pca_features_train[l] = feature_train[l]@pca_loadings.T - offset
                pca_features_test[l] = feature_test[l]@pca_loadings.T - offset

        del features
        feature_train = pca_features_train
        feature_test = pca_features_test
    if modality == 'shuffle':
        train_shuffled_indices = np.random.permutation(num_samples_train)
        test_shuffled_indices = np.random.permutation(num_samples_test)
        feature_train = feature_train[:,train_shuffled_indices]
        feature_test = feature_test[:,test_shuffled_indices]
    if return_brightness:
        return feature_train, feature_test, brightness, eye_tracking_exist_flags_test
    else:
        return feature_train, feature_test, eye_tracking_exist_flags_test



def run(savedir:str, delay:int, iou_q:float, eyemovement_q:float, frame_offset:float, mask_offset:float,
        saliency_q:float, gaze_shift_label:List[str], sampling_mode:str, saliency_label:str,
        remove_brightness:bool, remove_empty_eyetrack:bool, n_components:int, modality:str, control_modality:str,
        hidden_state:bool, p_criteria:float,
        use_large_cnt:bool, label_mode:int, rois:list, slice_:slice=slice(1,5,1), saliency_TR_q:float=0.5, use_own_pca:bool=False):
    remove_str = ''
    if remove_brightness:
        remove_str += '-remove_brightness'
    if remove_empty_eyetrack:
        remove_str += '-remove_empty_eyetrack'

    saliency_dir = os.path.join(SAVE_ROOT, 'saliency', 'deepgaze2e_predict')
    ret = {'sbj':[], 'roi':[], 'w_sal_corr':[], 'wo_sal_corr':[]}
    if control_modality == 'previous_fixated_image':
        control_modality = 'masked_image'
        previous_fixated_image = True
    else:
        previous_fixated_image = False
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

        _, gt_test_control, eye_tracking_exist_flags_test = prepare_dataset(sbj, control_modality, frame_offset, mask_offset, hidden_state,
                                    delay, sampling_mode, remove_brightness, remove_empty_eyetrack,
                                    return_brightness=False, pca_components=n_components, slice_=slice_, use_own_pca=use_own_pca)
        save_dir_sbj = os.path.join(savedir, sbj, 'rel_errors', 'delay'+str(delay))
        os.makedirs(save_dir_sbj, exist_ok=True)
        for roi in rois: # range(NUM_ROIS):
            ret['sbj'].append(sbj)
            ret['roi'].append(roi)
            pred, tmp_gt = get_pred_and_gt(pred_and_gt_dir, roi, hidden_state, slice_=slice_)# num_hidden_layer x n_samples x n_components
            # import pdb; pdb.set_trace()
            assert tmp_gt.shape[1] == gt_test_control.shape[1], 'tmp_gt:{} vs gt_test_control:{}'.format(tmp_gt.shape[1], gt_test_control.shape[1])
            # print(correlation_score(gt_test_control[0], tmp_gt[0]))
            # import pdb; pdb.set_trace()
            assert pred.shape[2] == n_components, f'pred.shape[1]={pred.shape[2]}, n_components={n_components}'
            n_hidden_layers = len(pred)
            # pred, gt: num_hidden_layer x n_samples x n_components -> n_samples x (num_hidden_layer * n_components)
            pred = np.transpose(pred, (1, 0, 2)).reshape(-1, n_components*n_hidden_layers)
            gt_test = np.transpose(gt_test_control, (1, 0, 2)).reshape(-1, n_components*n_hidden_layers)
            if previous_fixated_image:
                w_sal_in_test_gt = np.array(w_sal_in_test)-1 # time lagによるmodelの精度に依存している
                wo_sal_in_test_gt = np.array(wo_sal_in_test)-1
            else:
                w_sal_in_test_gt = np.array(w_sal_in_test)
                wo_sal_in_test_gt = np.array(wo_sal_in_test)
            errors = np.sqrt(np.sum(((gt_test - pred)/gt_test)**2, axis=1)) # 相対誤差
            # assert len(errors) == np.sum(eye_tracking_exist_flags_test), 'errors:{} vs eye_tracking_exist_flags_test:{}'.format(len(errors), np.sum(eye_tracking_exist_flags_test))
            # import pdb; pdb.set_trace()
            errors_nan = np.zeros(len(eye_tracking_exist_flags_test))
            errors = list(errors)
            for i, flag in enumerate(eye_tracking_exist_flags_test):
                if flag and len(errors) > 0:
                    errors_nan[i] = errors.pop(0)
                else:
                    errors_nan[i] = np.nan
            print('errors_nan size: ', len(errors_nan))
            np.save(os.path.join(save_dir_sbj, f'roi{roi}.npy'), errors_nan)
            print(os.path.join(save_dir_sbj, f'roi{roi}.npy'))

        gt_test_size = np.sqrt((gt_test**2).sum(axis=1))
        gt_test_theta = np.arctan2(gt_test[:,1], gt_test[:,0])

        gt_test_size_nan = np.zeros(len(eye_tracking_exist_flags_test))
        gt_test_size = list(gt_test_size)
        for i, flag in enumerate(eye_tracking_exist_flags_test):
            if flag and len(gt_test_size) > 0:
                gt_test_size_nan[i] = gt_test_size.pop(0)
            else:
                gt_test_size_nan[i] = np.nan
        gt_test_theta_nan = np.zeros(len(eye_tracking_exist_flags_test))
        gt_test_theta = list(gt_test_theta)
        for i, flag in enumerate(eye_tracking_exist_flags_test):
            if flag and len(gt_test_theta) > 0:
               gt_test_theta_nan[i] =gt_test_theta.pop(0)
            else:
               gt_test_theta_nan[i] = np.nan
        gt_test_nan = np.zeros((len(eye_tracking_exist_flags_test),2))
        gt_test1 = list(gt_test[:,0])
        gt_test2 = list(gt_test[:,1])
        for i, flag in enumerate(eye_tracking_exist_flags_test):
            if flag and len(gt_test1) > 0:
               gt_test_nan[i,0] =gt_test1.pop(0)
               gt_test_nan[i,1] =gt_test2.pop(0)
            else:
               gt_test_nan[i,0] = np.nan
               gt_test_nan[i,1] = np.nan

        os.makedirs(os.path.join(SAVE_ROOT, f'bold2feat_control_RSSE_strict_ree2/infos', exist_ok=True))
        np.save(os.path.join(SAVE_ROOT, f'bold2feat_control_RSSE_strict_ree2/infos/gt_test_size-d{delay}.npy', gt_test_size_nan))
        np.save(os.path.join(SAVE_ROOT, f'bold2feat_control_RSSE_strict_ree2/infos/gt_test_theta-d{delay}.npy', gt_test_theta_nan))
        np.save(os.path.join(SAVE_ROOT, f'bold2feat_control_RSSE_strict_ree2/infos/gt_test-d{delay}.npy', gt_test_nan))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--saliency_TR_q', type=float, default=0.7)
    parser.add_argument('--iou_q', type=float, default=0.5)
    args = parser.parse_args()

    gaze_shift_label = ['FIXA', 'PURS'] # ['FIXA', 'PURS']  # ['SACC'] ['FIXA', 'PURS', 'SACC']
    frame_offset = 0
    mask_offset = 0
    eyemovement_q = 0.9
    saliency_q = 0.9
    sampling_mode = 'segment' # session, sandwitch
    saliency_label = 'avg_shift_norm' # 'avg_shift_norm' 'max_shift_norm'
    remove_brightness = True# True
    remove_empty_eyetrack = True # True
    n_components = 2
    modality = 'masked_image'
    hidden_state = True
    p_criteria = 0.05
    use_large_cnt = False# True
    label_mode = 'public'

    slice_ = slice(4,5,1)# slice(4,5,1) # slice(1,5,1) slice(2,3,1)
    iou_q = args.iou_q# 0.3
    saliency_TR_q = args.saliency_TR_q# 0.7  # 0.7 1.61でも36でる  # 0.75, 1.61 < 0.1   1.62: 0.3 - 0.7
    use_own_pca = False # False
    rois = range(NUM_ROIS) # [36] # range(NUM_ROIS)

    remove_str = ''
    if remove_brightness:
        remove_str += '-remove_brightness'
    if remove_empty_eyetrack:
        remove_str += '-remove_empty_eyetrack'
    if use_own_pca:
        savedir = os.path.join(SAVE_ROOT, 'bold2feat_control_RSSE_strict_ree2', f'eval{label_mode}', remove_str+'-own_pca')
    else:
        savedir = os.path.join(SAVE_ROOT, 'bold2feat_control_RSSE_strict_ree2', f'eval{label_mode}', remove_str+f'{iou_q}-{saliency_TR_q}')
    os.makedirs(savedir, exist_ok=True)

    for control_modality in ['masked_image']:# ,'previous_fixated_image']: #['previous_masked_image', 'sampled_reverse_masked_image', 'sampled_whole_image','saliency_masked_image', 'shuffled_masked_image2', 'masked_image', 'reverse_masked_image', 'whole_image']: #['previous_masked_image', 'sampled_reverse_masked_image', 'sampled_whole_image','masked_image', 'saliency_masked_image', 'shuffled_masked_image2', 'masked_image', 'reverse_masked_image', 'whole_image']:
        print('control_modality:: ', control_modality)
        for delay in [-3, -2, -1, 0, 1, 2, 3, 4, 5]:
            run(savedir, delay, iou_q, eyemovement_q, frame_offset, mask_offset, saliency_q, gaze_shift_label,
                                    sampling_mode, saliency_label, remove_brightness, remove_empty_eyetrack, n_components, modality, control_modality,
                                    hidden_state, p_criteria, use_large_cnt, label_mode, rois, slice_, saliency_TR_q=saliency_TR_q, use_own_pca=use_own_pca)

    print('target hidden layer :: ', [1,10,20,30,40][slice_])