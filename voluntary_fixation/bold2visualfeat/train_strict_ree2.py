import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from voluntary_fixation.envs import TR, RUN_IDS, SUBJECT_IDS, SAVE_ROOT, NUM_ROIS, BRIGHTNESS_DIR, RUN_VOLUMES, MOVIE_FPS
from voluntary_fixation.dataset.utils import delayed_label, torch_fix_seed, orthogonal_mat_against_vector
from voluntary_fixation.dataset.bold_dataset import bold44_dateset, get_non_overlap_indices_for_concatenate_data
from voluntary_fixation.bold2feat.regression.kamitani_regression import feature_prediction
from sklearn.decomposition import PCA
from himalaya.scoring import correlation_score
from himalaya.backend import set_backend
import time
import torch


def prepare_dataset(sbj_id:str, modality:str, frame_offset:int, mask_offset:int, hidden_state:bool,
                    delay:int, sampling_mode:str, remove_brightness:bool, remove_empty_eyetrack:bool,
                    return_brightness:bool=False)->Tuple[List[np.ndarray],List[np.ndarray], np.ndarray, np.ndarray]:
    # load bold
    bolds = bold44_dateset(sbj_id, rois=None) # n_rois x time x n_voxel

    # load features
    feature_dir = os.path.join(SAVE_ROOT, 'features', f'pooled_{modality}-tr{TR}-fo{frame_offset}-mo{mask_offset}')
    # if modality == 'masked_image' or modality == 'both' or modality == 'image' or modality == 'reversed_masked_image':
    #     feature_dir = os.path.join(feature_dir, sbj_id)
    masked_image_dir = os.path.join(SAVE_ROOT, 'features', f'pooled_masked_image-tr{TR}-fo{frame_offset}-mo{mask_offset}', sbj_id)
    feature_dir = os.path.join(feature_dir, sbj_id)
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
        eye_tracking_exist_flag = np.load(os.path.join(masked_image_dir, f'fg_b_{run_id}-flag.npy')) # n_samples
        np.save(os.path.join(feature_dir, f'fg_b_{run_id}-flag.npy'), eye_tracking_exist_flag)
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
    assert len(bolds[0]) == len(features[0]), 'bold:{} vs features:{}'.format(len(bolds[0]), len(features[0]))

    # load label
    label_train, label_test = get_non_overlap_indices_for_concatenate_data(sampling_mode)
    if remove_empty_eyetrack:
        print('Removing empty eyetrack')
        label_train = np.array([i for i in label_train if eye_tracking_exist_flags[i]])
        label_test = np.array([i for i in label_test if eye_tracking_exist_flags[i]])

    label_train_image, label_train_bold = delayed_label(label_train, delay)
    label_test_image, label_test_bold = delayed_label(label_test, delay)
    assert len(label_train_bold) == len(label_train_image), 'label_train_bold:{} vs label_train:{}'.format(len(label_train_bold), len(label_train_image))
    assert len(label_test_bold) == len(label_test_image), 'label_test_bold:{} vs label_test:{}'.format(len(label_test_bold), len(label_test_image))
    if sampling_mode == 'sandwitch':
        assert min(min(label_train_image), min(label_train_bold)) == 0
        assert max(max(label_train_image), max(label_train_bold)) == len(bolds[0])-1
    # labeling
    bold_train = [b[label_train_bold, :] for b in bolds]
    bold_test = [b[label_test_bold, :] for b in bolds]

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
    if return_brightness:
        return bold_train, bold_test, feature_train, feature_test, brightness
    else:
        return bold_train, bold_test, feature_train, feature_test

def fit(bold_train:np.ndarray, bold_test:np.ndarray, feature_train:np.ndarray, feature_test:np.ndarray,
        n_components_bold:int)-> Tuple[np.ndarray, List[float], List[float]]:
    # bold_train/test: n_samples x n_voxel
    # feature_train/test: n_hidden_layers x n_samples x n_features
    print(f'X {bold_train.shape}, Y {feature_train.shape}, X_te {bold_test.shape}, Y_te {feature_test.shape}')
    print('n_components_bold: ', n_components_bold)
    num_hidden_layers = len(feature_train)

    y_predicted_tests = []
    train_scores = []
    test_scores = []
    for i in range(num_hidden_layers):
        y_predicted_test, y_true_test, y_predicted_train, y_true_train = feature_prediction(bold_train, feature_train[i,:,:],
                                                                                            bold_test, feature_test[i,:,:],
                                                                                            n_voxel=n_components_bold,
                                                                                            n_iter=200,
                                                                                            beta=1)

        print(f'Train_pr {y_predicted_train.shape}, Train_gt {y_true_train.shape}, Test_pr {y_predicted_test.shape}, Test_gt {y_true_test.shape}')
        train_rs = correlation_score(y_predicted_train, y_true_train)
        test_rs = correlation_score(y_predicted_test,y_true_test)
        if not backend_type == 'numpy':
            train_rs = train_rs.cpu().numpy()
            test_rs = test_rs.cpu().numpy()
            # y_predicted_test = y_predicted_test.cpu().numpy()
        print(f'Prediction accuracy (Train) is: {np.nanmean(train_rs):3.3}')
        print(f'Prediction accuracy (Test) is: {np.nanmean(test_rs):3.3}')
        y_predicted_tests.append(y_predicted_test)
        train_scores.append(np.nanmean(train_rs))
        test_scores.append(np.nanmean(test_rs))

    y_predicted_tests = np.stack(y_predicted_tests, axis=0) # n_hidden_layers x n_samples x n_voxel

    return y_predicted_tests, train_scores, test_scores


def run(sbj_id:str, modality:str, frame_offset:int, mask_offset:int, hidden_state:bool,
        delay:int, sampling_mode:str,n_components_feature:int, n_components_bold:int, remove_brightness:bool,
        remove_empty_eyetrack:bool, save_dir:str):

    # dataset
    bold_train, bold_test, feature_train, feature_test = prepare_dataset(sbj_id, modality,
                                                                         frame_offset, mask_offset,
                                                                         hidden_state,
                                                                         delay, sampling_mode,
                                                                         remove_brightness, remove_empty_eyetrack)
    num_hidden_layers = len(feature_train)
    print('num_hidden_layers: ', num_hidden_layers)

    # pca
    if n_components_feature < 0:
        pass
    else:
        print('PCA running... {} to {}'.format(feature_train.shape[-1], n_components_feature))
        pca = PCA(n_components=n_components_feature)
        feature_train_pca = []
        feature_test_pca = []
        for i in range(num_hidden_layers):
            feature_train_pca.append(pca.fit_transform(feature_train[i]))
            feature_test_pca.append(pca.transform(feature_test[i]))
            print(f'{i} th layer:: Contribution:: ', np.cumsum(pca.explained_variance_ratio_)[n_components_feature-1])
            pca_loadings = pca.components_
            pca_loadings_savefile = os.path.join(save_dir, f'pca_loadings_layer{i}.npy')
            np.save(pca_loadings_savefile, pca_loadings)
            print('pca_loadings save to ', pca_loadings_savefile)
        feature_train = np.stack(feature_train_pca, axis=0)
        feature_test = np.stack(feature_test_pca, axis=0)
        del feature_train_pca, feature_test_pca

        print('feature_train.shape: ', feature_train.shape)
        print('feature_test.shape: ', feature_test.shape)


    # save
    if hidden_state:
        gt_savefile = os.path.join(save_dir, f'gt-hs-test.npy')
    else:
        gt_savefile = os.path.join(save_dir, f'gt-test.npy')
    np.save(gt_savefile, feature_test)
    print('gt of test save to ', gt_savefile)

    log_df = {'roi':[], 'train_score':[], 'test_score':[], 'hidden_layer_id':[]} # for logging
    # for roi in tqdm(range(NUM_ROIS)):
    with tqdm(range(NUM_ROIS)) as pbar:
        for roi in pbar:
            pbar.set_description(f'ROI: {roi}' )
            # fit
            y_predicted_tests, train_scores, test_scores = fit(bold_train[roi], bold_test[roi], feature_train,
                                                            feature_test, n_components_bold)
            assert isinstance(train_scores, list), 'train_scores must be list'
            assert isinstance(test_scores, list), 'test_scores must be list'
            assert len(train_scores) == num_hidden_layers, 'train_scores:{} vs num_hidden_layers:{}'.format(len(train_scores), num_hidden_layers)
            log_df['roi'] += [roi]*num_hidden_layers
            log_df['train_score'] += train_scores
            log_df['test_score'] += test_scores
            log_df['hidden_layer_id'] += list(range(num_hidden_layers))


            # save
            if hidden_state:
                pred_savefile = os.path.join(save_dir, f'pred-roi_{roi}-hs-test.npy')
            else:
                pred_savefile = os.path.join(save_dir, f'pred-roi_{roi}-test.npy')
            np.save(pred_savefile, y_predicted_tests)
            print('prediction of test save to ', pred_savefile)
            print('')
    log_df = pd.DataFrame(log_df)
    csv_path = os.path.join(save_dir, 'scores.csv')
    log_df.to_csv(csv_path, index=False)
    print('csv save to ', csv_path)


if __name__ == '__main__':
    backend_type = 'torch_cuda' if torch.cuda.is_available() else 'numpy'
    backend = set_backend(backend_type, on_error="warn")
    torch_fix_seed(1126)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--sbj",
        required=True,
        type=str,
        help="sunject_name"
    )
    parser.add_argument(
        "-m",
        "--modality",
        required=True,
        type=str,
        help="run image, masked_image, both"
    )
    parser.add_argument(
        "-hs",
        "--hidden_state",
        action='store_true',
        help="get hidden state of vision encoder"
    )
    parser.add_argument(
        "-fo",
        "--frame_offset",
        type=int,
        default=0,
        help="frame offset"
    )
    parser.add_argument(
        "-mo",
        "--mask_offset",
        type=int,
        default=0,
        help="mask offset"
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=int,
        default=0,
        help="delay"
    )
    parser.add_argument(
        "-sm",
        "--sampling_mode",
        type=str,
        default='sandwitch',
        help="sampling mode"
    )
    parser.add_argument(
        "-ncf",
        "--n_components_feature",
        type=int,
        default=100,
        help="n_components_feature (using PCA)"
    )
    parser.add_argument(
        "-ncb",
        "--n_components_bold",
        type=int,
        default=250,
        help="n_components_bold (using voxel slection by correlation)"
    )



    opt = parser.parse_args()
    print(opt)
    remove_str = ''
    if opt.remove_brightness:
        remove_str += '-remove_brightness'
    if opt.remove_empty_eyetrack:
        remove_str += '-remove_empty_eyetrack'
    if opt.n_components_bold == 250:
        save_dir = os.path.join(SAVE_ROOT, 'bold2feat_strict_ree2', f'delay{opt.delay}-fo{opt.frame_offset}-mo{opt.mask_offset}',
                                f'pca{opt.n_components_feature}-{opt.modality}-{opt.sampling_mode}{remove_str}', opt.sbj)
    else:
        save_dir = os.path.join(SAVE_ROOT, 'bold2feat_strict_ree2', f'ncb{opt.n_components_bold}', f'delay{opt.delay}-fo{opt.frame_offset}-mo{opt.mask_offset}',
                                f'pca{opt.n_components_feature}-{opt.modality}-{opt.sampling_mode}{remove_str}', opt.sbj)
    os.makedirs(save_dir, exist_ok=True)
    print('save_dir: ', save_dir)
    assert opt.sbj in SUBJECT_IDS
    print('subject:: ', opt.sbj)
    start_time = time.time()
    run(opt.sbj, opt.modality, opt.frame_offset, opt.mask_offset, opt.hidden_state, opt.delay,
        opt.sampling_mode, opt.n_components_feature, opt.n_components_bold,  opt.remove_brightness,
        opt.remove_empty_eyetrack, save_dir)
    print('elapsed time: ', time.time() - start_time)

