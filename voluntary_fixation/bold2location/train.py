import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from voluntary_fixation.envs import TR, RUN_IDS, SUBJECT_IDS, SAVE_ROOT, NUM_ROIS, BRIGHTNESS_DIR, RUN_VOLUMES, MOVIE_FPS, MOVIE_WIDTH, MOVIE_HEIGHT, EYEMOVE_ROOT
from voluntary_fixation.dataset.utils import delayed_label, torch_fix_seed, orthogonal_mat_against_vector
from voluntary_fixation.dataset.bold_dataset import bold44_dateset, get_non_overlap_indices_for_concatenate_data
from voluntary_fixation.behavior.src.temporal_alignment_filtering import sacc_cnt_shift_relation_df, alignment2VOLUME
from voluntary_fixation.bold2visualfeat.regression.kamitani_regression import feature_prediction
from voluntary_fixation.behavior.src.data_loader import read_csvs
from sklearn.decomposition import PCA
from himalaya.scoring import correlation_score
from himalaya.backend import set_backend
import time
import torch


def prepare_dataset(sbj_id:str, gaze_shift_label:List[str], delay:int,
                    sampling_mode:str, rm_empty_eyetrack:bool, save_dir:str)->Tuple[List[np.ndarray],List[np.ndarray], np.ndarray, np.ndarray]:
    # load bold
    bolds = bold44_dateset(sbj_id, rois=None) # n_rois x time x n_voxel
    eyetrack_df_list = []
    eye_tracking_exist_flags = []
    masked_image_dir = os.path.join(SAVE_ROOT, 'features', f'pooled_masked_image-tr{TR}-fo0-mo0', sbj_id)
    idx_offsets = np.cumsum([0] + RUN_VOLUMES)
    for run_id in tqdm(RUN_IDS):
        eye_tracking_exist_flag = np.load(os.path.join(masked_image_dir, f'fg_b_{run_id}-flag.npy')) # n_samples
        eye_tracking_exist_flags.append(eye_tracking_exist_flag)
        assert len(eye_tracking_exist_flag) == RUN_VOLUMES[run_id-1], 'eye_tracking_exist_flag:{} vs RUN_VOLUMES:{}'.format(len(eye_tracking_exist_flag), RUN_VOLUMES[run_id-1])

        original_eyetrack_path = os.path.join(EYEMOVE_ROOT,  f'sub-{sbj_id}/sub-{sbj_id}_task-movie_run-{run_id}_events.tsv')
        # get eyemovement
        _, original_eyetrack_df = read_csvs(None, original_eyetrack_path,
                                                    gaze_shift_label, video_width=MOVIE_WIDTH,
                                                    video_height=MOVIE_HEIGHT, fps=MOVIE_FPS)
        original_eyetrack_df.drop(columns=['label'], inplace=True)
        original_eyetrack_df = alignment2VOLUME(original_eyetrack_df, run_id, agg='mean')
        assert len(original_eyetrack_df) == RUN_VOLUMES[run_id-1], f'len(original_eyetrack_Df_df)={len(original_eyetrack_df)}, RUN_VOLUMES[run_id-1]={RUN_VOLUMES[run_id-1]}'
        idx_in_exp = list(np.arange(RUN_VOLUMES[run_id-1]) + idx_offsets[run_id-1])
        original_eyetrack_df['idx_in_exp'] = idx_in_exp
        eyetrack_df_list.append(original_eyetrack_df)

    eyetrack_df = pd.concat(eyetrack_df_list, axis=0)
    eyetrack_df.set_index('idx_in_exp', inplace=True)
    eye_tracking_exist_flags = np.concatenate(eye_tracking_exist_flags, axis=0) # n_samples
    # load label
    label_train, label_test = get_non_overlap_indices_for_concatenate_data(sampling_mode)
    if rm_empty_eyetrack:
        print('Removing empty eyetrack')
        label_train_ = np.array([i for i in label_train if eye_tracking_exist_flags[i]])
        label_test_ = np.array([i for i in label_test if eye_tracking_exist_flags[i]])



    label_train = label_train_
    label_test = label_test_
    label_train.sort()
    label_test.sort()
    label_train_eyetrack, label_train_bold = delayed_label(label_train, delay)
    label_test_eyetrack, label_test_bold = delayed_label(label_test, delay)

    # it_save_path = os.path.join(save_dir, f'{sbj_id}-{"_".join(gaze_shift_label)}.npy')
    # label_test = list(label_test)
    # indices_in_test = np.array([label_test.index(i) for i in label_test_ ])
    # os.makedirs(os.path.dirname(it_save_path), exist_ok=True)
    # np.save(it_save_path, indices_in_test)
    # import pdb; pdb.set_trace()
    assert len(label_train_bold) == len(label_train_eyetrack), 'label_train_bold:{} vs label_train:{}'.format(len(label_train_bold), len(label_train_eyetrack))
    assert len(label_test_bold) == len(label_test_eyetrack), 'label_test_bold:{} vs label_test:{}'.format(len(label_test_bold), len(label_test_eyetrack))
    if sampling_mode == 'sandwitch':
        assert min(min(label_train_eyetrack), min(label_train_bold)) == 0
        assert max(max(label_train_eyetrack), max(label_train_bold)) == len(bolds[0])-1
    bold_train = [b[label_train_bold, :] for b in bolds]
    bold_test = [b[label_test_bold, :] for b in bolds]


    end_x_train = eyetrack_df.iloc[label_train_eyetrack]['end_x'].to_numpy() # n_samples
    end_x_test = eyetrack_df.iloc[label_test_eyetrack]['end_x'].to_numpy() # n_samples
    end_y_train = eyetrack_df.iloc[label_train_eyetrack]['end_y'].to_numpy() # n_samples
    end_y_test = eyetrack_df.iloc[label_test_eyetrack]['end_y'].to_numpy() # n_samples
    print('NUM_SAMPLES:: TRAIN: {}\t TEST: {}'.format(len(label_train), len(label_test)))

    return bold_train, bold_test, np.stack([end_x_train, end_y_train], axis=-1), np.stack([end_x_test, end_y_test], axis=-1)



def fit(bold_train:np.ndarray, bold_test:np.ndarray, eyetrack_train:np.ndarray, eyetrack_test:np.ndarray,
        n_components_bold:int)-> Tuple[np.ndarray, List[float], List[float], np.ndarray]:
    # bold_train/test: n_samples x n_voxel
    # eyetrack_train/test: n_hidden_layers x n_samples x n_features
    print(f'X {bold_train.shape}, Y {eyetrack_train.shape}, X_te {bold_test.shape}, Y_te {eyetrack_test.shape}')
    print('n_components_bold: ', n_components_bold)

    y_predicted_test, y_true_test, y_predicted_train, y_true_train, voxel_indices = feature_prediction(bold_train, eyetrack_train,
                                                                                        bold_test, eyetrack_test,
                                                                                        n_voxel=n_components_bold,
                                                                                        n_iter=200,
                                                                                        beta=1,
                                                                                        return_voxel=True)

    print(f'Train_pr {y_predicted_train.shape}, Train_gt {y_true_train.shape}, Test_pr {y_predicted_test.shape}, Test_gt {y_true_test.shape}')
    train_rs = correlation_score(y_predicted_train, y_true_train)
    test_rs = correlation_score(y_predicted_test,y_true_test)
    if not backend_type == 'numpy':
        train_rs = train_rs.cpu().numpy()
        test_rs = test_rs.cpu().numpy()
        # y_predicted_test = y_predicted_test.cpu().numpy()
    print(f'Prediction accuracy (Train) is: {np.nanmean(train_rs):3.3}')
    print(f'Prediction accuracy (Test) is: {np.nanmean(test_rs):3.3}')

    return y_predicted_test, np.nanmean(train_rs), np.nanmean(test_rs), np.concatenate(voxel_indices, axis=0)


def run(sbj_id:str, delay:int, sampling_mode:str, n_components_bold:int,
        save_dir:str, gaze_shift_label:List[str], rm_empty_eyetrack:bool):

    # dataset
    bold_train, bold_test, feature_train, feature_test = prepare_dataset(sbj_id, gaze_shift_label,
                                                                         delay, sampling_mode,
                                                                         rm_empty_eyetrack, save_dir)

    # save
    gt_savefile = os.path.join(save_dir, f'gt-test.npy')
    np.save(gt_savefile, feature_test)
    print('gt of test save to ', gt_savefile)

    log_df = {'roi':[], 'train_score':[], 'test_score':[]} # for logging
    # for roi in tqdm(range(NUM_ROIS)):
    with tqdm(range(NUM_ROIS)) as pbar:
        for roi in pbar:
            pbar.set_description(f'ROI: {roi}' )
            # fit
            y_predicted_test, train_score, test_score, voxel_indices = fit(bold_train[roi], bold_test[roi], feature_train,
                                                            feature_test, n_components_bold)

            log_df['roi'].append(roi)
            log_df['train_score'].append(train_score)
            log_df['test_score'].append(test_score)
            # save
            pred_savefile = os.path.join(save_dir, f'pred-roi_{roi}-test.npy')
            np.save(pred_savefile, y_predicted_test)
            print('prediction of test save to ', pred_savefile)
            print('')
            voxel_savefile = os.path.join(save_dir, f'voxels-roi_{roi}.npy')
            np.save(voxel_savefile, voxel_indices)
            print('voxels  save to ', voxel_savefile)

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
        help="sampling mode. session, segment, sandwitch"
    )
    parser.add_argument(
        "-ncb",
        "--n_components_bold",
        type=int,
        default=10000,
        help="n_components_bold (using voxel slection by correlation)"
    )
    parser.add_argument(
        "-gsl",
        "--gaze_shift_label",
        nargs="*", type=str,  # FIXA PURS SACC
        default=['FIXA', 'PURS'],
        help="n_components_bold (using voxel slection by correlation)"
    )
    parser.add_argument(
        "-ree",
        "--rm_empty_eyetrack",
        action='store_true',
        help="n_components_bold (using voxel slection by correlation)"
    )

    opt = parser.parse_args()

    rs_str = '-rm_empty_eyetrack' if opt.rm_empty_eyetrack else ''
    save_dir = os.path.join(SAVE_ROOT, f'bold2location', f'delay{opt.delay}-{"_".join(opt.gaze_shift_label)}',
                                f'{opt.sampling_mode}-ncb{opt.n_components_bold}{rs_str}', opt.sbj)
    os.makedirs(save_dir, exist_ok=True)
    print('save_dir: ', save_dir)
    assert opt.sbj in SUBJECT_IDS
    print('subject:: ', opt.sbj)

    start_time = time.time()
    run(opt.sbj, opt.delay, opt.sampling_mode, opt.n_components_bold, save_dir, opt.gaze_shift_label, opt.rm_empty_eyetrack)
    print('elapsed time: ', time.time() - start_time)

