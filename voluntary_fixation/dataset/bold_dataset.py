import numpy as np
import os
from typing import List, Tuple
from voluntary_fixation.envs import TR, RUN_VOLUMES, RUN_IDS, LABEL_ROOT, BOLD44_ROOT, SUBJECT_IDS

def bold44_dateset(sbj_id:str, rois:List[int]=None, jupyter:bool=False)->np.ndarray|List[np.ndarray]:
    bold_path = os.path.join(BOLD44_ROOT, f'sub{sbj_id}_sideLR_roi44_data-12345678.npz')
    if jupyter:
        if not os.path.exists(bold_path):
            bold_path = '../' + bold_path
        if not os.path.exists(bold_path):
            bold_path = '../' + bold_path

    print('Use bold file : ', bold_path)
    bold_data = np.load(bold_path, allow_pickle=True)['roi_data']
    if rois is None:
        print('Use rois : all ')
        print('bold data shape::: ', bold_data.shape)
        assert bold_data[0].shape[0] == np.sum(RUN_VOLUMES)
        assert len(bold_data) == 44
        return bold_data # n_rois x time x n_voxel # List
    else:
        print('Use rois : ', rois)
        rois_data = []
        for roi in rois:
            roi_data = bold_data[int(roi)]
            rois_data.append(roi_data)
        # rois_data = np.concatenate(rois_data, axis=-1)
        # assert len(rois_data) == np.sum(RUN_VOLUMES)
        # print('bold data shape::: ', bold_data.shape)
        return rois_data # time x n_voxel

def get_non_overlap_indices_for_concatenate_data(sampling_mode:str, exclude_overlap:bool=False,
                                                 jupyter:bool=False,
                                                 train_runs=[1,2,3,5,6,7],
                                                 train_sessions=[1,2,3,4])->Tuple[np.ndarray, np.ndarray]:
    if jupyter:
        label_root = '../' + LABEL_ROOT
        if not os.path.exists(label_root):
            label_root = '../' + label_root
    else:
        label_root = LABEL_ROOT
    # bold data is concatenated from all runs
    label_train = np.zeros(np.sum(RUN_VOLUMES))
    label_test = np.zeros(np.sum(RUN_VOLUMES))
    pre_last_id = 0
    for run_id, n_samples_per_run in zip(RUN_IDS, RUN_VOLUMES):
        if sampling_mode == 'segment':
            non_overlap_indices_path = os.path.join(label_root, f'non_overlap_indices/full-run{run_id}.npy')
            non_overlap_indices = np.load(non_overlap_indices_path)
            if run_id in train_runs:
                label_train[pre_last_id + non_overlap_indices] += 1
            else:
                label_test[pre_last_id + non_overlap_indices] += 1
        elif sampling_mode == 'session':
            non_overlap_indices_path = os.path.join(label_root, f'non_overlap_indices/full-run{run_id}.npy')
            non_overlap_indices = np.load(non_overlap_indices_path)
            if run_id in train_sessions:
                label_train[pre_last_id + non_overlap_indices] += 1
            else:
                label_test[pre_last_id + non_overlap_indices] += 1
        else:
            non_overlap_train_indices_path = os.path.join(label_root, f'non_overlap_indices/{sampling_mode}/train-run{run_id}.npy')
            non_overlap_test_indices_path = os.path.join(label_root, f'non_overlap_indices/{sampling_mode}/test-run{run_id}.npy')
            non_overlap_train_indices = np.load(non_overlap_train_indices_path)
            non_overlap_test_indices = np.load(non_overlap_test_indices_path)
            label_train[pre_last_id + non_overlap_train_indices] += 1
            label_test[pre_last_id + non_overlap_test_indices] += 1
        if exclude_overlap:
            if run_id == 1:
                n_samples_per_run -= 5
            elif run_id == 8:
                n_samples_per_run -= 3
            else:
                n_samples_per_run -= 8
        pre_last_id += n_samples_per_run
    assert not ((label_test + label_train) == 2).any(), 'label_test and label_train have overlapped indices'
    assert np.sum(label_test + label_train) == np.sum(RUN_VOLUMES)-56, 'label_test and label_train have overlapped indices'
    return np.where(label_train>0)[0], np.where(label_test>0)[0]


def extract_sample_idxs(n_run_sample:int, sampling_mode:str)->np.ndarray:
    if sampling_mode == 'run':
        # n_run_sample個全て
        run_idxs = np.arange(n_run_sample)
    elif 'sandwitch' in sampling_mode:
        n_split = 5 # 分割数
        n_unit = n_run_sample // n_split # 各セグメントの個数
        n_start = ((n_split - 1) // 2) * n_unit # n_splitのうち前半の(n_split-1)/2はtrain
        n_end = (n_split - 1) * n_unit - n_start # last (n_split-1)/2はtrain。真ん中部分が残り、それがtest
        if 'train' in sampling_mode:
            r_start = np.arange(0, n_start)
            r_end = np.arange(n_run_sample - n_end, n_run_sample)
            run_idxs = np.concatenate([r_start, r_end], axis=0)
        else:  # 'test'
            r_middle = np.arange(n_start, n_run_sample - n_end)
            run_idxs = r_middle
    elif 'step' in sampling_mode:
        print('You select "step" sampling mode. This mode is not recommended because of the possibility of overfitting.')
        n_skip = 10
        run_all_idxs = np.arange(n_run_sample)
        run_test_idxs = np.arange(n_skip - 1, n_run_sample, step=n_skip)
        run_val_idxs = np.arange(n_skip - 2, n_run_sample, step=n_skip)
        run_holdout_idxs = np.concatenate([run_test_idxs, run_val_idxs], axis=0)
        run_train_idxs = np.setdiff1d(run_all_idxs, run_holdout_idxs)
        if 'train-excl-val' in sampling_mode:
            run_idxs = run_train_idxs
        elif 'train' in sampling_mode:
            run_idxs = np.concatenate([run_train_idxs, run_val_idxs], axis=0)
        elif 'val' in sampling_mode:
            run_idxs = run_val_idxs
        else:  # 'test'
            run_idxs = run_test_idxs
    return run_idxs

def split_train_and_test(n_samples_per_run:int, mode:str)->Tuple[np.ndarray, np.ndarray]:
    """_summary_
    split each run data into train and test
    Args:
        n_samples_per_run (int):
        mode (str): sandwitch, step

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    train_indices = extract_sample_idxs(n_samples_per_run, mode+'-train')
    test_indices = extract_sample_idxs(n_samples_per_run, mode+'-test')

    return train_indices, test_indices

def remove_overlapped_frame(n_samples_per_run:int, run_id:int)->np.ndarray:
    # see Michael Hanke, 2014, Scientific Data. Fig3 (a).
    assert RUN_VOLUMES[run_id-1] == n_samples_per_run, 'RUN_VOLUMES[run_id-1] != n_samples_per_run'
    indices = np.arange(n_samples_per_run)
    if run_id == 1:
        indices = indices[:-5]
    elif run_id == 8:
        indices = indices[3:]
    else:
        indices = indices[3:-5]
    return indices



if __name__ == '__main__':
    # create exluded overlapped frame indices
    # create split indices and save
    mode = 'sandwitch'
    for run_id in RUN_IDS:
        n_samples_per_run = RUN_VOLUMES[run_id-1]
        # remove overlapped frame
        non_overlap_indices = remove_overlapped_frame(n_samples_per_run, run_id)
        non_overlap_indices_path = os.path.join(LABEL_ROOT, f'non_overlap_indices/full-run{run_id}.npy')
        np.save(non_overlap_indices_path, non_overlap_indices)
        # split train and test
        train_indices, test_indices = split_train_and_test(len(non_overlap_indices), mode)
        assert len(train_indices) + len(test_indices) == len(non_overlap_indices), f'{len(train_indices)} + {len(test_indices)} != {len(non_overlap_indices)}'
        non_overlap_train_indices_path = os.path.join(LABEL_ROOT, f'non_overlap_indices/{mode}/train-run{run_id}.npy')
        non_overlap_test_indices_path = os.path.join(LABEL_ROOT, f'non_overlap_indices/{mode}/test-run{run_id}.npy')
        np.save(non_overlap_train_indices_path, non_overlap_indices[train_indices])
        np.save(non_overlap_test_indices_path, non_overlap_indices[test_indices])
        print('run{}: full: {}, train: {}, test: {}'.format(run_id, len(non_overlap_indices), len(train_indices), len(test_indices)))

    """SANDWITCH
    run1: full: 446, train: 356, test: 90
    run2: full: 433, train: 344, test: 89
    run3: full: 430, train: 344, test: 86
    run4: full: 480, train: 384, test: 96
    run5: full: 454, train: 360, test: 94
    run6: full: 431, train: 344, test: 87
    run7: full: 534, train: 424, test: 110
    run8: full: 335, train: 268, test: 67
    """