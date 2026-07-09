import argparse
import itertools
import math
from multiprocessing import Pool
import os
import pickle
import time
import warnings

from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt
from nltools.mask import expand_mask
from nltools.data import Brain_Data
# from nltools.stats import align
from sklearn.decomposition import PCA

# from align_utils import obtain_mask, calc_irc, compare_irc, draw_heatmap, align_2_rois, plot_dims
from align_utils import obtain_mask, plot_dims
from atlas_definition import parc_path, parc_path_lr, ROIs_22, ROIs_22_LR, ROIs_22_sub, ROIs_180, ROIs_180_LR, ROIs_180_sub

# warnings.simplefilter('ignore')
warnings.simplefilter('ignore', category=RuntimeWarning)
warnings.simplefilter('ignore', category=FutureWarning)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare NPZ files of fmri data")

    parser.add_argument('--side', type=str, choices=['', 'LR'], default='')
    parser.add_argument('--n-roi', type=int, choices=[22, 180], default=22)

    return parser.parse_args()

def prepare_npz(bold_base, int_base, npz_base, sub, side, n_roi, trs, data_runs, tr_sec):
    n_run = len(trs)
    runs = [r+1 for r in range(n_run)]
    data_idxs = []
    for data_run in data_runs:
        if data_run > 1:
            run_start_idx = np.sum(trs[:(data_run-1)])
        else:
            run_start_idx = 0
        run_stop_idx = np.sum(trs[:data_run])
        run_idxs = [r for r in range(run_start_idx, run_stop_idx)]
        data_idxs.extend(run_idxs)
    data_idxs = np.array(data_idxs)
    # print(data_idxs.shape, data_idxs)

    # n_roi = len(ROIs)
    if side == 'LR':
        if n_roi == 22:
            ROIs = ROIs_22_LR
        elif n_roi == 180:
            ROIs = ROIs_180_LR
        else:
            raise NotImplementedError()
        n_roi = 2 * n_roi
        parcellation = parc_path_lr
    else:
        if n_roi == 22:
            ROIs = ROIs_22
        elif n_roi == 180:
            ROIs = ROIs_180
        else:
            raise NotImplementedError()
        parcellation = parc_path  # このparcellationとparc_path_lrは微妙に異なる->と思ったが左右のsetをとると一致

    # Regenerate intermediate files if params change (other than n_feature). Also depends on ROI definition.
    cond_str_0 = f"sub{sub}_"
    cond_str_1 = cond_str_0 + f"side{side}_roi{n_roi}_"
    cond_str_2 = cond_str_1 + f"data-{''.join([str(r) for r in data_runs])}"
    cond_str = cond_str_2
    print(cond_str)

    data_base_fn = int_base + cond_str_0 + "base.bin"
    data_rois_fn = int_base + cond_str_1 + "rois.bin"
    all_data_rois_fn = int_base + cond_str_1 + "all.bin"
    extract_data_rois_fn = npz_base + cond_str_2 + ".npz"

    # Load data
    print("Loading data")
    if os.path.isfile(data_base_fn):
        with open(data_base_fn, 'rb') as p:
            data_base = pickle.load(p)
    else:
        data_base = []
        for run in tqdm(runs):
            # print(run)
            bold_path = bold_base + f'sub-{sub}_run-{run}_space-MNI152NLin2009cAsym_bold.nii.gz'
            # print("Loading", bold_path)
            data = Brain_Data(bold_path)
            # print(data.shape())
            data_base.append(data)
        with open(data_base_fn, 'wb') as p:
            pickle.dump(data_base, p)

    print("Loading ROI data")
    if os.path.isfile(data_rois_fn) and os.path.isfile(all_data_rois_fn):
        with open(data_rois_fn, 'rb') as p:
            data_rois_ = pickle.load(p)
        roi_names, n_dims = data_rois_
        with open(all_data_rois_fn, 'rb') as p:
            all_data_rois = pickle.load(p)
    else:
        roi_names, n_dims = [], []
        all_data_rois = []
        # Load mask
        # mask = Brain_Data(parc_path)

        mask = Brain_Data(parcellation)
        # import pdb; pdb.set_trace()
        mask_x = expand_mask(mask)
        # Run this portion in parallel?
        for roi_i, roi in enumerate(tqdm(ROIs)):
        # for roi_i, roi in enumerate(ROIs):
            # Load ROI mask
            roi_idxs = np.array(roi['idxs'], dtype=int) - 1
            roi_name = str(roi_i) + "_" + roi['name']
            roi_names.append(roi_name)
            # print(f"{roi_name} has {len(roi_idxs)} ROIs")
            roi_mask = obtain_mask(mask_x, roi_idxs, mask_id=roi_name)

            # Load ROI data
            data_rois = []
            # run_means, run_stds = [], []
            for run_i in range(n_run):
                data = data_base[run_i].copy()
                data_roi = data.apply_mask(roi_mask)
                # Normalization
                # print(data_roi.data.shape)
                # import pdb; pdb.set_trace()
                if True :#False: # inoue@20240110
                    data_roi_mean = np.mean(data_roi.data, axis=0, keepdims=True) # time x voxel
                    data_roi_std = np.std(data_roi.data, axis=0, keepdims=True)
                    # print(data_roi_mean, data_roi_std)
                    # run_means.append(data_roi_mean)
                    # run_stds.append(data_roi_std)
                    data_roi.data = (data_roi.data - data_roi_mean) / data_roi_std
                # print(data_roi.data.shape)
                n_dim = data_roi.shape()[1]
                if run_i == 0:
                    n_dims.append(n_dim)
                # print(f" # of dim: {n_dim}")
                data_rois.append(data_roi)
            # run_means = np.concatenate(run_means, axis=0)
            # run_stds = np.concatenate(run_stds, axis=0)
            # run_means_stat = (np.max(run_means, axis=0, keepdims=True) - np.min(run_means, axis=0, keepdims=True)) / np.mean(run_means, axis=0, keepdims=True)
            # run_stds_stat = (np.max(run_stds, axis=0, keepdims=True) - np.min(run_stds, axis=0, keepdims=True)) / np.mean(run_stds, axis=0, keepdims=True)
            # print('MEAN across runs:', np.min(run_means_stat), np.max(run_means_stat), np.mean(run_means_stat), np.std(run_means_stat))
            # print('STD across runs:', np.min(run_stds_stat), np.max(run_stds_stat), np.mean(run_stds_stat), np.std(run_stds_stat))

            # Save intermediate data here?
            # print(len(data_rois))
            # all_data_roi = data_rois[0].copy()
            # print(all_data_roi.shape())
            # for run_i in range(1, n_run):
                # print(run_i)
                # print(data_rois[run_i].shape())
            #     all_data_roi.append(data_rois[run_i].copy())
            #     print(all_data_roi.shape())
            # LIFE
            # all_data_roi = data_rois[0].append(data_rois[1]).append(data_rois[2]).append(data_rois[3])
            # forrest
            all_data_roi = data_rois[0].append(data_rois[1]).append(data_rois[2]).append(data_rois[3]).append(data_rois[4]).append(data_rois[5]).append(data_rois[6]).append(data_rois[7])
            # print(all_data_roi.shape())
            all_data_rois.append(all_data_roi)

        data_rois_ = (roi_names, n_dims)
        with open(data_rois_fn, 'wb') as p:
            pickle.dump(data_rois_, p)
        with open(all_data_rois_fn, 'wb') as p:
            pickle.dump(all_data_rois, p)

    # print(type(roi_names), len(roi_names))
    roi_names = np.array(roi_names, dtype=object)
    # print(type(roi_names), roi_names.dtype, roi_names.shape)

    n_dims_str = "".join([str(n_dim) + ", " for n_dim in n_dims])
    print("# of dim:", n_dims_str)
    # plot_dims(n_dims, cond_str)

    print("Extracting ROI data")
    if False :#os.path.isfile(extract_data_rois_fn):
        pass
    else:
        # extract_data_rois = []
        extract_data_rois = np.empty(n_roi, dtype=object)
        # print(extract_data_rois)
        for roi_i, roi in enumerate(ROIs):
            all_data_roi = all_data_rois[roi_i]
            # print(all_data_roi.data.shape)
            # extract_data_roi = all_data_roi[data_idxs[0]:data_idxs[1]].copy()
            extract_data_roi = all_data_roi[data_idxs].copy()

            # extract_data_rois.append(extract_data_roi)
            # print(type(extract_data_roi.data), extract_data_roi.data.dtype, extract_data_roi.data.shape)
            # extract_data_rois.append(extract_data_roi.data)
            extract_data_rois[roi_i] = extract_data_roi.data

        # print(type(extract_data_rois), extract_data_rois.dtype, extract_data_rois.shape)

        # with open(extract_data_rois_fn, 'wb') as p:
        #     pickle.dump(extract_data_rois, p)
        trs = np.array(trs, dtype=int)
        np.savez(extract_data_rois_fn, roi_data=extract_data_rois, roi_name=roi_names, n_scan_per_run=trs, tr=tr_sec)

    # print("Done!")
    # delete variables to free up memory


def main():
    # ROI definition
    args = parse_arguments()
    side = args.side
    n_roi = args.n_roi

    # bold_base = '../../data/life/fmri_2/bold/'
    # bold_base = '../../data_transfer/life/fmri_2/bold/'
    bold_base = '../../data/forrestgump/studyforrest/bold/' # '../../data_transfer/studyforrest/fmri/bold/'
    # int_base = '../../data/life/fmri_2/bold_int/'
    # int_base = '../../data_transfer/life/fmri_2/bold_int/'
    int_base = '../../data/forrestgump/studyforrest/bold_int/'

    # subs = ['01']
    # LIFE
    # subs = ['01', '05', '06', '09', '12', '14', '17', '19', '20', '24', '27', '31', '32', '33', '34', '36', '37', '38', '41']  # 19
    # subs = ['01', '05', '06', '09', '12', '14', '17', '19', '24', '27', '31', '32', '33', '34', '36', '37', '38', '41']  # 18
    # forrest
    subs = ['01']# , '02', '03', '04', '06', '10', '14', '15', '16', '17', '18', '19']
    n_sub = len(subs)

    # npz_base = f'../../data/life/fmri_2/roi{n_roi}_np/'
    # LIFE
    # npz_base = f'../../data_transfer/life/fmri_2/roi{n_roi}_np/'
    # forrest
    npz_base = f'../../data/forrestgump/studyforrest/fmri/roi{n_roi}{side}_np_unnorm/'
    os.makedirs(npz_base, exist_ok=True)
    # LIFE
    # run 1: 374, run 2: 346, run 3: 377, run 4: 412
    # except sub '20' with 366 for run 1 and 338 for run 2
    # TR = 2.5
    # tr_1 = 374
    # tr_2 = 346
    # tr_3 = 377
    # tr_4 = 412
    # trs = [tr_1, tr_2, tr_3, tr_4]
    # data_runs = [1, 2, 3, 4]
    # forrest
    TR = 2.0
    tr_1 = 451
    tr_2 = 441
    tr_3 = 438
    tr_4 = 488
    tr_5 = 462
    tr_6 = 439
    tr_7 = 542
    tr_8 = 338
    trs = [tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7, tr_8]
    data_runs = [1, 2, 3, 4, 5, 6, 7, 8]
    for s, sub in enumerate(subs):
        print(f"Subject {sub}, {s + 1} / {n_sub}")
        prepare_npz(bold_base, int_base, npz_base, sub, side, n_roi, trs, data_runs, TR)


if __name__ == '__main__':
    main()
