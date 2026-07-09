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

from align_utils import obtain_mask, calc_irc, compare_irc, draw_heatmap, align_2_rois, plot_dims
from atlas_definition import parc_path, ROIs_22, ROIs_22_sub, ROIs_180, ROIs_180_sub

# warnings.simplefilter('ignore')
warnings.simplefilter('ignore', category=RuntimeWarning)
warnings.simplefilter('ignore', category=FutureWarning)


def evaluate_alignment(bold_base, sub, ROIs, train_idxs, test_idxs, n_latent, n_feature):
    runs = [1, 2, 3, 4, 5, 6, 7, 8]  # fixed
    n_run = len(runs)

    n_roi = len(ROIs)

    if n_latent > 0:
        assert n_latent >= n_feature

    # Regenerate intermediate files if params change (other than n_feature). Also depends on ROI definition.
    cond_str_0 = f"sub{sub}_"
    cond_str_1 = cond_str_0 + f"roi{n_roi}_"
    cond_str_2 = cond_str_1 + f"train{train_idxs[0]}-{train_idxs[1] - 1}_test{test_idxs[0]}-{test_idxs[1] - 1}_dim{n_latent}_"
    cond_str = cond_str_2 + f"feat{n_feature}_"
    print(cond_str)

    data_base_fn = cond_str_0 + "base.bin"
    data_rois_fn = cond_str_1 + "rois.bin"
    all_data_rois_fn = cond_str_1 + "all.bin"
    evr_rois_fn = cond_str_2 + "evr.bin"
    train_data_rois_fn = cond_str_2 + "train.bin"
    test_data_rois_fn = cond_str_2 + "test.bin"

    # save parameters for analysis
    params_fn = cond_str + "params.npz"
    np.savez(params_fn, sub=sub, train_idxs=train_idxs, test_idxs=test_idxs, n_latent=n_latent, n_feature=n_feature)

    # Load data
    print("Loading data")
    if os.path.isfile(data_base_fn):
        with open(data_base_fn, 'rb') as p:
            data_base = pickle.load(p)
    else:
        data_base = []
        for run in tqdm(runs):
            bold_path = bold_base + f'sub-{sub}_run-{run}_space-MNI152NLin2009cAsym_bold.nii.gz'
            # print("Loading", bold_path)
            data = Brain_Data(bold_path)
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
        mask = Brain_Data(parc_path)
        mask_x = expand_mask(mask)
        # Run this portion in parallel?
        for r, roi in enumerate(tqdm(ROIs)):
            # Load ROI mask
            roi_idxs = np.array(roi['idxs'], dtype=int) - 1
            roi_name = str(r) + "_" + roi['name']
            roi_names.append(roi_name)
            # print(f"{roi_name} has {len(roi_idxs)} ROIs")
            roi_mask = obtain_mask(mask_x, roi_idxs, mask_id=roi_name)

            # Load ROI data
            data_rois = []
            for i in range(n_run):
                data = data_base[i]
                data_roi = data.apply_mask(roi_mask)
                n_dim = data_roi.shape()[1]
                if i == 0:
                    n_dims.append(n_dim)
                # print(f" # of dim: {n_dim}")
                data_roi.data = data_roi.data[3:, :]  # tmp, discard = 3 (6s for TR 2s)
                data_rois.append(data_roi)

            # Save intermediate data here?
            all_data_roi = data_rois[0].append(data_rois[1]).append(data_rois[2]).append(data_rois[3]).append(data_rois[4]).append(data_rois[5]).append(data_rois[6]).append(data_rois[7])
            all_data_rois.append(all_data_roi)

        data_rois_ = (roi_names, n_dims)
        with open(data_rois_fn, 'wb') as p:
            pickle.dump(data_rois_, p)
        with open(all_data_rois_fn, 'wb') as p:
            pickle.dump(all_data_rois, p)

    n_dims_str = "".join([str(n_dim) + ", " for n_dim in n_dims])
    print("# of dim:", n_dims_str)
    # plot_dims(n_dims, cond_str)

    print("Extracting ROI data")
    if os.path.isfile(evr_rois_fn) and os.path.isfile(train_data_rois_fn) and os.path.isfile(test_data_rois_fn):
        pass
    else:
        evr_rois = []  # not required anymore?
        train_data_rois, test_data_rois = [], []
        for r, roi in enumerate(ROIs):
            all_data_roi = all_data_rois[r]
            # train-test split
            train_data_roi = all_data_roi[train_idxs[0]:train_idxs[1]].copy()
            test_data_roi = all_data_roi[test_idxs[0]:test_idxs[1]].copy()

            # dimensionality reduction
            if n_latent > 0:
                if n_latent > n_dim:
                    n_pad_dim = n_latent - n_dim
                    n_train = train_data_roi.shape()[0]
                    n_test = test_data_roi.shape()[0]
                    train_data_roi.data = np.concatenate([train_data_roi.data, np.zeros([n_train, n_pad_dim])], axis=1)
                    test_data_roi.data = np.concatenate([test_data_roi.data, np.zeros([n_test, n_pad_dim])], axis=1)

                pca_roi = PCA(n_components=n_latent)
                latent_train = pca_roi.fit_transform(train_data_roi.data)
                evr_roi = np.sum(pca_roi.explained_variance_ratio_)
                evr_rois.append(evr_roi)
                print(f" EVR: {evr_roi}")
                train_data_roi.data = latent_train
                latent_test = pca_roi.transform(test_data_roi.data)
                test_data_roi.data = latent_test
            else:
                evr_rois.append(-1)

            train_data_rois.append(train_data_roi)
            test_data_rois.append(test_data_roi)

        with open(evr_rois_fn, 'wb') as p:
            pickle.dump(evr_rois, p)
        with open(train_data_rois_fn, 'wb') as p:
            pickle.dump(train_data_rois, p)
        with open(test_data_rois_fn, 'wb') as p:
            pickle.dump(test_data_rois, p)

    isc_train_raw_means = np.zeros([n_roi, n_roi], dtype=float)
    isc_test_raw_learnt_means = np.zeros([n_roi, n_roi], dtype=float)
    isc_test_raw_means = np.zeros([n_roi, n_roi], dtype=float)
    isc_train_means = np.zeros([n_roi, n_roi], dtype=float)
    isc_test_means = np.zeros([n_roi, n_roi], dtype=float)
    roi_combs = list(itertools.product(range(n_roi), range(n_roi)))
    n_comb = len(roi_combs)
    cond_strs = [cond_str] * n_comb

    n_process = 32
    # n_process = 1
    # chunk_size = 4
    chunk_size = 1
    print("Running alignment")
    start = time.time()
    with Pool(n_process) as p:
        results = p.starmap(align_2_rois, tqdm(zip(cond_strs, roi_combs), total=len(roi_combs)), chunksize=chunk_size)

    t_s = math.ceil(time.time() - start)
    t_m, t_s = divmod(t_s, 60)
    t_h, t_m = divmod(t_m, 60)
    print(f"Total: {t_h} h {t_m} m {t_s} s")

    # print(results)
    # pack data
    for comb, (roi_1, roi_2) in enumerate(roi_combs):
        if roi_1 == roi_2:
            continue
        if roi_1 < roi_2:
            isc_train_raw_means[roi_1, roi_2] = results[comb][0]
            isc_test_raw_learnt_means[roi_1, roi_2] = results[comb][1]
            isc_test_raw_means[roi_1, roi_2] = results[comb][2]
            isc_train_means[roi_1, roi_2] = results[comb][3]
            isc_test_means[roi_1, roi_2] = results[comb][4]
        else:
            isc_train_raw_means[roi_1, roi_2] = isc_train_raw_means[roi_2, roi_1]
            isc_test_raw_learnt_means[roi_1, roi_2] = isc_test_raw_learnt_means[roi_2, roi_1]
            isc_test_raw_means[roi_1, roi_2] = isc_test_raw_means[roi_2, roi_1]
            isc_train_means[roi_1, roi_2] = isc_train_means[roi_2, roi_1]
            isc_test_means[roi_1, roi_2] = isc_test_means[roi_2, roi_1]

    # Save affinity matrix
    np.savez(cond_str + "affinity", train_raw=isc_train_raw_means, test_raw_learnt=isc_test_raw_learnt_means, test_raw=isc_test_raw_means, train=isc_train_means, test=isc_test_means)

    # Visualize heatmap
    draw_heatmap(isc_train_raw_means, range(n_roi), range(n_roi), cond_str + "irc-train-raw")
    draw_heatmap(isc_test_raw_learnt_means, range(n_roi), range(n_roi), cond_str + "irc-test-raw-learnt")
    draw_heatmap(isc_test_raw_means, range(n_roi), range(n_roi), cond_str + "irc-test-raw")
    draw_heatmap(isc_train_means, range(n_roi), range(n_roi), cond_str + "irc-train")
    draw_heatmap(isc_test_means, range(n_roi), range(n_roi), cond_str + "irc-test")

    # print("Done!")


def main():
    bold_base = './'

    subs = ['01']
    n_sub = len(subs)

    # ROI definition
    rois = ROIs_22
    # rois = ROIs_22_sub
    # rois = ROIs_180
    # rois = ROIs_180_sub
    # rois = rois[:2]  # for testing

    # tmp, discard = 3 (6s for TR 2s)
    tr_1 = 451 - 3
    tr_2 = 441 - 3
    tr_3 = 438 - 3
    tr_4 = 488 - 3
    tr_5 = 462 - 3
    tr_6 = 439 - 3
    tr_7 = 542 - 3
    tr_8 = 338 - 3
    tr_all = tr_1 + tr_2 + tr_3 + tr_4 + tr_5 + tr_6 + tr_7 + tr_8
    train_idxs = [0, tr_1 + tr_2 + tr_3 + tr_4 + tr_5 + tr_6]
    test_idxs = [tr_1 + tr_2 + tr_3 + tr_4 + tr_5 + tr_6, tr_all]

    # (Requires n_latent >= n_feature ?)
    n_latent = -1  # no PCA
    # n_latent = 100  # use PCA for dimensionality reduction

    # n_feature = -1  # use all features
    # n_feature = 25  # use some features
    # n_feature = 50
    # n_feature = 100
    n_feature = 200

    # can also differ whether to use 'rank' or 'pca' for irc
    for s, sub in enumerate(subs):
        print(f"Subject {sub}, {s + 1} / {n_sub}")
        evaluate_alignment(bold_base, sub, rois, train_idxs, test_idxs, n_latent, n_feature)


if __name__ == '__main__':
    main()
