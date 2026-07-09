import pickle

from align_utils import obtain_mask
from nltools.data import Brain_Data, Design_Matrix
from nltools.mask import expand_mask
from nltools.stats import zscore
import numpy as np
import pandas as pd
import scipy.stats

from atlas_definition import parc_path


def make_covariates(cv, tr):
    z_cv = zscore(cv)
    all_cv = pd.concat([z_cv, z_cv**2, z_cv.diff(), z_cv.diff()**2], axis=1)
    all_cv.fillna(value=0, inplace=True)
    return Design_Matrix(all_cv, sampling_freq=1/tr)


def make_design_matrix(covariates, tr):
    # mc = covariates[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]
    # csf = covariates['csf']  # use csf mask? zscore?
    # wm = covariates['white_matter']
    # gs = covariates['global_signal']
    # cv_names = []
    cv_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter', 'global_signal']
    # use aroma_motion?
    # for cv_name in covariates.columns.values:
    #     if 'aroma' in cv_name:
    #         cv_names.append(cv_name)
    cv = covariates[cv_names]
    cov = make_covariates(cv, tr)
    # dm = Design_Matrix(pd.concat([csf, mc_cov], axis=1), sampling_freq=1/tr)  # add spike?
    dm = Design_Matrix(cov, sampling_freq=1/tr)
    dm = dm.add_poly(order=2, include_lower=True)
    return dm


def preprocess(bold, confound, tr, output_zscore=True, output_write=True, output_bold='preproc.nii.gz'):
    data = Brain_Data(bold)
    covariates = pd.read_csv(confound, sep='\t')
    # print(' Loaded data and covariates')

    dm = make_design_matrix(covariates, tr)
    # print(' Created design matrix for regression')
    data.X = dm
    stats = data.regress()
    # print(' Regressed out confounds')
    out = stats['residual']
    out = out.filter(sampling_freq=1/tr, high_pass=0.00667, low_pass=0.1)  # band pass filter
    # perform smoothing?
    # print(' Filtered data')
    # print(' Denoised data')

    if output_zscore:
        out = out.standardize(axis=0, method='zscore')  # along observation
        # print(' Z-scored data')
    out.data = np.float32(out.data)  # reduced size?
    if output_write:
        out.write(output_bold)
        # print(' Saved file')

    return out


def preprocess_roi(rois, denoised_bold, output_zscore=True, output_write=True, output_roi_bold='preproc_roi.bin'):
    n_roi = len(rois)
    n_sample = denoised_bold.shape()[0]
    roi_names = []
    roi_datas = np.zeros((n_roi, n_sample), dtype=float)
    mask = Brain_Data(parc_path)
    mask_x = expand_mask(mask)
    for r, roi in enumerate(rois):
        roi_idxs = np.array(roi['idxs'], dtype=int) - 1
        roi_name = str(r) + "_" + roi['name']
        roi_names.append(roi_name)
        roi_mask = obtain_mask(mask_x, roi_idxs, mask_id=roi_name)
        # print(type(roi_mask), roi_mask.shape())
        # print(np.unique(roi_mask.data))
        roi_data = denoised_bold.extract_roi(mask=roi_mask)  # mean activity
        # print(type(roi_data), roi_data.shape)
        if output_zscore:
            roi_data = scipy.stats.zscore(roi_data, axis=0)  # along observation
        roi_datas[r,:] = roi_data.copy()

    if output_write:
        with open(output_roi_bold, 'wb') as p:
            pickle.dump((roi_names, roi_datas), p)

    return roi_names, roi_datas
