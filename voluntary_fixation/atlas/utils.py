from nilearn import datasets, plotting, surface
from nltools.mask import expand_mask, collapse_mask
from nltools.data import Brain_Data
from tqdm import tqdm
from .atlas_definition import parc_path, parc_lh_path, parc_rh_path
import matplotlib.pyplot as plt
import numpy as np


def obtain_mask(mask_x, rois, mask_id='', viz=False):
    if len(rois) == 1:
        roi_mask = mask_x[rois[0]]
        # print(f'# of vs in {mask_id}: {np.sum(roi_mask.data)}')
    else:
        ms = []
        for roi in rois:
            m = mask_x[roi]
            # print(f'# of vs in {mask_id}: {np.sum(m.data)}')
            ms.append(m.data)
        mask = mask_x.copy()
        mask.data = np.stack(ms)
        roi_mask = collapse_mask(mask)
        roi_mask.data[roi_mask.data > 0] = 1  # for extract_roi

    if viz:
        roi_mask.plot()
        # plt.show()
        plt.savefig(f'align_mask_{mask_id}.png')
        plt.close()

    return roi_mask


def obtain_roi_surface(rois, parcellation=None):
    if parcellation is None:
        parcellation = parc_path
    mask = Brain_Data(parcellation)
    mask_x = expand_mask(mask)
    parc_lh = surface.load_surf_data(parc_lh_path)
    parc_rh = surface.load_surf_data(parc_rh_path)
    n_vertices_lh = parc_lh.shape[0]
    n_vertices_rh = parc_rh.shape[0]
    roi_v_lh_idxs, roi_v_rh_idxs = [], []
    roi_masks = []
    for r, roi in enumerate(tqdm(rois)):
        roi_idxs = roi['idxs']  # 1-start
        v_lh_idxs, v_rh_idxs = [], []
        for roi_idx in roi_idxs:
            v_lh_idx = list(np.where(parc_lh == roi_idx)[0])
            v_rh_idx = list(np.where(parc_rh == roi_idx)[0])
            v_lh_idxs.extend(v_lh_idx)
            v_rh_idxs.extend(v_rh_idx)
        roi_v_lh_idxs.append(v_lh_idxs)
        roi_v_rh_idxs.append(v_rh_idxs)

        roi_idxs = np.array(roi_idxs, dtype=int) - 1  # 0-start
        roi_mask = obtain_mask(mask_x, roi_idxs)
        # print(roi_mask)
        roi_masks.append(roi_mask)

    return roi_masks, roi_v_lh_idxs, roi_v_rh_idxs, n_vertices_lh, n_vertices_rh