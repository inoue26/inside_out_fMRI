"""_summary_
predict sbj Static FC (refined)
FC値の定義:
    FC = mean( | corr( srcROIの全voxel BOLD, tgtROIの全voxel BOLD ) の部分行列 | )
      - 相関行列 original_fc は BOLD-BOLD 相関を fast_correlation でその場計算
      - 行(src)を bold2feat_strict_ree2_voxels の voxel で選択
      - 列(tgt)を bold2location(delay2) の voxel で選択
"""

import os
import numpy as np
from typing import List
from voluntary_fixation.envs import SUBJECT_IDS, SAVE_ROOT, NUM_ROIS
from voluntary_fixation.functional_connectivity.utils import get_prediciton, prepare_bold_dataset


def ensure_static_fc_bold(meaning_roi: int, move_rois: List, n_components_bold_src: int,
                          sampling_mode: str, pca: bool, pca_str: str = None, delay: int = 0):
    """raw_correlation に必要な test-split BOLD (gt-roi_src / gt-roi_tgt-test) が無ければ生成する。

    train_static_fc_nopca.py を回さなくても済むように、回帰の副産物ではなく
    prepare_bold_dataset() から直接 ROI 別 BOLD (全 voxel, test split) を書き出す。
    nopca かつ normalize なしのとき、train_static_fc_nopca.py が保存する
        gt-roi_src{src}.npy      == bold_test_src[src]
        gt-roi_tgt{tgt}-test.npy == y_true_test == bold_test_tgt[tgt]
    と一致する。
    """
    assert delay == 0
    if pca_str is None:
        pca_str = 'pca' if pca else 'nopca'

    for sbj in SUBJECT_IDS:
        fc_dir = os.path.join(SAVE_ROOT, 'static_fc', pca_str, f'delay{delay}',
                              f'{sampling_mode}-ncb{n_components_bold_src}', sbj)
        src_savefile = os.path.join(fc_dir, f'gt-roi_src{meaning_roi}.npy')
        tgt_savefiles = {r: os.path.join(fc_dir, f'gt-roi_tgt{r}-test.npy') for r in move_rois}

        # 必要ファイルが全て揃っていればスキップ
        if os.path.exists(src_savefile) and all(os.path.exists(p) for p in tgt_savefiles.values()):
            print(f'[ensure_static_fc_bold] sbj {sbj} already exists in {fc_dir}, skip generating BOLD')
            continue

        # 生成: test split の ROI 別 BOLD (全 voxel)
        print(f'[ensure_static_fc_bold] generating BOLD for sbj {sbj} into {fc_dir}')
        os.makedirs(fc_dir, exist_ok=True)
        _, bold_test_src, _, bold_test_tgt = prepare_bold_dataset(sbj, delay, sampling_mode, rois=None)

        if not os.path.exists(src_savefile):
            np.save(src_savefile, bold_test_src[meaning_roi])
        for r, p in tgt_savefiles.items():
            if not os.path.exists(p):
                np.save(p, bold_test_tgt[r])


def fast_correlation(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    '''Calculate correlation matrix'''
    # time x voxels
    n_samples = x_train.shape[0]
    x_train_norm = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    y_train_norm = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
    corr_mat = np.einsum('ij,ik->jk', x_train_norm, y_train_norm) / n_samples  # x_voxel x y_voxel
    return corr_mat  # x_voxel x y_voxel


def fc_definition(grids: List, meaning_roi: int, move_rois: List, tgt_rois: List,
                  n_components_bold_src: int, sampling_mode: str, pca: bool, normalize_by_other_rois: bool,
                  src_delay:int, tgt_delay:int,pca_str: str = None, **kwargs) -> np.ndarray:

    # 必要な test-split BOLD (gt-roi_src / gt-roi_tgt-test) が無ければ生成する
    ensure_static_fc_bold(meaning_roi, move_rois, n_components_bold_src, sampling_mode, pca, pca_str)

    # raw_correlation モードなので pred/coef/pca_loadings は読まず BOLD (gt-roi_src, gt-roi_tgt) のみ取得
    use_dummy_for_pred = True
    src_sbjs, pred_sbjs, gt_sbjs, high_corr_idx_sbjs, coef_sbjs = get_prediciton(
        meaning_roi, move_rois, n_components_bold_src, sampling_mode, 0,
        pca=pca, pca_str=pca_str, use_dummy_for_pred=use_dummy_for_pred)

    # ---- 1. BOLD-BOLD 相関行列 original_fc を計算（src ROI voxels x tgt ROI voxels）----
    for sbj in range(len(SUBJECT_IDS)):
        for t in range(len(move_rois)):
            tmp_savedir = './tmp/raw_correlation/test'
            os.makedirs(tmp_savedir, exist_ok=True)
            if move_rois[t] == meaning_roi:
                test_bold_corr_mat_savefile = os.path.join(tmp_savedir, f'{sbj}_s{meaning_roi}_t_as_s.npy')
            else:
                if (meaning_roi in move_rois and meaning_roi < t):
                    test_bold_corr_mat_savefile = os.path.join(tmp_savedir, f'{sbj}_s{meaning_roi}_t{t-1}.npy')
                else:
                    test_bold_corr_mat_savefile = os.path.join(tmp_savedir, f'{sbj}_s{meaning_roi}_t{t}.npy')

            if os.path.exists(test_bold_corr_mat_savefile):
                coef = np.load(test_bold_corr_mat_savefile)
            else:
                coef = fast_correlation(src_sbjs[sbj][0], gt_sbjs[sbj][t])
                np.save(test_bold_corr_mat_savefile, coef)

            coef_sbjs[sbj][t] = coef  # (src_voxels, tgt_voxels)

    # ---- 2. voxel selection して mean(|corr|) を FC 値にする ----
    print('calculating dynamic FC of each grids...')
    static_fcs_sbjs_grids = np.zeros((len(grids), len(SUBJECT_IDS), len(move_rois)))  # grids x sbjs x tgt_rois
    for sbj in range(len(SUBJECT_IDS)):
        for t in range(len(move_rois)):
            original_fc = coef_sbjs[sbj][t]  # (src_voxels, tgt_voxels)
            for g, topk in enumerate(grids):
                hidden_layer_id = -1
                # src voxel: BOLD -> visual feature エンコーディングで選ばれた voxel
                src_voxel_path = os.path.join(SAVE_ROOT, f'bold2feat_strict_ree2_voxels/delay{src_delay}-fo0-mo0/pca2-masked_image-segment-remove_brightness-remove_empty_eyetrack/{SUBJECT_IDS[sbj]}/voxels-roi_{meaning_roi}.npy')
                src_feat_voxel_indices = np.unique(np.load(src_voxel_path)[hidden_layer_id])
                # tgt voxel: BOLD -> gaze location デコーディングで選ばれた voxel
                tgt_voxel_path = os.path.join(SAVE_ROOT, f'bold2location/delay{tgt_delay}-FIXA_PURS/segment-ncb250-rm_empty_eyetrack/{SUBJECT_IDS[sbj]}/voxels-roi_{move_rois[t]}.npy')
                print('tgt voxel path: ', tgt_voxel_path)
                tgt_voxel_indices = np.unique(np.load(tgt_voxel_path))

                fc = original_fc
                fc = original_fc[:, tgt_voxel_indices]   # 列(tgt)選択
                fc = fc[src_feat_voxel_indices, :]       # 行(src)選択
                fc = np.abs(fc)
                topk_fc = np.mean(fc)

                static_fcs_sbjs_grids[g, sbj, t] = np.mean(np.abs(topk_fc))

    static_fcs_sbjs_grids = static_fcs_sbjs_grids.reshape((len(grids), len(SUBJECT_IDS), -1))  # grids x sbjs x n_features

    # ---- 3. 正規化・tgt ROI 抽出 ----
    tgt_roi_idx = [move_rois.index(r) for r in tgt_rois]
    if normalize_by_other_rois:
        # 全 tgt ROI 平均で割る（他ROIとの相対値化）
        static_fcs_sbjs_grids_ = np.abs(static_fcs_sbjs_grids)
        static_fcs_sbjs_grids = static_fcs_sbjs_grids_.copy()
        static_fcs_sbjs_grids /= static_fcs_sbjs_grids.mean(axis=-1, keepdims=True)
        static_fcs_sbjs_grids = static_fcs_sbjs_grids[:, :, tgt_roi_idx]
    else:
        static_fcs_sbjs_grids = static_fcs_sbjs_grids[:, :, tgt_roi_idx]

    if static_fcs_sbjs_grids.shape[-1] == 2:
        static_fcs_sbjs_grids = static_fcs_sbjs_grids[:, :, 1] - static_fcs_sbjs_grids[:, :, 0]  # (sbjs, tgt_rois)
        static_fcs_sbjs_grids = static_fcs_sbjs_grids[:, :, np.newaxis]
    elif static_fcs_sbjs_grids.shape[-1] == 1:
        pass
    else:
        raise ValueError(f'static_fcs_sbjs_grids.shape[-1] should be 1 or 2, {static_fcs_sbjs_grids.shape[-1]}')
    return static_fcs_sbjs_grids
