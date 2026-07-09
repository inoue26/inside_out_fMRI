
import os
import numpy as np
from typing import List, Tuple
from voluntary_fixation.envs import SUBJECT_IDS, SAVE_ROOT, NUM_ROIS, RUN_VOLUMES
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from tqdm import tqdm
from voluntary_fixation.dataset.utils import delayed_label
from voluntary_fixation.dataset.bold_dataset import bold44_dateset, get_non_overlap_indices_for_concatenate_data
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt

def get_prediciton2(src_roi:int, tgt_rois:List[int], n_components_bold_src:int, sampling_mode:str, delay:int=0, pca:bool=True):
    assert sampling_mode == 'segment', 'sampling_mode should be segment'
    assert delay == 0
    pred_sbjs = []
    gt_sbjs = []
    high_corr_idx_sbjs =[]
    coef_sbjs = []
    pca_str = 'pca' if pca else 'nopca'
    for sbj in SUBJECT_IDS:
        fc_dir = os.path.join(SAVE_ROOT, 'static_fc', pca_str, f'delay{delay}',
                                    f'{sampling_mode}-ncb{n_components_bold_src}', sbj)
        pred_sbj = []
        gt_sbj = []
        high_corr_idx_sbj = []
        coef_sbj = []
        for t, tgt_roi in enumerate(tgt_rois):
            gt = np.load(os.path.join(fc_dir, f'gt-roi_tgt{tgt_roi}-test.npy')) # n_samples x n_voxels
            gt_sbj.append(gt)
            pred = np.load(os.path.join(fc_dir, f'pred-roi_tgt{tgt_roi}-src{src_roi}-test.npy')) # n_samples x n_voxels
            pred_sbj.append(pred)
            high_corr_idx = np.load(os.path.join(fc_dir, f'pred-roi_tgt{tgt_roi}-src{src_roi}-top_train_corr_idx.npy'))
            high_corr_idx_sbj.append(high_corr_idx)
            coef = np.load(os.path.join(fc_dir, f'pred-roi_tgt{tgt_roi}-src{src_roi}-coef.npy')) # (src_voxels, tgt_voxels) (251, 1059)
            coef_sbj.append(coef)

        pred_sbjs.append(pred_sbj)
        gt_sbjs.append(gt_sbj)
        high_corr_idx_sbjs.append(high_corr_idx_sbj) # sbj x n_tgt_rois x n_voxels
        coef_sbjs.append(coef_sbj)
    return pred_sbjs, gt_sbjs, high_corr_idx_sbjs, coef_sbjs # sbj x n_tgt_rois x [n_samples x n_voxels]


def topk_selection(static_fcs:List[List[np.ndarray]], high_corr_idx_sbjs:List[np.ndarray], topk:int):
    # static_fcs: sbj x tgt_rois x [251 x tgt_voxels]
    n_sbjs = len(static_fcs)
    n_tgt_rois = len(static_fcs[0])
    n_src_voxels = static_fcs[0][0].shape[0] # including bias term
    min_size = np.min([len(static_fcs[0][0][t]) for t in range(n_tgt_rois)])
    if topk >= 0:
        # topk selection
        fcs = np.zeros((n_sbjs, n_tgt_rois, n_src_voxels, topk))
        for sbj in range(len(static_fcs)):
            for t in range(n_tgt_rois):
                # if control_shuffle:
                #     np.random.seed(seed)
                #     np.random.shuffle(high_corr_idx_sbjs[sbj][t])
                high_corr_idx = high_corr_idx_sbjs[sbj][t][:topk]
                fcs[sbj,t,:,:] = static_fcs[sbj][t][:, high_corr_idx] # 251 x topk
    else:
        # cut off min tgt voxel size
        fcs = np.zeros((n_sbjs, n_tgt_rois, n_src_voxels, min_size))
        for sbj in range(len(static_fcs)):
            for t in range(n_tgt_rois):
                # if control_shuffle:
                #     np.random.seed(seed)
                #     np.random.shuffle(high_corr_idx_sbjs[sbj][t])
                high_corr_idx = high_corr_idx_sbjs[sbj][t][:min_size]
                fcs[sbj,t,:,:] = static_fcs[sbj][t][:, high_corr_idx] # 251 x topk
    return fcs

def apply_grid_parameter(Xs:List[List[np.ndarray]], high_corr_idx_sbjs:List[np.ndarray], topk:int, svd_idx:int):
        # apply svd param
        newX = [X[svd_idx] for X in Xs] # num_sbjs-1 x tgt_rois x [251 x tgt_voxels]
        # apply topk param
        newX = topk_selection(newX, high_corr_idx_sbjs, topk) # num_sbjs-1 x tgt_rois x 251 x topk
        return newX # num_sbjs-1 x tgt_rois x 251 x topk

def grid_search_nested_k_fold_validation(Xs:np.ndarray, Y:np.ndarray, grids:List[int],
                                         alpha=1e-2, mode='ridge'):
    # Xs: grids x sbjs x tgt_rois
    # Ys: sbj
    # high_corr_idx_sbjs: sbj x tgt_rois x [tgt_voxels]
    num_sbjs = len(Xs[0])
    assert num_sbjs  == len(SUBJECT_IDS), f'n_sbjs should be equal to len(SUBJECT_IDS), {num_sbjs} != {len(SUBJECT_IDS)}'
    assert len(Xs) == len(grids)
    sbj_ids = list(range(num_sbjs))

    # grids
    best_grid_predYs = []
    trueYs = []
    coefs = []
    for test_sbj in sbj_ids:
        train_sbjs = sbj_ids.copy()
        train_sbjs.remove(test_sbj)
        trainXs = Xs[:,train_sbjs,:]# [Xs[i] for i in train_sbjs] # num_sbjs-1 x num_svd_grid x n_features
        trainY = Y[train_sbjs] # num_sbjs-1
        testXs = Xs[:,[test_sbj],:]# [Xs[test_sbj]] # num_svg_grid x n_features
        testY = [Y[test_sbj]] # 1
        # grid search
        grid_scores = []
        for g, grid in enumerate(grids):

            n_train_sbjs = len(train_sbjs)
            groups = np.arange(n_train_sbjs)
            gridX = trainXs[g]
            assert len(gridX) == n_train_sbjs
            assert gridX.ndim == 2
            gridY = trainY
            # nested validation
            if mode == 'corr':
                assert gridX.shape[1] == 1, 'in corr mode, gridX should be 2d (num_samples x 1) but got {}'.format(gridX.shape)
                grid_test_Ys, grid_test_preds = gridX[:,0], gridY
            else:
                grid_test_Ys, grid_test_preds = regression_k_fold_validation(gridX, gridY, n_train_sbjs, alpha, mode=mode, groups=groups)
            test_corr = np.corrcoef(grid_test_Ys, grid_test_preds)[0,1]
            test_mae_neg = -np.mean(np.abs(grid_test_Ys, grid_test_preds))
            grid_scores.append(test_corr)
            # grid_scores.append(test_mae_neg)

        # validation with best grid parameter
        best_grid_idx = np.argmax(grid_scores)
        grid = grids[best_grid_idx]
        best_param_X_train = trainXs[best_grid_idx]
        best_param_X_test = testXs[best_grid_idx]
        if mode == 'corr':
            coef_ = 1
            intercept_ = 0
            assert best_param_X_test.shape[1] == 1, 'in corr mode, gridX should be 2d (num_samples x 1) but got {}'.format(gridX.shape)
            pred_Y = best_param_X_test[:,0]
            best_param_X_test_transformed = best_param_X_test
            pred_wo_b = best_param_X_test
        else:
            pipeline = make_fitting_pipeline(mode, alpha)
            pipeline.fit(best_param_X_train, trainY)
            pred_Y = pipeline.predict(best_param_X_test)
            coef_ = pipeline[1].coef_
            intercept_ = pipeline[1].intercept_
            best_param_X_test_transformed = pipeline[0].transform(best_param_X_test)
            pred_wo_b = best_param_X_test_transformed@coef_
        trueYs.append(testY)
        best_grid_predYs.append(pred_Y)
        # pipeline[0].transform(best_param_X_test)@pipeline[1].coef_ + pipeline[1].intercept_
        print('===========================================')
        print('\tGRID: ', grid)
        print('\tBEST TEST: ', best_param_X_test,'\tBEST TEST(Z_score): ', best_param_X_test_transformed)
        print('\tRIDGE: ', coef_, intercept_, '+', pred_wo_b)
        coefs.append(coef_)
    # import pdb; pdb.set_trace()
    print('CORR: ', np.corrcoef(np.concatenate(trueYs, axis=0), np.concatenate(best_grid_predYs, axis=0))[0,1])
    return np.concatenate(trueYs, axis=0), np.concatenate(best_grid_predYs, axis=0), np.array(coefs)

def make_fitting_pipeline(mode, alpha):
    if mode=='lasso':
        reg = linear_model.Lasso(alpha=alpha)
    elif mode=='ridge':
        reg = linear_model.Ridge(alpha=alpha)
    else:
        raise ValueError('mode should be lasso or ridge')
    if alpha == 0:
        reg = linear_model.LinearRegression()

    pipeline = make_pipeline(StandardScaler(), reg)
    # pipeline = make_pipeline(reg)
    return pipeline

def regression_k_fold_validation(X, Y, k, alpha, mode='lasso', groups=None, kfold_shuffle:bool=False):
    # X: n_windows x n_features
    # Y: n_windows
    pipeline = make_fitting_pipeline(mode, alpha)# make_pipeline(StandardScaler(), reg)

    if groups is None:
        kf = KFold(n_splits=k, shuffle=kfold_shuffle, random_state=None)
        kf.get_n_splits(X)
        split = kf.split(X)
    else:
        kf = GroupKFold(n_splits=k)
        kf.get_n_splits(X, Y, groups)
        split = kf.split(X, Y, groups)

    test_Ys = []
    test_preds = []

    for i, (train_index, test_index) in enumerate(split):
        train_X = X[train_index,:]
        train_Y = Y[train_index]
        pipeline.fit(train_X, train_Y)
        # pred_Y_train = pipeline.predict(train_X)
        # print(train_Y[:10], pred_Y_train[:10])

        test_X = X[test_index,:]
        test_Y = Y[test_index]
        pred_Y = pipeline.predict(test_X)
        test_Ys.append(test_Y)
        test_preds.append(pred_Y)
    return np.concatenate(test_Ys, axis=0), np.concatenate(test_preds, axis=0)



def fit_behavior_on_staticFC(fcs:np.ndarray, behaviors:np.ndarray, grids:List[int], use_fit:bool=True):
    # fcs: grids x sbjs x tgt_rois
    # behaviors: n_sbjs
    n_sbjs = len(fcs[0])
    assert n_sbjs  == len(SUBJECT_IDS), f'n_sbjs should be equal to len(SUBJECT_IDS), {n_sbjs} != {len(SUBJECT_IDS)}'

    # import pdb; pdb.set_trace()
    if use_fit:
        test_Ys, test_preds, coefs = grid_search_nested_k_fold_validation(fcs, behaviors,
                                                grids, alpha=1e-2, mode='ridge')
    else:
        test_Ys, test_preds, coefs = grid_search_nested_k_fold_validation(fcs, behaviors,
                                                grids, alpha=1e-2, mode='corr')


    test_error = [np.abs(test_Ys[i] - test_preds[i]) for i in range(n_sbjs)]
    test_corr = np.corrcoef(test_Ys, test_preds)[0,1]
    # import pdb; pdb.set_trace()
    print(f'============================sbj:all selection========================')
    print(f"\tTest:  error:{np.mean(test_error):.3f}\t corr:{test_corr:.3f}")
    print('\tTest true:', [f'{test_Ys[i]:.3f}' for i in range(n_sbjs)])
    print('\tTest pred:', [f'{test_preds[i]:.3f}' for i in range(n_sbjs)])
    print('====================================================================')

    # import matplotlib.pyplot as plt
    # plt.scatter(test_preds, test_Ys)
    # plt.savefig('scatter.png')
    # plt.close()
    # import pdb; pdb.set_trace()
    return np.mean(test_error), test_corr, coefs.mean(), coefs.std()



def get_behavior(fcdir:str, moving_window:int, moving_step:int, saliency_TR_q:int, iou_q:int,
                 use_saliency:bool=False, behavior_mode:str='subtract', num_segments:int=2):
    # select_mode: topk, auto, svd
    # static_fcs: grids x sbjs x tgt_rois
    # high_corr_idx_sbjs: sbj x tgt_rois x [tgt_voxels]
    SAVEDIR = fcdir # os.path.join(SAVE_ROOT, fc_mode_name)
    concate_behavior = []
    if isinstance(num_segments, int):
        num_segments = list(range(num_segments))
    for seg in num_segments:
        w_saliency_path = os.path.join(SAVEDIR, 'saliency', f'w_sal_vs_wo_sal_{moving_window}TR_{moving_step}TR-iou{1}-sal{saliency_TR_q}-w_sal-seg{seg}.npy')
        wo_saliency_path = os.path.join(SAVEDIR, 'saliency', f'w_sal_vs_wo_sal_{moving_window}TR_{moving_step}TR-iou{1}-sal{saliency_TR_q}-wo_sal-seg{seg}.npy')
        w_behavior_path = os.path.join(SAVEDIR, 'behavior', f'w_eye_w_sal_vs_wo_sal_{moving_window}TR_{moving_step}TR-iou{iou_q}-sal{saliency_TR_q}-w_sal-seg{seg}.npy')
        wo_behavior_path = os.path.join(SAVEDIR, 'behavior', f'w_eye_w_sal_vs_wo_sal_{moving_window}TR_{moving_step}TR-iou{iou_q}-sal{saliency_TR_q}-wo_sal-seg{seg}.npy')

        w_saliency = np.load(w_saliency_path)
        wo_saliency = np.load(wo_saliency_path)
        w_behavior = np.load(w_behavior_path)
        wo_behavior = np.load(wo_behavior_path)
        eye_prob = w_behavior + wo_behavior
        if use_saliency:
            window_behavior = [w_saliency]
        else:
            if behavior_mode == 'subtract':
                window_behavior = [(wo_behavior/wo_saliency)-(w_behavior/w_saliency)]
            elif behavior_mode == 'divide':
                window_behavior = [(wo_behavior/wo_saliency)/(w_behavior/w_saliency)]
            elif behavior_mode == 'endogenous':
                window_behavior = [(wo_behavior/wo_saliency)/eye_prob]
            elif behavior_mode == 'exogenous':
                window_behavior = [(w_behavior/w_saliency)/eye_prob]
            elif behavior_mode == 'endogenous2':
                window_behavior = [(wo_behavior/wo_saliency)]
                # # window_behavior = np.array([4.0073891625615765, 3.9028132992327365, 2.7766497461928936, 3.299719887955182, 3.9164619164619165, 4.293670886075949, 2.9646464646464645, 3.7815533980582523, 3.700534759358289, 3.975609756097561, 3.8425, 3.861985472154964])
                # # window_behavior = np.array([3.936619718309859, 3.8194945848375452, 2.6066176470588234, 3.326771653543307, 3.8105263157894735, 4.237226277372263, 2.812274368231047, 3.6448275862068966, 3.376425855513308, 3.895104895104895, 3.667870036101083, 3.7448275862068967])
                # window_behavior = np.array([3.903225806451613, 3.2605042016806722, 2.4054054054054053, 3.0892857142857144, 3.4959349593495936, 3.7758620689655173, 2.2796610169491527, 3.0952380952380953, 2.725806451612903, 3.4634146341463414, 3.2222222222222223, 3.409448818897638])
                # window_behavior = window_behavior[:,np.newaxis]
                # window_behavior = [window_behavior]

            elif behavior_mode == 'exogenous2':
                window_behavior = [(w_behavior/w_saliency)]
                # window_behavior = np.array([4.129113924050633, 4.370843989769821, 3.237851662404092, 3.1012658227848102, 4.215189873417722, 3.9792746113989637, 3.207161125319693, 4.281725888324873, 4.292397660818714, 4.100755667506297, 3.9645569620253163, 4.0227272727272725])
                # window_behavior = np.array([4.240875912408759, 4.3127272727272725, 3.422794117647059, 3.018018018018018, 4.317689530685921, 3.962825278810409, 3.2169117647058822, 4.2644927536231885, 4.366666666666666, 4.086642599277979, 3.9054545454545453, 3.9890909090909092])
                # window_behavior = np.array([4.350746268656716, 4.2518518518518515, 3.511111111111111, 3.051282051282051, 4.274074074074074, 3.6390977443609023, 3.3157894736842106, 4.444444444444445, 4.432, 4.014814814814815, 3.753731343283582, 4.037313432835821])
                # window_behavior = window_behavior[:,np.newaxis]
                # window_behavior = [window_behavior]
            elif behavior_mode == 'eyemove':
                # window_behavior = [eye_prob]
                # window_behavior = np.array([4.058530510585305, 4.118020304568528, 3.00126582278481, 3.178826895565093, 4.058530510585305, 4.137055837563452, 3.081012658227848, 4.028465346534653, 3.962962962962963, 4.033415841584159, 3.894604767879548, 3.9406674907292953])
                window_behavior = np.array([0.44956413, 0.49873096, 0.49367089, 0.52360515, 0.46824408,
                                            0.47461929, 0.51012658, 0.5210396 , 0.39643347, 0.48886139,
                                            0.52697616, 0.45982695])  # iou_q =0.5
                window_behavior = window_behavior[:,np.newaxis]
                window_behavior = [window_behavior]
            elif behavior_mode == 'saliency':
                window_behavior = [w_saliency]
            else:
                raise ValueError('behavior_mode should be subtract, endogenous, or exogenous')
        concate_behavior.append(window_behavior[0])
    # import pdb; pdb.set_trace()
    concate_behavior = np.concatenate(concate_behavior, axis=1)

    # avg prob
    concate_behavior_avg = np.nanmean(concate_behavior, axis=1)
    return concate_behavior_avg



def get_behavior_genuine(fcdir:str, moving_window:int, moving_step:int, saliency_TR_q:int, iou_q:int,
                 use_saliency:bool=False, behavior_mode:str='subtract', num_segments:int=2):
    # select_mode: topk, auto, svd
    # static_fcs: grids x sbjs x tgt_rois
    # high_corr_idx_sbjs: sbj x tgt_rois x [tgt_voxels]
    SAVEDIR = fcdir # os.path.join(SAVE_ROOT, fc_mode_name)
    concate_behavior = []
    if isinstance(num_segments, int):
        num_segments = list(range(num_segments))
    for seg in num_segments:
        w_saliency_path = os.path.join(SAVEDIR, 'saliency', f'w_sal_vs_wo_sal_{moving_window}TR_{moving_step}TR-iou{1}-sal{saliency_TR_q}-w_sal-seg{seg}.npy')
        wo_saliency_path = os.path.join(SAVEDIR, 'saliency', f'w_sal_vs_wo_sal_{moving_window}TR_{moving_step}TR-iou{1}-sal{saliency_TR_q}-wo_sal-seg{seg}.npy')
        w_behavior_path = os.path.join(SAVEDIR, 'behavior', f'w_eye_w_sal_vs_wo_sal_{moving_window}TR_{moving_step}TR-iou{iou_q}-sal{saliency_TR_q}-w_sal-seg{seg}.npy')
        wo_behavior_path = os.path.join(SAVEDIR, 'behavior', f'w_eye_w_sal_vs_wo_sal_{moving_window}TR_{moving_step}TR-iou{iou_q}-sal{saliency_TR_q}-wo_sal-seg{seg}.npy')

        w_saliency = np.load(w_saliency_path)
        wo_saliency = np.load(wo_saliency_path)
        w_behavior = np.load(w_behavior_path)
        wo_behavior = np.load(wo_behavior_path)
        eye_prob = w_behavior + wo_behavior
        if use_saliency:
            window_behavior = [w_saliency]
        else:
            if behavior_mode == 'subtract':
                window_behavior = [(wo_behavior/wo_saliency)-(w_behavior/w_saliency)]
            elif behavior_mode == 'divide':
                window_behavior = [(wo_behavior/wo_saliency)/(w_behavior/w_saliency)]
            elif behavior_mode == 'endogenous':
                window_behavior = [(wo_behavior/wo_saliency)/eye_prob]
            elif behavior_mode == 'exogenous':
                window_behavior = [(w_behavior/w_saliency)/eye_prob]
            elif behavior_mode == 'endogenous2':
                window_behavior = [(wo_behavior/wo_saliency)]
            elif behavior_mode == 'exogenous2':
                window_behavior = [(w_behavior/w_saliency)]
            elif behavior_mode == 'eyemove':
                window_behavior = [eye_prob]
            elif behavior_mode == 'saliency':
                window_behavior = [w_saliency]
            else:
                raise ValueError('behavior_mode should be subtract, endogenous, or exogenous')
        concate_behavior.append(window_behavior[0])
    # import pdb; pdb.set_trace()
    concate_behavior = np.concatenate(concate_behavior, axis=1)

    # avg prob
    concate_behavior_avg = np.nanmean(concate_behavior, axis=1)
    return concate_behavior_avg

def run_eval(fcdir:str, moving_window:int, moving_step:int, saliency_TR_q:int, iou_q:int,
             static_fcs_sbjs_grids:np.ndarray, grids:List[int], shuffle_times:int, seed:int=1126,
             use_saliency:bool=False, behavior_mode:str='subtract', use_fit:bool=True, num_segments:int=2):
    # select_mode: topk, auto, svd
    # static_fcs: grids x sbjs x tgt_rois
    # high_corr_idx_sbjs: sbj x tgt_rois x [tgt_voxels]

    # SAVEDIR = fcdir # os.path.join(SAVE_ROOT, fc_mode_name)
    # concate_behavior = []
    # if isinstance(num_segments, int):
    #     num_segments = list(range(num_segments))
    # for seg in num_segments:
    #     w_saliency_path = os.path.join(SAVEDIR, 'saliency', f'w_sal_vs_wo_sal_{moving_window}TR_{moving_step}TR-iou{1}-sal{saliency_TR_q}-w_sal-seg{seg}.npy')
    #     wo_saliency_path = os.path.join(SAVEDIR, 'saliency', f'w_sal_vs_wo_sal_{moving_window}TR_{moving_step}TR-iou{1}-sal{saliency_TR_q}-wo_sal-seg{seg}.npy')
    #     w_behavior_path = os.path.join(SAVEDIR, 'behavior', f'w_eye_w_sal_vs_wo_sal_{moving_window}TR_{moving_step}TR-iou{iou_q}-sal{saliency_TR_q}-w_sal-seg{seg}.npy')
    #     wo_behavior_path = os.path.join(SAVEDIR, 'behavior', f'w_eye_w_sal_vs_wo_sal_{moving_window}TR_{moving_step}TR-iou{iou_q}-sal{saliency_TR_q}-wo_sal-seg{seg}.npy')

    #     w_saliency = np.load(w_saliency_path)
    #     wo_saliency = np.load(wo_saliency_path)
    #     w_behavior = np.load(w_behavior_path)
    #     wo_behavior = np.load(wo_behavior_path)
    #     eye_prob = w_behavior + wo_behavior
    #     if use_saliency:
    #         window_behavior = [w_saliency]
    #     else:
    #         if behavior_mode == 'subtract':
    #             window_behavior = [(wo_behavior/wo_saliency)-(w_behavior/w_saliency)]
    #         elif behavior_mode == 'divide':
    #             window_behavior = [(wo_behavior/wo_saliency)/(w_behavior/w_saliency)]
    #         elif behavior_mode == 'endogenous':
    #             window_behavior = [(wo_behavior/wo_saliency)/eye_prob]
    #         elif behavior_mode == 'exogenous':
    #             window_behavior = [(w_behavior/w_saliency)/eye_prob]
    #         elif behavior_mode == 'endogenous2':
    #             window_behavior = [(wo_behavior/wo_saliency)]
    #         elif behavior_mode == 'exogenous2':
    #             window_behavior = [(w_behavior/w_saliency)]
    #         elif behavior_mode == 'eyemove':
    #             window_behavior = [eye_prob]
    #         else:
    #             raise ValueError('behavior_mode should be subtract, endogenous, or exogenous')
    #     concate_behavior.append(window_behavior[0])

    # concate_behavior = np.concatenate(concate_behavior, axis=1)

    # # avg prob
    # concate_behavior_avg = np.nanmean(concate_behavior, axis=1)
    n_sbjs = len(static_fcs_sbjs_grids[0])
    assert n_sbjs  == len(SUBJECT_IDS), f'n_sbjs should be equal to len(SUBJECT_IDS), {n_sbjs} != {len(SUBJECT_IDS)}'
    if  'partial_corr' in behavior_mode:
        num_tgt_rois = static_fcs_sbjs_grids.shape[-1]
        num_grids = len(static_fcs_sbjs_grids)
        assert num_tgt_rois == 1
        assert num_grids == 1
        concate_endogenous2_avg = get_behavior(fcdir, moving_window, moving_step, saliency_TR_q, iou_q,
                                            use_saliency, 'endogenous2', num_segments)
        concate_exogenous2_avg = get_behavior(fcdir, moving_window, moving_step, saliency_TR_q, iou_q,
                                            use_saliency, 'exogenous2', num_segments)
        concate_eyemove_avg = get_behavior(fcdir, moving_window, moving_step, saliency_TR_q, iou_q,
                                            use_saliency, 'eyemove', num_segments)
        # import pdb; pdb.set_trace()
        # eye = np.array([0.50311333, 0.49492386, 0.51772152, 0.45636624, 0.46699875, 0.4251269 , 0.49113924, 0.43193069, 0.43758573, 0.52846535,0.51693852, 0.4499382 ])
        # print(np.corrcoef(concate_endogenous2_avg, eye)[0,1])
        # partial correlaton
        # df = pd.DataFrame({'fc': static_fcs_sbjs_grids[0,:,0], 'endogenous2': concate_endogenous2_avg, 'exogenous2': concate_exogenous2_avg, 'eyemove': concate_eyemove_avg})
        target_behavior_mode = behavior_mode.replace('partial_corr_', '')

        concate_target_avg = get_behavior(fcdir, moving_window, moving_step, saliency_TR_q, iou_q,
                                            use_saliency, target_behavior_mode, num_segments)
        df = pd.DataFrame({'fc': static_fcs_sbjs_grids[0,:,0], target_behavior_mode: concate_target_avg, 'eyemove': concate_eyemove_avg})

        # partial_corr_matrix = pg.pcorr(df)
        # test_corr = partial_corr_matrix['fc'][behavior_mode.replace('partial_corr_', '')]
        test_corr = pg.partial_corr(data=df, x='fc', y=target_behavior_mode,
                     covar='eyemove', x_covar=None, y_covar=None)['r'].item()
        df = pd.DataFrame({'fc': static_fcs_sbjs_grids[0,:,0], 'endogenous2': concate_endogenous2_avg,
                           'exogenous2': concate_exogenous2_avg, 'eyemove': concate_eyemove_avg})
        # partial_corr_matrix = pg.pcorr(df)
        # test_corr = partial_corr_matrix['fc'][target_behavior_mode]
        if target_behavior_mode == 'endogenous2':
            control_target_behavior_mode = 'exogenous2'
        elif target_behavior_mode == 'exogenous2':
            control_target_behavior_mode = 'endogenous2'
        # test_corr = pg.partial_corr(data=df, x='fc', y=target_behavior_mode,
        #              covar=None, x_covar=None, y_covar=[control_target_behavior_mode])['r'].item()
        test_corr = pg.partial_corr(data=df, x='fc', y=target_behavior_mode,
                     covar='eyemove', x_covar=None, y_covar=None)['r'].item()
        # import pdb; pdb.set_trace()
        # pg.partial_corr(data=df, x='endogenous2', y='eyemove',covar=None, x_covar='exogenous2', y_covar=None)['r'].item()
        # test_corr = result__['r']
        # df['endogenous2'] = concate_endogenous2_avg
        # df['exogenous2'] = concate_exogenous2_avg
        # print(pg.partial_corr(data=df, x='endogenous2', y='eyemove'))
        # print(pg.partial_corr(data=df, x='exogenous2', y='eyemove'))
        # print(pg.partial_corr(data=df, x='exogenous2', y='endogenous2'))
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()


        test_error = -1
        test_coef_mean = -1
        test_coef_std = -1
        concate_behavior_avg = None
        # fig = plt.figure(figsize=(10,8))
        # plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
        # plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます。
        # plt.rcParams['xtick.labelsize'] = 20 # 軸だけ変更されます。
        # plt.rcParams['ytick.labelsize'] = 20 # 軸だけ変更されます
        # plt.scatter(static_fcs_sbjs_grids[0,:,0], concate_target_avg, color='black', s=60)
        # figpath = os.path.join(SAVE_ROOT, 'fc6', f'{target_behavior_mode}-scatter.png')
        # plt.xlabel('FC')
        # if 'endogenous' in target_behavior_mode:
        #     plt.ylabel('Prob. of voluntary gaze shift')
        # if 'exogenous' in target_behavior_mode:
        #     plt.ylabel('Prob. of involuntary gaze shift')
        # plt.savefig(figpath)
        # plt.close()

        # plt.scatter(static_fcs_sbjs_grids[0,:,0], concate_eyemove_avg, color='black', s=60)
        # figpath = os.path.join(SAVE_ROOT, 'fc6', f'eyemove-scatter.png')
        # plt.xlabel('FC')
        # plt.ylabel('Prob. of gaze shift')
        # plt.savefig(figpath)
        # plt.close()
    else:
        concate_behavior_avg = get_behavior(fcdir, moving_window, moving_step, saliency_TR_q, iou_q,
                                            use_saliency, behavior_mode, num_segments)
        test_error, test_corr, test_coef_mean, test_coef_std = fit_behavior_on_staticFC(static_fcs_sbjs_grids, concate_behavior_avg, grids, use_fit=use_fit)
        # test_error: scalar
        # test_corr: scalar
        # test_coef_mean: scalar
        # test_coef_std: scalar
        # import pdb; pdb.set_trace()


    np.random.seed(seed)
    surrogate_errors = []
    surrogate_corrs = []
    sbj_indices = np.arange(n_sbjs)
    for i in tqdm(range(shuffle_times)):
        shuffle_sbj_indices = np.random.permutation(sbj_indices)
        if  'partial_corr' in behavior_mode:
            concate_endogenous2_avg_shuffled = concate_endogenous2_avg[shuffle_sbj_indices]
            concate_exogenous2_avg_shuffled = concate_exogenous2_avg[shuffle_sbj_indices]
            concate_eyemove_avg_shuffled = concate_eyemove_avg[shuffle_sbj_indices]
            # partial correlaton
            df = pd.DataFrame({'fc': static_fcs_sbjs_grids[0,:,0], 'endogenous2': concate_endogenous2_avg_shuffled, 'exogenous2': concate_exogenous2_avg_shuffled, 'eyemove': concate_eyemove_avg_shuffled})
            target_behavior_mode = behavior_mode.replace('partial_corr_', '')
            concate_target_avg_shuffled = concate_target_avg[shuffle_sbj_indices]
            # test_corr = np.corrcoef(test_Ys, test_preds)[0,1]
            # partial_corr_matrix = pg.pcorr(df)
            # shuffle_test_corr = partial_corr_matrix['fc'][behavior_mode.replace('partial_corr_', '')]
            if target_behavior_mode == 'endogenous2':
                control_target_behavior_mode = 'exogenous2'
            elif target_behavior_mode == 'exogenous2':
                control_target_behavior_mode = 'endogenous2'
            # shuffle_test_corr = pg.partial_corr(data=df, x='fc', y=behavior_mode.replace('partial_corr_', ''),
            #          covar=None, x_covar=None, y_covar=control_target_behavior_mode)['r'].item()
            shuffle_test_corr = pg.partial_corr(data=df, x='fc', y=behavior_mode.replace('partial_corr_', ''),
                     covar='eyemove', x_covar=None, y_covar=None)['r'].item()
            shuffle_test_error = -1
        else:
            concate_behavior_avg_shuffled = concate_behavior_avg[shuffle_sbj_indices]
            shuffle_test_error, shuffle_test_corr,_,_ = fit_behavior_on_staticFC(static_fcs_sbjs_grids, concate_behavior_avg_shuffled, grids, use_fit=use_fit)
        surrogate_errors.append(shuffle_test_error)
        surrogate_corrs.append(shuffle_test_corr)
    # import pdb; pdb.set_trace()
    return test_error, test_corr, surrogate_errors, surrogate_corrs, concate_behavior_avg, test_coef_mean, test_coef_std

def get_prediciton(src_roi:int, tgt_rois:List[int], n_components_bold_src:int,
                   sampling_mode:str, delay:int=0, pca:bool=True, pca_str:str=None,
                   use_dummy_for_pred:bool=False):
    assert sampling_mode == 'segment', 'sampling_mode should be segment'
    assert delay == 0
    pred_sbjs = []
    gt_sbjs = []
    high_corr_idx_sbjs =[]
    coef_sbjs = []
    src_sbjs = []
    if pca_str is None:
        pca_str = 'pca' if pca else 'nopca'
    for sbj in SUBJECT_IDS:
        fc_dir = os.path.join(SAVE_ROOT, 'static_fc', pca_str, f'delay{delay}',
                                    f'{sampling_mode}-ncb{n_components_bold_src}', sbj)
        if not os.path.exists(fc_dir):
            fc_dir = os.path.join('../', fc_dir) # for jupyter
        if not os.path.exists(fc_dir):
            fc_dir = os.path.join('../', fc_dir) # for jupyter
        pred_sbj = []
        gt_sbj = []
        high_corr_idx_sbj = []
        coef_sbj = []
        src_sbj = []
        for t, tgt_roi in enumerate(tgt_rois):
            gt = np.load(os.path.join(fc_dir, f'gt-roi_tgt{tgt_roi}-test.npy')) # n_samples x n_voxels
            gt_sbj.append(gt)
            if use_dummy_for_pred:
                pred=None
                pred_sbj.append(pred)
                high_corr_idx = None
                high_corr_idx_sbj.append(high_corr_idx)
                coef = None
                coef_sbj.append(coef)
            else:
                pred = np.load(os.path.join(fc_dir, f'pred-roi_tgt{tgt_roi}-src{src_roi}-test.npy')) # n_samples x n_voxels
                pred_sbj.append(pred)
                high_corr_idx = np.load(os.path.join(fc_dir, f'pred-roi_tgt{tgt_roi}-src{src_roi}-top_train_corr_idx.npy'))
                high_corr_idx_sbj.append(high_corr_idx)
                pca_loadings = np.load(os.path.join(fc_dir, f'pca_loadings-roi_src{src_roi}.npy')) # (250, 1171)
                # # import pdb; pdb.set_trace()
                # src_gt = src_gt@pca_loadings.T # (815, 250)
                # src_gt = add_bias(src_gt, axis=1) # (815, 251)
                coef = np.load(os.path.join(fc_dir, f'pred-roi_tgt{tgt_roi}-src{src_roi}-coef.npy')) # (251, 1059)
                coef_sbj.append(coef)

        src_gt = np.load(os.path.join(fc_dir, f'gt-roi_src{src_roi}.npy')) # (815, 1171)
        src_sbj.append(src_gt) # predとsrc_gt@coefの結果は多少ズレる
        pred_sbjs.append(pred_sbj)
        gt_sbjs.append(gt_sbj)
        high_corr_idx_sbjs.append(high_corr_idx_sbj) # sbj x n_tgt_rois x n_voxels
        coef_sbjs.append(coef_sbj)
        src_sbjs.append(src_sbj)
    return src_sbjs, pred_sbjs, gt_sbjs, high_corr_idx_sbjs, coef_sbjs # sbj x n_tgt_rois x [n_samples x n_voxels]

def clip_data(data:np.ndarray, k:int=1):
    # data: n_samples x n_features
    # upper = np.mean(data, axis=0,keepdims=True) + k * np.std(data, axis=0,keepdims=True)
    # lower = np.mean(data, axis=0,keepdims=True) - k * np.std(data, axis=0,keepdims=True)
    # data[data>upper] = upper[data>upper]
    # data[data<lower] = lower[data<lower]
    upper = np.mean(data) + k * np.std(data)
    lower = np.mean(data) - k * np.std(data)
    data = np.clip(data, lower, upper)
    return data


def prepare_bold_dataset(sbj_id:str, delay:int, sampling_mode:str, rois=None)->Tuple[List[np.ndarray],List[np.ndarray], np.ndarray, np.ndarray]:
    # load bold
    bolds = bold44_dateset(sbj_id, rois=rois) # n_rois x time x n_voxel
        # load label
    label_train, label_test = get_non_overlap_indices_for_concatenate_data(sampling_mode)

    label_train_tgt, label_train_src = delayed_label(label_train, delay)
    label_test_tgt, label_test_src = delayed_label(label_test, delay)
    assert len(label_train_src) == len(label_train_tgt), 'label_train_bold:{} vs label_train:{}'.format(len(label_train_src), len(label_train_tgt))
    assert len(label_test_src) == len(label_test_tgt), 'label_test_bold:{} vs label_test:{}'.format(len(label_test_src), len(label_test_tgt))
    if sampling_mode == 'sandwitch':
        assert min(min(label_train_tgt), min(label_train_src)) == 0
        assert max(max(label_train_tgt), max(label_train_src)) == len(bolds[0])-1
    # labeling
    bold_train_src = [b[label_train_src, :] for b in bolds]
    bold_test_src = [b[label_test_src, :] for b in bolds]
    bold_train_tgt = [b[label_train_tgt, :] for b in bolds]
    bold_test_tgt = [b[label_test_tgt, :] for b in bolds]


    return bold_train_src, bold_test_src, bold_train_tgt, bold_test_tgt

def segments_flags_volume()->Tuple[List[int], List[int]]:
    train_segments = [1]*(RUN_VOLUMES[0]-5) + [2]*(RUN_VOLUMES[1]-8) + [3]*(RUN_VOLUMES[2]-8) + [5]*(RUN_VOLUMES[4]-8) + [6]*(RUN_VOLUMES[5]-8) + [7]*(RUN_VOLUMES[6]-8)
    test_segments = [4]*(RUN_VOLUMES[3]-8) + [8]*(RUN_VOLUMES[7]-3)
    return train_segments, test_segments