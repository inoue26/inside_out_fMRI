from __future__ import print_function

import numpy as np
import torch
from slir import SparseLinearRegression
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from himalaya.backend import set_backend
from himalaya.ridge import Ridge
import torch.multiprocessing as mp
import time
from tqdm import tqdm
from shapreg import removal, games, shapley,stochastic_games
import numpy as np
from sklearn.linear_model import LinearRegression
from shapreg.utils import crossentropyloss, mseloss

if mp.get_start_method() == 'fork':
    mp.set_start_method('spawn', force=True)
    print("{} setup done".format(mp.get_start_method()))


# Functions ############################################################

def gpu_corr_coeff(x, y, device):
    # x: (n_sample, n_voxel)
    # y: (n_sample, )
    # return (n_voxel,)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    x = x.to(device)
    y = y.to(device)

    x = x - torch.mean(x, dim=0)
    y = y - torch.mean(y, dim=0)

    cosin_similarity = torch.nn.CosineSimilarity(dim=0)(x, y.unsqueeze(1))
    return cosin_similarity.cpu().numpy()

def ridge_rigression_gpu(x_train_unit, x_test_unit, y_train_unit, device, beta=None, return_model=False):
    '''Run regression
    '''
    force_cpu = False if device == 'cuda' else True
    model = Ridge(alpha=beta, force_cpu=force_cpu)
    if device == 'cuda':
        x_train = torch.Tensor(x_train_unit).to(device)
        x_test = torch.Tensor(x_test_unit).to(device)
        y_train = torch.Tensor(y_train_unit).to(device)
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    # import pdb; pdb.set_trace()
    if return_model:
        return y_pred_test, y_pred_train, model
    else:
        return y_pred_test, y_pred_train


def feature_prediction(x_train, y_train, x_test, y_test, n_voxel=500, n_iter=200, use_gpu=True, beta=None, return_voxel=False):
    '''Run feature prediction

    Parameters
    ----------
    x_train, y_train : array_like [shape = (n_sample, n_voxel)]
        Brain data and image features for training
    x_test, y_test : array_like [shape = (n_sample, n_unit)]
        Brain data and image features for test
    n_voxel : int
        The number of voxels
    n_iter : int
        The number of iterations

    Returns
    -------
    predicted_label : array_like [shape = (n_sample, n_unit)]
        Predicted features
    ture_label : array_like [shape = (n_sample, n_unit)]
        True features in test data
    '''

    n_unit = y_train.shape[1]

    # Normalize brian data (x)
    norm_mean_x = np.mean(x_train, axis=0)
    norm_scale_x = np.std(x_train, axis=0, ddof=1)

    x_train = (x_train - norm_mean_x) / norm_scale_x
    x_test = (x_test - norm_mean_x) / norm_scale_x

    # Feature prediction for each unit
    print('Running feature prediction')

    y_true_test_list = []
    y_true_train_list = []
    y_pred_test_list = []
    y_pred_train_list = []
    voxel_index_list = []

    i_start = 0
    i_end = n_unit # 30 # n_unit

    print('start id:: ', i_start, '\tend id:: ', i_end)
    with tqdm(range(i_start, i_end)) as pbar:
        for i in pbar:

            # print('Unit {} / {}'.format((i + 1), n_unit))
            start_time = time.time()

            # Get unit features
            y_train_unit = y_train[:, i]
            y_test_unit =  y_test[:, i]

            # Normalize image features for training (y_train_unit)
            norm_mean_y = np.mean(y_train_unit, axis=0)
            std_y = np.std(y_train_unit, axis=0, ddof=1)
            norm_scale_y = 1 if std_y == 0 else std_y

            y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y
            if use_gpu:
                device='cuda'
                corr = gpu_corr_coeff(x_train, y_train_unit, device)

            else:
                # Voxel selection
                corr = corrcoef(y_train_unit, x_train, var='col') # 2804, (2804,14062) -> 14062
            if n_voxel > 0:
                x_train_unit, voxel_index = select_top(x_train, np.abs(corr), n_voxel, axis=1, verbose=False)
                x_test_unit = x_test[:, voxel_index]
                voxel_index_list.append(voxel_index)

            else:
                x_train_unit = x_train
                x_test_unit = x_test
                voxel_index_list.append([np.nan])

            # Add bias terms
            x_train_unit = add_bias(x_train_unit, axis=1)
            x_test_unit = add_bias(x_test_unit, axis=1)

            if use_gpu:
                device='cuda'
                backend_type = 'torch_cuda' # numpy
                backend = set_backend(backend_type, on_error="warn")
                y_pred_test, y_pred_train = ridge_rigression_gpu(x_train_unit, x_test_unit, y_train_unit, device, beta=beta)

            else:
                # Setup regression
                # For quick demo, use linaer regression
                # model = LinearRegression()
                model = SparseLinearRegression(n_iter=n_iter, prune_mode=1)
                # Training and test
                try:
                    model.fit(x_train_unit, y_train_unit)  # Training
                    y_pred_test = model.predict(x_test_unit)  # (715,)# Test
                    y_pred_train = model.predict(x_train_unit) # (2804,)
                except:
                    # When SLiR failed, returns zero-filled array as predicted features
                    y_pred_test = np.zeros(y_test_unit.shape)
                    y_pred_train = np.zeros(y_train_unit.shape)

            # Denormalize predicted features
            y_pred_test = y_pred_test * norm_scale_y + norm_mean_y
            y_pred_train = y_pred_train * norm_scale_y + norm_mean_y

            y_true_test_list.append(y_test_unit)
            y_true_train_list.append(y_train_unit)
            y_pred_test_list.append(y_pred_test)
            y_pred_train_list.append(y_pred_train)

            pbar.set_description('Time: %.3f sec' % (time.time() - start_time))

    # Create numpy arrays for return values
    y_predicted_test = np.vstack(y_pred_test_list).T
    y_true_test = np.vstack(y_true_test_list).T

    y_predicted_train = np.vstack(y_pred_train_list).T
    y_true_train = np.vstack(y_true_train_list).T
    if return_voxel:
        return y_predicted_test, y_true_test, y_predicted_train, y_true_train, voxel_index_list
    else:
        return y_predicted_test, y_true_test, y_predicted_train, y_true_train




def mseloss_gpu(pred, target):
    '''MSE loss that does not average across samples.'''
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().detach().numpy()
    if len(pred.shape) == 1:
        pred = pred[:, np.newaxis]
    if len(target.shape) == 1:
        target = target[:, np.newaxis]
    loss =  np.sum((pred - target) ** 2, axis=1)
    # print(loss.shape)
    return loss

class ShapleyModel():
    def __init__(self, model):
        self.model = model
    def predict(self, x):
        return self.model.predict(x)
    def __call__(self, x):
        x = torch.from_numpy(x).cuda()
        y = self.predict(x)
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        return y

def feature_prediction_with_shapley(x_train, y_train, x_test, y_test, n_voxel=500, n_iter=200, use_gpu=True, beta=None):
    '''Run feature prediction

    Parameters
    ----------
    x_train, y_train : array_like [shape = (n_sample, n_voxel)]
        Brain data and image features for training
    x_test, y_test : array_like [shape = (n_sample, n_unit)]
        Brain data and image features for test
    n_voxel : int
        The number of voxels
    n_iter : int
        The number of iterations

    Returns
    -------
    predicted_label : array_like [shape = (n_sample, n_unit)]
        Predicted features
    ture_label : array_like [shape = (n_sample, n_unit)]
        True features in test data
    '''

    n_unit = y_train.shape[1]

    # Normalize brian data (x)
    norm_mean_x = np.mean(x_train, axis=0)
    norm_scale_x = np.std(x_train, axis=0, ddof=1)

    x_train = (x_train - norm_mean_x) / norm_scale_x
    x_test = (x_test - norm_mean_x) / norm_scale_x

    # Feature prediction for each unit
    print('Running feature prediction')

    y_true_test_list = []
    y_true_train_list = []
    y_pred_test_list = []
    y_pred_train_list = []
    voxel_index_list = []
    explanation_list = []
    explanation_std_list = []

    i_start = 0
    i_end = n_unit # 30 # n_unit

    print('start id:: ', i_start, '\tend id:: ', i_end)
    with tqdm(range(i_start, i_end)) as pbar:
        for i in pbar:

            # print('Unit {} / {}'.format((i + 1), n_unit))
            start_time = time.time()

            # Get unit features
            y_train_unit = y_train[:, i]
            y_test_unit =  y_test[:, i]

            # Normalize image features for training (y_train_unit)
            norm_mean_y = np.mean(y_train_unit, axis=0)
            std_y = np.std(y_train_unit, axis=0, ddof=1)
            norm_scale_y = 1 if std_y == 0 else std_y

            y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y
            if use_gpu:
                device='cuda'
                corr = gpu_corr_coeff(x_train, y_train_unit, device)

            else:
                # Voxel selection
                corr = corrcoef(y_train_unit, x_train, var='col') # 2804, (2804,14062) -> 14062
            if n_voxel > 0:
                x_train_unit, voxel_index = select_top(x_train, np.abs(corr), n_voxel, axis=1, verbose=False)
                x_test_unit = x_test[:, voxel_index]
                voxel_index_list.append(voxel_index)

            else:
                x_train_unit = x_train
                x_test_unit = x_test
                voxel_index_list.append([np.nan])

            # Add bias terms
            x_train_unit = add_bias(x_train_unit, axis=1)
            x_test_unit = add_bias(x_test_unit, axis=1)

            if use_gpu:
                device='cuda'
                backend_type = 'torch_cuda' # numpy
                backend = set_backend(backend_type, on_error="warn")
                y_pred_test, y_pred_train, model = ridge_rigression_gpu(x_train_unit, x_test_unit, y_train_unit, device, beta=beta, return_model=True)

            else:
                # Setup regression
                # For quick demo, use linaer regression
                # model = LinearRegression()
                model = SparseLinearRegression(n_iter=n_iter, prune_mode=1)
                # Training and test
                try:
                    model.fit(x_train_unit, y_train_unit)  # Training
                    y_pred_test = model.predict(x_test_unit)  # (715,)# Test
                    y_pred_train = model.predict(x_train_unit) # (2804,)
                except:
                    # When SLiR failed, returns zero-filled array as predicted features
                    y_pred_test = np.zeros(y_test_unit.shape)
                    y_pred_train = np.zeros(y_train_unit.shape)

            # Denormalize predicted features
            y_pred_test = y_pred_test * norm_scale_y + norm_mean_y
            y_pred_train = y_pred_train * norm_scale_y + norm_mean_y

            y_true_test_list.append(y_test_unit)
            y_true_train_list.append(y_train_unit)
            y_pred_test_list.append(y_pred_test)
            y_pred_train_list.append(y_pred_train)

            # shapley
            s_model = ShapleyModel(model)
            imputer = removal.MarginalExtension(x_train_unit, s_model)
            game = stochastic_games.DatasetOutputGame(imputer, x_test_unit, mseloss_gpu)
            explanation = shapley.ShapleyRegression(game)
            explanation_list.append(explanation.values)
            explanation_std_list.append(explanation.std)
            pbar.set_description('Time: %.3f sec' % (time.time() - start_time))


    # Create numpy arrays for return values
    y_predicted_test = np.vstack(y_pred_test_list).T
    y_true_test = np.vstack(y_true_test_list).T

    y_predicted_train = np.vstack(y_pred_train_list).T
    y_true_train = np.vstack(y_true_train_list).T

    return y_predicted_test, y_true_test, y_predicted_train, y_true_train, voxel_index_list, explanation_list, explanation_std_list




